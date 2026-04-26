import json
import logging
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
import requests
from .explain_parser import ExplainParser


def _handle_profile(profile_data, context):
    logging.info("=== IMPALA QUERY PROFILE ===")
    logging.info(profile_data)


class ImpalaExecuteOperator(SQLExecuteQueryOperator):
    def __init__(self, *, sql: str | list[str], configurations=None, ml_service_url=None, **kwargs):
        super().__init__(sql=sql, **kwargs)
        self.configurations = configurations
        self.split_statements = True
        self.ml_service_url = ml_service_url

    def get_session_params(self, cursor, query):
        ep = ExplainParser()
        q = f"explain {query}"
        try:
            cursor.execute(q)
            ep_res = cursor.fetchall()
            features = ep.parse(ep_res)
        except Exception as e:
            features = None
            logging.error(f"Extract plan features failed. Error: {e}")

        if features and self.ml_service_url is not None:
            try:
                response = requests.post(
                    f"{self.ml_service_url}/predict",
                    json=features,
                    timeout=5
                )
                response.raise_for_status()
                params = response.json()
                logging.info(f"get ml {params}")
            except Exception as e:
                params = None
                logging.error(f"ML Service unavailable, using default settings. Error: {e}")
        return params

    def execute(self, context):
        params = self.configurations

        set_statements = ""
        hook = self.get_db_hook()
        with hook.get_conn() as conn:
            with conn.cursor() as cursor:
                logging.info("Executing query on Impala...")
                ml_params = self.get_session_params(cursor, self.sql)
                if ml_params is not None:
                    params = ml_params
                if params is not None:
                    set_statements = "\n".join([f"SET {k}={v};\n" for k, v in params.items()])
                logging.info(params)
                self.sql = set_statements + self.sql
                queries = self.sql.split(";\n")
                for query in queries:
                    logging.info(f"Running: {query}")
                    try:
                        cursor.execute(query)
                        logging.info(cursor.fetchall())
                    except Exception as e:
                        logging.warning(f"query failed: {e}")
                profile_data = None
                try:
                    profile_data = cursor.get_profile(profile_format=3)
                except Exception as e:
                    logging.warning(e)
                _handle_profile(profile_data, context)
                try:
                    return cursor.fetchall()
                except Exception:
                    return None
