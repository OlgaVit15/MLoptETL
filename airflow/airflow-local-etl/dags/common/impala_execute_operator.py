import logging

from .parser.explain_parser import ExplainParser
from .ml_operator import BaseMLSQLOperator


class ImpalaMLExecuteOperator(BaseMLSQLOperator):
    def __init__(self, *, sql: str, **kwargs):
        super().__init__(sql=sql, **kwargs)
        self.params_to_config = ['mem_limit', 'mt_dop', 'num_scanner_threads', 'default_join_distribution_mode', 'disable_codegen']

    def _get_features(self, cursor, query: str) -> dict:
        try:
            logging.info("Fetching Impala EXPLAIN plan...")
            cursor.execute(f"EXPLAIN {query}")
            explain_results = cursor.fetchall()

            # Используем твой внешний парсер
            parser = ExplainParser()
            return parser.parse(explain_results)
        except Exception as e:
            logging.error(f"Failed to parse Impala plan: {e}")
            return None

    def _apply_session_params(self, cursor, params: dict):
        for key, value in params.items():
            # Специфичный синтаксис Impala
            if key in self.params_to_config:
                cursor.execute(f"SET {key}='{value}';")

    def _post_execute(self, cursor, context):
        """
        Специфичное для Impala получение профиля после запроса.
        """
        try:
            profile_data = cursor.get_profile(profile_format=3)
            logging.info("=== IMPALA QUERY PROFILE ===")
            logging.info(profile_data)
        except Exception as e:
            logging.warning(f"Could not fetch Impala profile: {e}")

        return super()._post_execute(cursor, context)
