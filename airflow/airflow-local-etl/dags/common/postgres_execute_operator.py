import logging

from .ml_operator import BaseMLSQLOperator
from .parser.pg_plan_parser import PostgresPlanParser


class PostgresMLExecuteOperator(BaseMLSQLOperator):
    def __init__(self, *, sql: str, **kwargs):
        super().__init__(sql=sql, **kwargs)
        self.params_to_config = ['work_mem', 'max_parallel_workers_per_gather', 'jit', 'enable_indexscan',
                                 'enable_hashjoin']

    def _get_features(self, cursor, query: str) -> dict:
        try:
            logging.info("Fetching PG EXPLAIN plan...")
            cursor.execute(f"EXPLAIN (BUFFERS, VERBOSE, MEMORY, FORMAT JSON) {query}")
            raw_json = cursor.fetchone()[0]

            parser = PostgresPlanParser()
            features, metrics = parser.extract_features_and_metrics(raw_json)
            return features
        except Exception as e:
            logging.error(f"Failed to parse PG plan: {e}")
            return None

    def _apply_session_params(self, cursor, params: dict):
        for key, value in params.items():
            if key in self.params_to_config:
                if key == 'max_parallel_workers_per_gather' and value > 0:
                    cursor.execute("SET LOCAL min_parallel_table_scan_size = 0;")
                    cursor.execute("SET LOCAL parallel_setup_cost = 0;")
                    cursor.execute("SET LOCAL parallel_tuple_cost = 0;")
                cursor.execute(f"SET LOCAL {key} TO '{value}';")
