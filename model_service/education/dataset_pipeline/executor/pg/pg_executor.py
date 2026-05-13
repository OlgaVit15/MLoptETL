import json
import logging
import time
import psycopg2


class PostgresExecutor:
    """Исполнитель запросов PostgreSQL с поддержкой параметров сессии и EXPLAIN ANALYZE"""

    def __init__(self, conn_params, db_schema):
        # conn_params должен быть словарем с ключами host, dbname, user, password, port
        self.conn_params = conn_params
        self.db_schema = db_schema

    def execute(self, sql, params=None):
        print(sql)
        conn = None
        cur = None
        try:
            # 1. Подключение
            conn = psycopg2.connect(**self.conn_params)
            conn.autocommit = False
            cur = conn.cursor()

            # 2. Установка схемы (search_path)
            cur.execute(f"SET search_path TO {self.db_schema}, public;")
            # cur.execute("set debug_parallel_query = on;")

            # 3. Применение настроек сессии
            if params:
                for key, value in params.items():
                    if key == 'max_parallel_workers_per_gather' and value > 0:
                        cur.execute("SET LOCAL min_parallel_table_scan_size = 0;")
                        cur.execute("SET LOCAL parallel_setup_cost = 0;")
                        cur.execute("SET LOCAL parallel_tuple_cost = 0;")
                    cur.execute(f"SET LOCAL {key} = %s;", (value,))

            # 4. Выполнение EXPLAIN ANALYZE
            # BUFFERS - обязателен для оценки I/O и Spill
            profile_sql = f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE, MEMORY, FORMAT JSON) {sql}"

            start_time = time.time()
            cur.execute(profile_sql)
            explain_result = cur.fetchone()[0][0]
            wall_duration = (time.time() - start_time) * 1000
            print(f"duration: {wall_duration}")

            if not explain_result:
                raise Exception("EXPLAIN ANALYZE failed to return a plan")

            return json.dumps(explain_result)

        except Exception as e:
            logging.error(f"Error executing query: {e}")
            if conn:
                conn.rollback()
            return json.dumps({"errors": str(e)})
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
