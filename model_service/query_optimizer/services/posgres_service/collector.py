import threading
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import logging


class PostgresStateCollector:
    def __init__(self, dsn, interval=10):
        self.dsn = dsn
        self.interval = interval
        self.state = {
            'running_queries': 0,
            'blocked_queries': 0,
            'active_connections_pct': 0,
            'load_avg': 0.0,
            'is_alive': False
        }
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)

    def start(self):
        self.thread.start()

    def _get_db_metrics(self):
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Исправленный запрос: используем split_part для получения первого числа (1 min load)
                query = """
                SELECT 
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active' AND backend_type = 'client backend') as running,
                    (SELECT count(*) FROM pg_stat_activity WHERE wait_event_type IS NOT NULL AND backend_type = 'client backend') as blocked,
                    (SELECT (count(*)::float / current_setting('max_connections')::float) * 100 FROM pg_stat_activity) as conn_p,
                    (SELECT split_part(pg_read_file('/proc/loadavg'), ' ', 1)::float) as load_1min;
                """
                cur.execute(query)
                res = cur.fetchone()
                return {
                    'running_queries': res['running'],
                    'blocked_queries': res['blocked'],
                    'active_connections_pct': res['conn_p'],
                    'load_avg': res['load_1min']
                }

    def _collect_loop(self):
        while not self._stop_event.is_set():
            try:
                new_state = self._get_db_metrics()
                self.state.update(new_state)
                self.state['is_alive'] = True
            except Exception as e:
                logging.error(f"Postgres Collector Error: {e}")
                self.state['is_alive'] = False
            time.sleep(self.interval)

    def get_current_state(self):
        return self.state
