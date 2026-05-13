import requests
import threading
import time
import logging


class ImpalaClusterCollector:
    def __init__(self, impala_coord_url, pool_name='default-pool', interval=10):
        self.impala_url = impala_coord_url
        self.pool_name = pool_name
        self.interval = interval
        self.snapshot = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _get_impala_data(self):
        """Опрос Координатора для получения метрик"""
        try:
            # 1. Данные Admission Control (Очереди и Память пула)
            adm_resp = requests.get(f"{self.impala_url}/admission?json", timeout=5).json()

            # 2. Общие метрики сервера (Активные запросы и Диски)
            met_resp = requests.get(f"{self.impala_url}/metrics?json", timeout=5).json()

            raw_metrics = met_resp.get('metric_group', {}).get('metrics', [])

            io_latencies = []
            for m in raw_metrics:
                # Ищем метрики read-latency для всех очередей (queue-0, queue-1, ...)
                if "io-mgr.queue" in m['name'] and "read-latency" in m['name']:
                    # Берем 75-й процентиль, лучше показывает "хвосты" задержек
                    # Значение в TIME_NS (наносекунды), переводим в миллисекунды
                    latency_ms = m.get('75th %-ile', 0) / 1_000_000
                    io_latencies.append(latency_ms)

            # Средняя задержка по всем дискам
            avg_io_latency = sum(io_latencies) / len(io_latencies) if io_latencies else 0

            # Превращаем метрики в словарь
            metrics = {m['name']: m['value'] for m in raw_metrics if 'value' in m}

            max_mem_pool = 0

            # Данные по пулу ресурсов
            pool_info = {'queued_queries': 0, 'admitted_mem_gb': 0, 'max_mem_gb': 1, 'live_nodes': 3}
            for pool in adm_resp.get('resource_pools', []):
                if pool['pool_name'] == self.pool_name:
                    pool_info = {
                        'queued_queries': pool.get('agg_num_queued', 0),
                        'admitted_mem_gb': pool.get('agg_mem_reserved', 0) / (1024 ** 3)
                    }
                    max_mem_pool = pool.get('pool_max_mem_resources', 1) / (1024 ** 3)

            # Метрики нагрузки
            live_nodes = metrics.get('cluster-membership.backends.total', 3)
            running_queries = metrics.get('impala-server.num-queries-registered', 0)
            node_limit_bytes = metrics.get('mem-tracker.process.limit', 0)
            if node_limit_bytes <= 0:
                # Если вдруг -1, попробуем взять физическую память (fallback)
                node_limit_bytes = metrics.get('memory.total-used', 64 * 1024 ** 3)

            max_mem_gb = max_mem_pool if max_mem_pool >= 0 else (node_limit_bytes * live_nodes) / (1024 ** 3)

            return {
                **pool_info,
                'max_mem_gb': max_mem_gb,
                'live_nodes': live_nodes,
                'running_queries': running_queries,
                'io_queue': avg_io_latency,
                'timestamp': time.time()
            }
        except Exception as e:
            logging.error(f"Collector Error: {e}")
            return None

    def _update_loop(self):
        while not self._stop_event.is_set():
            new_data = self._get_impala_data()
            if new_data:
                with self._lock:
                    self.snapshot = new_data
            time.sleep(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def get_current_state(self):
        with self._lock:
            if not self.snapshot:
                # Дефолты на случай, если сбор данных еще не прошел
                return {'running_queries': 0, 'io_queue': 0, 'live_nodes': 3,
                        'admitted_mem_gb': 0, 'max_mem_gb': 100, 'queued_queries': 0}
            return self.snapshot.copy()
