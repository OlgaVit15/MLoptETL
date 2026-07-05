import time
import json
import requests
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import errors

# Импорт твоего парсера
from pg_plan_parser import PostgresPlanParser
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT # Для автокоммита на временном соединении


# --- КОНФИГУРАЦИЯ ---
DSN = "host=localhost dbname=imdb user=imdb"
API_URL = "http://localhost:8001/predict"
QUERIES_PATH = "test_workload_tpcds_pg.csv"

ITERATIONS = 4
WARMUP = 1
QUERY_TIMEOUT_MS = 180000  # 3 минуты в миллисекундах
GLOBAL_MAX_TIME_SEC = 3 * 3600  # 3 часа



class ETLBenchmark:
    def __init__(self):
        self.parser = PostgresPlanParser()
        self.results = []
        self.start_time = time.time()

    def get_conn(self):
        conn = psycopg2.connect(DSN)
        conn.autocommit = True
        return conn

    def check_global_timeout(self):
        return (time.time() - self.start_time) > GLOBAL_MAX_TIME_SEC

    def _log_metrics(self, system, q_id, run_type, iteration, metrics, overhead=0, status="success"):
        row = {
            'system': system,
            'query_id': q_id,
            'run_type': run_type,
            'iteration': iteration,
            'overhead_ms': overhead,
            'duration_ms': metrics.get('duration_ms', 0),
            'total_duration_ms': metrics.get('total_duration_ms',0),
            'total_latency_ms': metrics.get('duration_ms', 0) + overhead,
            'temp_written_blocks': metrics.get('temp_written_blocks', 0),
            'status': status,
            'timestamp': time.time()
        }
        # Копируем предсказанные параметры если есть
        for k, v in metrics.items():
            if k.startswith('predicted_'): row[k] = v
        self.results.append(row)

    def run_system_baseline(self, queries):
        print("\n>>> [1/3] Запуск Baseline...")
        conn = self.get_conn()
        with conn.cursor() as cur:
            for q_id, sql in queries.items():
                if self.check_global_timeout(): break
                print(f"  Запрос {q_id}...", end=' ', flush=True)

                for i in range(ITERATIONS):
                    try:
                        cur.execute(f"SET statement_timeout = {QUERY_TIMEOUT_MS};")
                        cur.execute("SET search_path = tpcds_test;")

                        t0 = time.perf_counter()
                        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
                        plan_data = cur.fetchone()[0]
                        duration = int((time.perf_counter()-t0)*1000)
                        _, metrics = self.parser.extract_features_and_metrics(plan_data)
                        metrics['total_duration_ms'] = duration
                        run_type = 'warmup' if i < WARMUP else 'measured'
                        self._log_metrics('Baseline', q_id, run_type, i, metrics)
                    except Exception as e:
                        print(f"(Timeout/Error)", end='')
                        self._log_metrics('Baseline', q_id, 'failed', i, {}, status="timeout")
                        conn.rollback() if not conn.autocommit else None
                        break  # Пропускаем остальные итерации этого запроса
                print("Done.")
        conn.close()

    def _check_bao_compatibility(self, sql_query, timeout_sec=10):
        """
        Проверяет, не вызывает ли Bao "unrecognized node type" или зависание
        с помощью временного соединения и быстрого EXPLAIN.
        """
        temp_conn = None
        try:
            temp_conn = psycopg2.connect(DSN)
            # Устанавливаем автокоммит, чтобы SET команды не требовали COMMIT
            temp_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with temp_conn.cursor() as temp_cur:
                # Включаем Bao для проверки
                temp_cur.execute("SET pg_bao.enable_bao = on;")
                temp_cur.execute("SET pg_bao.enable_bao_selection = on;")
                temp_cur.execute("SET pg_bao.enable_bao_rewards = off;")  # Не нужно rewards для проверки
                temp_cur.execute("SET pg_bao.bao_num_arms = 2;")
                temp_cur.execute("SET pg_bao.bao_include_json_in_explain = off;")
                temp_cur.execute("SET search_path = tpcds_test;")

                # Короткий таймаут для EXPLAIN, чтобы быстро отловить зависания
                temp_cur.execute(f"SET statement_timeout = {timeout_sec * 1000};")

                # Делаем быстрый EXPLAIN без ANALYZE
                temp_cur.execute(f"EXPLAIN (FORMAT JSON) {sql_query}")
                temp_cur.fetchone()  # Пытаемся получить результат
                return True  # Если дошли сюда, значит Bao смог спланировать
        except (errors.QueryCanceled, Exception) as e:
            # Если возникла ошибка (включая unrecognized node type) или таймаут
            # print(f"  Bao compatibility check failed: {type(e).__name__} - {e}") # Для отладки
            return False
        finally:
            if temp_conn:
                temp_conn.close()  # Важно: закрытие соединения "убивает" зависший бэкенд

    def run_system_bao(self, train_queries, main_queries):
        print("\n>>> [2/3] Запуск Bao (Безопасный режим)...")

        # 1. ФАЗА ОБУЧЕНИЯ (Exploration)
        t_start_setup = time.time()
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                print("  Bao: Конфигурация для обучения...")
                cur.execute("RESET ALL;")
                cur.execute("SET pg_bao.enable_bao = on;")
                cur.execute("SET pg_bao.enable_bao_selection = on;")
                cur.execute("SET pg_bao.enable_bao_rewards = on;")  # Rewards ON для обучения
                cur.execute("SET pg_bao.bao_num_arms = 2;")
                cur.execute("SET pg_bao.bao_include_json_in_explain = off;")
                cur.execute("SET search_path = tpcds_test;")  # Устанавливаем search_path

                for loop in range(3):
                    if self.check_global_timeout():
                        print("  Global timeout reached. Skipping remaining Bao training loops.")
                        break
                    print(f"    Bao Train Loop {loop + 1}/3...")
                    for q_id, sql in train_queries.items():
                        # Пропускаем запросы, если Bao несовместим
                        if not self._check_bao_compatibility(sql):
                            print(f"    Запрос {q_id} Skipped (Incompatible Bao during Training).", flush=True)
                            continue

                        try:
                            cur.execute(f"SET statement_timeout = {QUERY_TIMEOUT_MS};")
                            cur.execute(sql)
                            cur.fetchall()
                        except (errors.QueryCanceled, Exception) as e:
                            print(f" (Bao Training Error: {type(e).__name__})", end='')
                            conn.rollback()
                            continue
        finally:
            conn.close()

        setup_time_ms = (time.time() - t_start_setup) * 1000
        print(f"  Bao: Обучение завершено за {setup_time_ms / 1000:.2f} сек.")

        # 2. ФАЗА ЗАМЕРА (Inference)
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                print("  Bao: Конфигурация для замера...")
                cur.execute("RESET ALL;")
                cur.execute("SET pg_bao.enable_bao = on;")
                cur.execute("SET pg_bao.enable_bao_selection = on;")
                cur.execute("SET pg_bao.enable_bao_rewards = off;")  # Rewards OFF для замера
                cur.execute("SET pg_bao.bao_num_arms = 2;")
                cur.execute("SET pg_bao.bao_include_json_in_explain = off;")
                cur.execute("SET search_path = tpcds_test;")  # Устанавливаем search_path

                for q_id, sql in main_queries.items():
                    if self.check_global_timeout():
                        print("  Global timeout reached. Skipping remaining Bao measurements.")
                        break
                    print(f"  Запрос {q_id} [Bao Measurement]...", end=' ', flush=True)

                    # ПЕРВЫЙ ШАГ: Быстрая проверка на совместимость перед началом итераций
                    if not self._check_bao_compatibility(sql):
                        print(f"Skipped (Incompatible SQL).", flush=True)
                        self._log_metrics('Bao', q_id, 'failed', 0, {'setup_time_ms': setup_time_ms},
                                          status="incompatible")
                        continue  # Переходим к следующему q_id

                    # ВТОРОЙ ШАГ: Если запрос совместим, запускаем итерации
                    for i in range(ITERATIONS):
                        try:
                            cur.execute(f"SET statement_timeout = '{QUERY_TIMEOUT_MS}ms';")
                            st = time.perf_counter()
                            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
                            plan_raw = cur.fetchone()[0]
                            duration = int((time.perf_counter() - st) * 1000)
                            _, metrics = self.parser.extract_features_and_metrics(plan_raw, 'Bao')
                            metrics['setup_time_ms'] = setup_time_ms
                            metrics['total_duration_ms'] = duration

                            run_type = 'warmup' if i < WARMUP else 'measured'
                            self._log_metrics('Bao', q_id, run_type, i, metrics)

                        except (errors.QueryCanceled, Exception) as e:
                            print(f"(T)", end='')
                            self._log_metrics('Bao', q_id, 'failed', i,
                                              {'setup_time_ms': setup_time_ms, 'duration_ms': QUERY_TIMEOUT_MS},
                                              status="timeout")
                            conn.rollback()
                            break
                    print("Done.")
        finally:
            try:
                with conn.cursor() as cur:  # Отдельный курсор для очистки настроек
                    cur.execute("SET pg_bao.enable_bao = off;")
                    cur.execute("SET pg_bao.enable_bao_selection = off;")
                    cur.execute("SET pg_bao.enable_bao_rewards = off;")
            except Exception as e:
                print(f"  Warning: Could not disable Bao settings: {e}")
            conn.close()

    def run_system_my_ml(self, ml_queries):
        print("\n>>> [3/3] Запуск My ML System...")
        conn = self.get_conn()
        with conn.cursor() as cur:
            for q_id, sql in ml_queries.items():
                if self.check_global_timeout(): break
                print(f"  Запрос {q_id}...", end=' ', flush=True)
                for i in range(ITERATIONS):
                    try:
                        # print(f"iteration {i}")
                        cur.execute("RESET ALL;")
                        cur.execute(f"SET statement_timeout = {QUERY_TIMEOUT_MS};")
                        cur.execute("SET search_path = tpcds_test;")

                        # Инференс
                        t_ov = time.perf_counter()
                        start = time.perf_counter()
                        cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
                        raw_plan = cur.fetchone()[0]
                        features, _ = self.parser.extract_features_and_metrics(raw_plan)
                        features = {"feature_" + str(k): v for k, v in features.items()}
                        # print(features)
                        overhead = 0
                        try:
                            # print("post api")
                            resp = requests.post(API_URL, json=features, timeout=12)
                            # print(f"get {resp}")
                            if resp.status_code == 200:
                                p = resp.json()
                                # print(f"get {p}")
                                cur.execute(f"SET jit = {p.get('jit', 'on')};")
                                cur.execute(f"SET work_mem = '{p.get('work_mem_mb', 4)}MB';")
                                cur.execute(f"SET enable_indexscan = {p.get('enable_indexscan', 'on')};")
                                cur.execute(f"SET enable_hashjoin = {p.get('enable_hashjoin', 'on')};")
                                cur.execute(f"SET max_parallel_workers_per_gather = {p.get('max_parallel_workers_per_gather', 1)};")
                        except:
                            pass
                        overhead = int((time.perf_counter() - t_ov) * 1000)

                        # Замер
                        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
                        plan_data = cur.fetchone()[0]
                        duration = int((time.perf_counter()-start)*1000)
                        _, metrics = self.parser.extract_features_and_metrics(plan_data)
                        metrics['total_duration_ms'] = duration
                        self._log_metrics('MyModule', q_id, 'warmup' if i < WARMUP else 'measured', i, metrics,
                                          overhead=overhead)
                    except:
                        self._log_metrics('MyModule', q_id, 'failed', i, {}, status="timeout")
                        break
                print("Done.")
        conn.close()

    def save(self):
        df = pd.DataFrame(self.results)
        df.to_csv("benchmark_results_full_all_3.csv", index=False, sep=";")


def load_queries(path, b, e):
    df_workload = pd.read_csv(path, sep=";", encoding='utf8')
    df = df_workload[~df_workload['sql_text'].str.contains('ROLLUP', case=False, na=False)]
    df = df[~df_workload['sql_text'].str.contains('GROUPING', case=False, na=False)]
    df = df[~df_workload['sql_text'].str.contains('SETS', case=False, na=False)]
    print(f"filtered df for test: {len(df)}")
    queries = df.iloc[:, 1].dropna().tolist()
    return {f"Q{i:02d}": q for i, q in enumerate(queries[b:e])}


if __name__ == "__main__":
    queries = load_queries(QUERIES_PATH, 0, 40)
    train = load_queries(QUERIES_PATH, 41, 50)
    bench = ETLBenchmark()

    # Порядок запуска
    # bench.run_system_baseline(queries)
    # bench.run_system_bao(train, queries)
    bench.run_system_my_ml(queries)

    bench.save()
