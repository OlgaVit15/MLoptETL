import json
import re
import time
import pandas as pd
from dataset_pipeline.executor.pg.pg_plan_parser import PostgresPlanParser

PLAN_METRIC_COLS = [
    'duration_ms', 'actual_rows', 'actual_total_time',
    'temp_written_blocks', 'shared_hit_blocks', 'shared_read_blocks',
    'local_hit_blocks', 'local_read_blocks', 'temp_read_blocks',
    'hit_ratio', 'peak_memory_mb'  # peak_memory_mb добавлен для работы adjust_settings
]

EXPLAIN_FEATURE_COLS = [
    'total_cost', 'plan_rows', 'plan_width',
    'num_joins', 'num_scans', 'num_aggs', 'num_sorts', 'num_filters',
    'num_index_scans', 'index_cond_present', 'total_scan_size_bytes'
]

SESSION_PARAMS = [
    'work_mem', 'max_parallel_workers_per_gather', 'jit',
    'enable_indexscan', 'enable_seqscan', 'enable_bitmapscan',
    'enable_hashjoin', 'enable_mergejoin', 'enable_nestloop'
]


class UltimatePGMLPipeline:
    def __init__(self, executor, engine, max_minutes=60):
        self.executor = executor
        self.parser = PostgresPlanParser()
        self.engine = engine
        self.max_minutes = max_minutes
        self.MAX_WORK_MEM_MB = 2048
        self.MAX_CORES = 8
        self.TABLE_OPTIMAL = 'ml_results_optimal'
        self.TABLE_FAILURES = 'ml_results_failures'

    @staticmethod
    def parse_mem_to_mb(val):
        if not val: return 128
        if isinstance(val, int): return val
        num = int(re.search(r'(\d+)', str(val)).group(1))
        if 'gb' in str(val).lower(): return num * 1024
        return num

    def calculate_score(self, row, is_discovery=False):
        error_str = str(row.get('errors', "")).lower()
        if "out of memory" in error_str:
            return 'bad', 'oom_error'
        if "syntax" in error_str or "function" in error_str:
            return 'fatal', 'syntax_error'
        if row.get('errors'): return 'bad', 'execution_error'

        m = row.get('profile_metrics')
        p = row.get('plan_metrics')
        if not m or not p:
            return 'bad', 'error_no_metrics'

        dur = m['duration_ms'] + 1
        spill_blocks = m.get('temp_written_blocks', 0)
        hit_ratio = m.get('hit_ratio', 1.0)

        # Расчет ошибки планировщика (Actual Rows / Plan Rows)
        # Если планировщик ошибся в 100+ раз — это сигнал проблемного индекса или джойна
        rows_error = m['actual_rows'] / (p['plan_rows'] + 1)

        # 1. Критическая нехватка памяти (Spill)
        if spill_blocks > 0:
            return 'bad', 'spill_to_disk'

        # 2. Неэффективность индексов (Index Inefficiency)
        if p.get('num_index_scans', 0) > 0:
            if m.get('hit_ratio', 1.0) < 0.4 and dur > 500:
                print(f"!!! Detected Index Inefficiency: Low hit ratio on Index Scan")
                return 'overhead', 'index_inefficiency'
            if rows_error > 200:
                print(f"!!! Detected Index Inefficiency: Planner misjudged index cardinality")
                return 'overhead', 'index_cardinality_error'

        # 3. Неэффективность соединений (Join Inefficiency)
        # Паттерн: Nested Loop используется там, где строк оказалось слишком много
        if p.get('node_type_nested_loop_count', 0) > 0 and m['actual_rows'] > 10000:
            return 'overhead', 'join_inefficiency'

        # 4. Неэффективное использование кэша
        if hit_ratio < 0.4 and dur > 1000:
            return 'overhead', 'low_cache_hit'

        # 5. Накладные расходы на JIT
        if dur < 1000 and row['session_params'].get('jit') == 'on':
            return 'overhead', 'jit_heavy'

        # 6. Проверка параллелизма
        if int(row['session_params'].get('parallel_workers', 0)) > 2 and m['actual_rows'] < 10000:
            return 'overhead', 'parallelism_waste'

        return 'good', 'optimal'

    def adjust_settings(self, row, score, reason):
        """Логика адаптации параметров ресурсов и стратегий планировщика"""
        new_set = row['session_params'].copy()
        curr_mem = self.parse_mem_to_mb(new_set.get('work_mem', '128MB'))
        curr_workers = int(new_set.get('max_parallel_workers_per_gather', 1))

        if score == 'bad':

            if reason in ['spill_to_disk', 'oom']:
                # Увеличиваем память
                new_limit = int(curr_mem * 2)
                new_set['work_mem'] = f"{min(new_limit, self.MAX_WORK_MEM_MB)}MB"
                if reason == 'oom':
                    new_set['max_parallel_workers_per_gather'] = max(1, curr_workers - 1)
            else:
                # Общая ошибка выполнения — пробуем добавить параллелизм
                new_set['debug_parallel_query'] = 'on'
                new_set['max_parallel_workers_per_gather'] = min(curr_workers + 1, self.MAX_CORES)

        elif score == 'overhead':
            if reason == 'jit_heavy':
                new_set['jit'] = 'off'

            elif reason == 'low_cache_hit':
                # Если кэш-хит низкий, возможно Index Scan заставляет прыгать по диску.
                # Пробуем Bitmap Scan или Seq Scan.
                new_set['enable_indexscan'] = 'off'
                new_set['enable_bitmapscan'] = 'on'

            elif reason == 'index_inefficiency':
                # Планировщик выбрал индекс, но строк слишком много.
                # Выключаем обычный индекс, форсируем Bitmap или Seq.
                # Если индекс плох, пробуем Bitmap Scan или чистый Seq Scan
                if new_set.get('enable_indexscan') != 'off':
                    print(">>> Action: Disabling Index Scan to force Bitmap/Seq Scan")
                    new_set['enable_indexscan'] = 'off'
                    new_set['enable_bitmapscan'] = 'on'
                else:
                    print(">>> Action: Disabling Bitmap Scan to force Seq Scan")
                    new_set['enable_bitmapscan'] = 'off'

            elif reason == 'join_inefficiency':
                # Nested Loop плох для больших данных. Форсируем Hash Join.
                new_set['enable_nestloop'] = 'off'
                new_set['enable_hashjoin'] = 'on'
                # Для Hash Join часто нужно больше памяти
                new_set['work_mem'] = f"{min(curr_mem + 128, self.MAX_WORK_MEM_MB)}MB"

            elif reason == 'parallelism_waste':
                new_set['debug_parallel_query'] = 'on'
                new_set['max_parallel_workers_per_gather'] = 1

        return new_set

    def flatten_results(self, res, sf):
        s = res.get('session_params', {})
        p = res.get('plan_metrics', {})
        m = res.get('profile_metrics', {})

        # Расчет ошибки планировщика как таргета
        planner_error = m.get('actual_rows', 0) / (p.get('plan_rows', 0) + 1)

        output = {
            'query_id': int(res['query_id']),
            'score': str(res['score']),
            'reason': str(res['reason']),
            'sf': int(sf),
            'sql_text': str(res.get('sql_text', '')),
            # Таргеты (сохраняем как есть)
            'target_work_mem_mb':  self.parse_mem_to_mb(s.get('work_mem')),
            'target_max_parallel_workers_per_gather': int(s.get('max_parallel_workers_per_gather', 0)),
            'target_jit': str(s.get('jit', 'off')),
            'target_enable_indexscan': str(s.get('enable_indexscan', 'on')),
            'target_enable_seqscan': str(s.get('enable_seqscan', 'on')),
            'target_enable_bitmapscan': str(s.get('enable_bitmapscan', 'on')),
            'target_enable_hashjoin': str(s.get('enable_hashjoin', 'on')),
            'target_enable_mergejoin': str(s.get('enable_mergejoin', 'on')),
            'target_enable_nestloop': str(s.get('enable_nestloop', 'on')),

            # Метрика ошибки планировщика (для обучения каскада)
            'metric_planner_error_ratio': float(planner_error),
            'max_mem_limit_dop0': float(res.get('max_mem_limit_dop0', 0))
        }

        # Фичи (Numbers)
        for col in self.parser.feature_cols:
            output[f'feature_{col}'] = float(p.get(col, 0))
        # Метрики (Numbers)
        for col in self.parser.metric_cols:
            output[f'metric_{col}'] = float(m.get(col, 0))

        return output

    def run(self, csv_path, sf, timeout_ms=120000):
        df_workload = pd.read_csv(csv_path, sep=";", encoding='utf8').sample(500)
        start_time = time.time()

        for idx, row in df_workload.iterrows():
            if (time.time() - start_time) / 60 > self.max_minutes: break

            print(f"\n>>> Processing Query {row['query_id']}")

            # 1. DISCOVERY
            params = {
                'work_mem': '64MB',
                'max_parallel_workers_per_gather': 0, 'jit': 'on',
                'enable_nestloop': 'on', 'enable_hashjoin': 'on',
                'statement_timeout': timeout_ms
            }
            raw = self.executor.execute(row['sql_text'], params)

            if 'errors' in str(raw).lower() and 'timeout' in str(raw).lower():
                print(f"  [!] Discovery timed out (> {timeout_ms}ms). Skipping heavy query.")
                continue

            f, m = self.parser.extract_features_and_metrics(raw)
            m["plan_count_rows_error"] = m['actual_rows'] / (f['plan_rows'] + 1)

            if not f:
                continue

            # 2. LOOP
            current_params = params.copy()
            current_params['work_mem'] = '4MB'  # Начинаем с малого

            baseline_dur = m['duration_ms']

            adaptive_timeout = min(max(baseline_dur * 3, 5000), 120000)

            for attempt in range(2):
                current_params["statement_timeout"] = adaptive_timeout
                print(f"  Attempt {attempt} | Mem: {current_params['work_mem']}")
                raw_opt = self.executor.execute(row['sql_text'], current_params)
                # f_opt, m_opt = self.parser.extract_features_and_metrics(raw_opt)

                if 'errors' in str(raw_opt).lower() and 'timeout' in str(raw_opt).lower():
                    # Если вылетели по тайм-ауту — это 'bad' результат (эквивалент нехватки ресурсов)
                    m_opt = {col: 0 for col in self.parser.metric_cols}
                    m_opt['duration_ms'] = adaptive_timeout
                    m_opt['actual_rows'] = 0
                    f_opt = f  # Оставляем фичи из дискавери
                    errors = "Statement timeout reached"
                else:
                    f_opt, m_opt = self.parser.extract_features_and_metrics(raw_opt)
                    errors = json.loads(raw_opt).get('errors')

                res_obj = {
                    'query_id': row['query_id'], 'sql_text': row['sql_text'],
                    'plan_metrics': f_opt, 'profile_metrics': m_opt,
                    'session_params': current_params, 'errors': None,
                    'max_mem_limit_dop0': baseline_dur
                }

                score, reason = self.calculate_score(res_obj)
                res_obj['score'] = score
                res_obj['reason'] = reason

                if score == 'good':
                    print(f"  [+] Success! Saving to Optimal.")
                    self._save(res_obj, self.TABLE_OPTIMAL, sf)
                    break
                else:
                    print(f"  [-] {reason}. Adjusting...")
                    self._save(res_obj, self.TABLE_FAILURES, sf)
                    current_params = self.adjust_settings(res_obj, score, reason)

                    if "timeout" in str(errors).lower():
                        adaptive_timeout = min(adaptive_timeout * 2, 160000)

    def _save(self, res, table, sf):
        flat = self.flatten_results(res, sf)
        pd.DataFrame([flat]).to_sql(table, self.engine, if_exists='append', index=False)
