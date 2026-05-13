import json
import random
import re
import time
import numpy as np
import pandas as pd

from dataset_pipeline.executor.pg.pg_plan_parser import PostgresPlanParser


class UltimatePGMLPipeline:
    def __init__(self, executor, engine, max_minutes=60):
        self.executor = executor
        self.engine = engine
        self.parser = PostgresPlanParser()
        self.max_minutes = max_minutes
        self.MAX_WORK_MEM_MB = 4096  # Увеличил лимит для вариативности
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

    @staticmethod
    def get_initial_random_settings(discovery_cost, attempt: int = None):
        """
        Генерирует настройки с высокой вариативностью стратегий.
        На каждом 'attempt' мы пробуем принципиально разные подходы.
        """
        cost_factor = np.log10(discovery_cost + 1)

        # 1. Базовые ресурсы (память и параллелизм)
        settings = {
            'work_mem': f"{random.choice([4, 16, 64, 256, 512])}MB",
            'max_parallel_workers_per_gather': random.choice([0, 2, 4, 8]),
            'jit': random.choice(['on', 'off']),
            'enable_indexscan': 'on',
            'enable_seqscan': 'on',
            'enable_bitmapscan': 'on',
            'enable_hashjoin': 'on',
            'enable_mergejoin': 'on',
            'enable_nestloop': 'on'
        }

        # 2. Принудительная вариативность стратегий (Ablation Study)
        # На разных попытках выключаем разные "традиционные" пути

        strategy_roll = random.randint(0, 3)
        if attempt == None:
            attempt = strategy_roll

        if attempt == 0:
            # Первая попытка: "Пусть PG решит сам" (все включено)
            pass
        elif attempt == 1:
            # Вторая попытка: "Анти-Хэш" (выключаем Hash Join, форсируем Merge или NL)
            settings['enable_hashjoin'] = 'off'
            # Если запрос тяжелый, дадим больше памяти для Merge Join (он любит сортировки)
            settings['work_mem'] = f'{random.choice([4, 16, 64, 256, 512])*1.5}MB'
        elif attempt == 2:
            # Третья попытка: "Анти-Индекс" (выключаем обычный Index Scan)
            # Это заставит PG использовать Bitmap Scan или Seq Scan
            settings['enable_indexscan'] = 'off'
            settings['enable_bitmapscan'] = 'on'

        # Дополнительный рандомный шум для флагов (чтобы модель видела редкие комбинации)
        if random.random() > 0.8:
            knob_to_disable = random.choice(['enable_nestloop', 'enable_mergejoin', 'enable_bitmapscan'])
            settings[knob_to_disable] = 'off'

        return settings

    @staticmethod
    def calculate_score(row, is_discovery=False):
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

    def run(self, csv_path, sf, timeout_ms=180000):
        df_workload = pd.read_csv(csv_path, sep=";", encoding='utf8').sample(
            min(500, len(pd.read_csv(csv_path, sep=";"))))
        start_time = time.time()

        for idx, row in df_workload.iterrows():
            if (time.time() - start_time) / 60 > self.max_minutes: break

            # 1. DISCOVERY (собираем фичи плана)
            discovery_params = {'work_mem': '64MB', 'statement_timeout': 30000}  # Короткий таймаут
            raw_discovery = self.executor.execute(row['sql_text'], discovery_params)

            # Парсим фичи плана (даже если упал, нам нужен cost)
            f_base, m_base = self.parser.extract_features_and_metrics(raw_discovery)
            if not f_base:
                continue

            # 2. ПОДГОТОВКА ЦИКЛА
            # Выбираем случайную стартовую точку на основе цены
            current_params = self.get_initial_random_settings(f_base['total_cost'])

            # Адаптивный таймаут: не более 3х от базы или константа
            base_dur = m_base.get('duration_ms', 10000)
            adaptive_timeout = min(max(base_dur * 3, 10000), timeout_ms)

            best_res = None
            res_obj = None

            for attempt in range(3):
                current_params["statement_timeout"] = adaptive_timeout
                print(
                    f"  Q:{row['query_id']} Try {attempt} | Mem:{current_params['work_mem']} | Workers:{current_params['max_parallel_workers_per_gather']}")

                raw_opt = self.executor.execute(row['sql_text'], current_params)

                # Обработка таймаута / ошибок
                if 'errors' in str(raw_opt).lower() or 'timeout' in str(raw_opt).lower():
                    score, reason = 'bad', 'execution_error'
                    f_opt, m_opt = f_base, {col: 0 for col in self.parser.metric_cols}
                    m_opt['duration_ms'] = adaptive_timeout
                else:
                    f_opt, m_opt = self.parser.extract_features_and_metrics(raw_opt)
                    res_temp = {
                        'session_params': current_params,
                        'plan_metrics': f_opt,
                        'profile_metrics': m_opt,
                        'query_id': row['query_id']
                    }
                    score, reason = self.calculate_score(res_temp)

                res_obj = {
                    'query_id': row['query_id'], 'sql_text': row['sql_text'],
                    'plan_metrics': f_opt, 'profile_metrics': m_opt,
                    'session_params': current_params, 'score': score, 'reason': reason,
                    'max_mem_limit_dop0': base_dur
                }

                if score == 'good':
                    best_res = res_obj
                    break
                else:
                    # Корректируем для следующей попытки
                    current_params = self.adjust_settings(res_obj, score, reason)
                    if score == 'bad':
                        adaptive_timeout = min(adaptive_timeout * 1.5, timeout_ms)

            # 3. СОХРАНЕНИЕ
            if best_res:
                print(f"  [+] Saving Optimal")
                self._save(best_res, self.TABLE_OPTIMAL, sf)
            elif res_obj and (
                    'error' not in str(res_obj['reason']).lower() or 'execution' in str(res_obj['reason']).lower()):
                # Сохраняем последнюю неудачную попытку как негативный пример
                print(f"  [-] Saving Failure")
                self._save(res_obj, self.TABLE_FAILURES, sf)

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
            'target_work_mem_mb': self.parse_mem_to_mb(s.get('work_mem')),
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

    def _save(self, res, table, sf):
        flat = self.flatten_results(res, sf)
        pd.DataFrame([flat]).to_sql(table, self.engine, if_exists='append', index=False)
