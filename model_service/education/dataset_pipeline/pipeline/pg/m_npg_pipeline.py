import hashlib
import re
import time
import random
import numpy as np
import pandas as pd
import json

from dataset_pipeline.executor.pg.pg_plan_parser import PostgresPlanParser


class UltimatePGMLPipeline:
    def __init__(self, executor, engine, max_minutes=300):
        self.executor = executor
        self.engine = engine
        self.parser = PostgresPlanParser()
        self.max_minutes = max_minutes
        self.TABLE_NAME = 'ml_training_dataset_new'
        # Порог значимости: новая стратегия должна быть быстрее базовой минимум на 10%,
        # чтобы мы сменили лейбл. Это решает Requirement 6.
        self.SIGNIFICANCE_THRESHOLD = 0.90

    def parse_mem_to_mb(self, val):
        if not val: return 64
        if isinstance(val, (int, float)): return val
        num = int(re.search(r'(\d+)', str(val)).group(1))
        if 'gb' in str(val).lower(): return num * 1024
        if 'kb' in str(val).lower(): return num / 1024
        return num

    def calculate_efficiency_score(self, duration, params):
        """
        Requirement 2: Избегаем перезакладки.
        Штрафуем за ресурсы нелинейно.
        """
        mem_mb = self.parse_mem_to_mb(params.get('work_mem', '64MB'))
        dop = params.get('max_parallel_workers_per_gather', 0)

        # Штраф: память дороже всего. DOP 4 в два раза "затратнее" DOP 0.
        # Формула: чем больше памяти, тем выше штрафной коэффициент.
        resource_penalty = (1 + (mem_mb / 2048)) * (1 + (dop * 0.15))
        return duration * resource_penalty

    def get_smart_candidates(self):
        """
        Генерируем фиксированные полярные архетипы.
        Всего 5 вариантов, покрывающих максимум пространства.
        """
        return [
            # 1. Минимум ресурсов (энергоэффективный)
            {'max_parallel_workers_per_gather': 0, 'jit': 'off', 'work_mem': '4MB', 'enable_hashjoin': 'off'},

            # 2. Баланс (средний)
            {'max_parallel_workers_per_gather': 2, 'jit': 'off', 'work_mem': '64MB', 'enable_hashjoin': 'on'},

            # 3. Агрессивный (параллелизм + JIT)
            {'max_parallel_workers_per_gather': 4, 'jit': 'on', 'work_mem': '128MB', 'enable_hashjoin': 'on'},

            # 4. Memory-intensive (много памяти, без параллелизма)
            {'max_parallel_workers_per_gather': 0, 'jit': 'off', 'work_mem': '512MB', 'enable_hashjoin': 'on',
             'enable_nestloop': 'off'},

            # 5. Случайный джиттер для вариативности памяти (Requirement 1)
            {'max_parallel_workers_per_gather': random.choice([0, 2]),
             'jit': 'off',
             'work_mem': f"{random.randint(8, 256)}MB",
             'enable_indexscan': 'on'}
        ]

    def run(self, csv_path, sf, timeout_ms=10000):
        df_workload = pd.read_csv(csv_path, sep=";", encoding='utf8')
        # Перемешиваем запросы, чтобы не застревать на пачке однотипных тяжелых задач
        queries = df_workload.sample(frac=1).to_dict('records')

        start_time = time.time()
        count = 0

        for q in queries:
            if (time.time() - start_time) / 60 > self.max_minutes: break
            if count >= 8000: break  # Достигли цели

            # 1. Baseline (X фичи)
            # Ставим короткий таймаут. Если запрос базово идет дольше 5 сек - он нам не нужен для быстрого обучения.
            discovery_res = self.executor.execute(q['sql_text'],
                                                  {'statement_timeout': 5000, 'max_parallel_workers_per_gather': 0})
            f_base, m_base = self.parser.extract_features_and_metrics(discovery_res)

            if not f_base or m_base.get('duration_ms', 0) == 0:
                continue

            # Устанавливаем порог для турнира (не более 1.5x от базового времени)
            current_best_duration = m_base['duration_ms']
            tournament_timeout = max(300, min(current_best_duration * 1.5, 7000))

            best_run_data = {
                'params': {'max_parallel_workers_per_gather': 0, 'jit': 'off', 'work_mem': '64MB'},
                'metrics': m_base,
                'plan_features': f_base
            }
            best_score = self.calculate_efficiency_score(current_best_duration, best_run_data['params'])

            # 2. ОГРАНИЧЕННЫЙ ТУРНИР (Максимум 3 попытки из архетипов)
            # Это ключевое изменение для скорости!
            candidates = random.sample(self.get_smart_candidates(), 3)

            for params in candidates:
                params['statement_timeout'] = int(tournament_timeout)

                res = self.executor.execute(q['sql_text'], params)
                _, metrics = self.parser.extract_features_and_metrics(res)

                duration = metrics.get('duration_ms', 0)
                if duration <= 0:
                    # Если таймаут — не пробуем другие тяжелые конфиги для этого запроса,
                    # просто переходим к следующему кандидату или запросу.
                    continue

                score = self.calculate_efficiency_score(duration, params)

                if score < best_score * 0.9:  # Только если значимо лучше
                    best_score = score
                    best_run_data = {'params': params, 'metrics': metrics, 'plan_features': f_base}
                    # Сжимаем таймаут еще сильнее
                    tournament_timeout = duration * 1.2

                    # 3. Сохранение
            self._save_optimal(q, best_run_data, sf)
            count += 1

    def _save_optimal(self, q, best_data, sf):
        q_hash = hashlib.md5(q['sql_text'].encode()).hexdigest()

        s = best_data['params']
        p = best_data['plan_features']
        m = best_data['metrics']

        flat = {
            'query_id': q.get('query_id'),
            'query_hash': q_hash,
            'sf': sf,
            'sql_text': q['sql_text'],
            'target_work_mem_mb': self.parse_mem_to_mb(s.get('work_mem', '64MB')),
            'target_max_parallel_workers_per_gather': s.get('max_parallel_workers_per_gather', 0),
            'target_jit': s.get('jit', 'off'),
            'target_enable_hashjoin': s.get('enable_hashjoin', 'on'),
            'target_enable_indexscan': s.get('enable_indexscan', 'on'),
            'target_enable_nestloop': s.get('enable_nestloop', 'on')
        }

        # Наполняем фичи (из p - которые всегда f_base)
        for col in self.parser.feature_cols:
            flat[f'feature_{col}'] = p.get(col, 0)

        # Наполняем метрики (из m - которые от победителя)
        for col in self.parser.metric_cols:
            # Превращаем np.float в обычный float для совместимости с SQL
            val = m.get(col, 0)
            if isinstance(val, (np.float64, np.float32)):
                val = float(val)
            flat[f'metric_{col}'] = val

        # Пишем в БД
        pd.DataFrame([flat]).to_sql(self.TABLE_NAME, self.engine, if_exists='append', index=False)
