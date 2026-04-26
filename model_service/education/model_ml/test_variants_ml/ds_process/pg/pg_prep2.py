import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class PostgresPhysicsSimulator:
    def __init__(self):
        self.CACHE_SIZE = 256 * 1024 * 1024  # 256MB

    def get_duration(self, row, settings):
        """Физика: расчет времени через созависимость параметров."""
        # Базовая стоимость от строк и сложности
        base = (row['feature_plan_rows'] * 0.0001) * (1 + row['feature_num_joins'] * 0.3)

        # Влияние индексов (если выключены - замедляем в 5 раз для больших данных)
        idx_factor = 1.0
        if settings['target_enable_indexscan'] == 'OFF' and row['feature_plan_rows'] > 10000:
            idx_factor = 5.0

        # Влияние параллелизма (нелинейно)
        dop = int(settings['target_parallel_workers'])
        parallel_factor = 1.0
        if dop > 0:
            # Закон Амдала + накладные расходы на запуск
            parallel_factor = (0.3 + (0.7 / dop)) + (dop * 0.05)
            if base < 50:  # Параллелизм вреден для очень быстрых запросов
                parallel_factor *= 1.5

        # Влияние памяти (Spill to disk)
        data_volume = row['feature_plan_rows'] * row['feature_plan_width']
        work_mem_bytes = settings['target_work_mem_mb'] * 1024 * 1024
        spill_penalty = 1.0
        temp_blocks = 0

        if data_volume > work_mem_bytes:
            temp_blocks = (data_volume - work_mem_bytes) / 8192
            spill_penalty = 3.0  # Резкое замедление при сбросе на диск

        final_dur = base * idx_factor * parallel_factor * spill_penalty
        return max(final_dur, 0.1), int(temp_blocks)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PostgresDatasetPreparer:
    def __init__(self):
        self.physics = PostgresPhysicsSimulator()
        self.plan_features = [
            'feature_total_cost', 'feature_plan_rows', 'feature_plan_width',
            'feature_num_joins', 'feature_num_scans', 'feature_num_aggs',
            'feature_num_sorts', 'feature_num_filters', 'feature_num_index_scans',
            'feature_num_mem_nodes', 'feature_total_scan_size_bytes', 'feature_max_node_cost'
        ]

    def _generate_coherent_settings(self, row, has_index_mask):  # <-- Добавили has_index_mask
        """Логический движок: созависимое назначение настроек через веса (Scores)."""
        # 1. Оценка параллелизма (зависит от стоимости и кол-ва сканов)
        p_score = (np.log1p(row['feature_total_cost']) * 0.5 + row['feature_num_scans'] * 2)
        p_score += np.random.normal(0, 2)  # Добавляем шум

        dop = 0
        if p_score > 15:
            dop = 4
        elif p_score > 8:
            dop = 2

        # 2. Индексы vs SeqScan (созависимость: если много джойнов, индекс важнее)
        idx_score = (row['feature_num_index_scans'] * 5) - (row['feature_num_filters'] * 2)
        idx_score += np.random.normal(0, 3)

        enable_idx = 'ON' if idx_score > 0 else 'OFF'
        # Если выключили индексы, заставляем использовать HashJoin (частая ситуация в PG)
        enable_hash = 'ON' if enable_idx == 'OFF' or row['feature_num_joins'] > 2 else 'OFF'

        # 3. Память (зависит от объема данных и типа джойна)
        needed_mem = (row['feature_plan_rows'] * row['feature_plan_width']) / (1024 * 1024)
        if enable_hash == 'ON': needed_mem *= 1.5  # HashJoin требует больше памяти

        # Целевой work_mem с запасом, но иногда (20% случаев) даем меньше, чтобы модель видела ошибки
        if np.random.rand() > 0.2:
            target_mem = int(needed_mem * np.random.uniform(1.2, 2.0))
        else:
            target_mem = int(needed_mem * np.random.uniform(0.5, 0.9))

        target_mem = max(64, min(target_mem, 4096))  # Разумные границы

        # 4. Bitmap Scan (теперь корректно использует has_index_mask)
        enable_bitmap = 'OFF'
        # BitmapScan используется, когда нет прямого индекса, но есть фильтры, которые могут его ускорить
        if not has_index_mask and row['feature_num_filters'] > 1 and np.random.rand() > 0.4:
            enable_bitmap = 'ON'
            enable_idx = 'OFF'  # BitmapScan и IndexScan обычно взаимоисключающие

        return {
            'target_parallel_workers': dop,
            'target_enable_indexscan': enable_idx,
            'target_enable_hashjoin': enable_hash,
            'target_enable_seqscan': 'ON' if enable_idx == 'OFF' and enable_bitmap == 'OFF' else 'OFF',
            'target_enable_bitmapscan': enable_bitmap,
            'target_work_mem_mb': target_mem,
            'target_jit': 'ON' if (row['feature_total_cost'] > 100000 and dop > 0) else 'OFF',
            'target_enable_mergejoin': 'OFF',  # Редкие
            'target_enable_nestloop': 'ON' if enable_idx == 'ON' else 'OFF'  # NestLoop часто идет с IndexScan
        }

    def prepare_full_pipeline(self, df_raw):
        # 1. Базовая очистка
        df = df_raw.copy()
        for col in self.plan_features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 2. Генерация вариантов (Аугментация)
        final_rows = []
        for _, base_row in df.iterrows():

            # Предварительно вычисляем has_index_mask для базовой строки
            has_index_mask = base_row['feature_num_index_scans'] > 0

            for _ in range(5):
                new_row = base_row.copy()

                scale = np.exp(np.random.uniform(-1, 3))
                new_row['feature_plan_rows'] *= scale
                new_row['feature_total_cost'] *= scale
                new_row['feature_total_scan_size_bytes'] *= scale

                # Генерируем созависимые настройки, передавая has_index_mask
                settings = self._generate_coherent_settings(new_row, has_index_mask)  # <-- Передаем mask
                for k, v in settings.items():
                    new_row[k] = v

                dur, temp = self.physics.get_duration(new_row, settings)
                new_row['metric_duration_ms'] = dur
                new_row['metric_temp_written_blocks'] = temp
                new_row['metric_planner_error_ratio'] = np.random.lognormal(0, 0.5) * (
                            1 + new_row['feature_num_joins'] * 0.2)

                new_row['target_max_parallel_workers_per_gather'] = settings['target_parallel_workers']

                final_rows.append(new_row)

        df_final = pd.DataFrame(final_rows)

        # 3. Feature Engineering (Логарифмы)
        for col in self.plan_features + ['metric_duration_ms', 'metric_planner_error_ratio']:
            df_final[f'log_{col}'] = np.log1p(df_final[col])

        # 4. Сплит со стратификацией по сложности
        # Увеличим кол-во корзин для стратификации, чтобы лучше покрыть весь диапазон
        df_final['strat_col'] = pd.qcut(df_final['log_feature_total_cost'].rank(method='first'), 10, labels=False,
                                        duplicates='drop')

        # Если qcut не удался (мало уникальных значений), то стратификация не нужна
        if df_final['strat_col'].isnull().all():
            stratify_param = None
            print("!!! WARNING: Could not stratify by log_feature_total_cost. Using default split.")
        else:
            stratify_param = df_final['strat_col']

        train, test = train_test_split(df_final, test_size=0.2, stratify=stratify_param, random_state=42)

        return train.drop(columns=['strat_col']), test.drop(columns=['strat_col'])

