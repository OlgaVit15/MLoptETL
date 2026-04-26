import pandas as pd
from sklearn.model_selection import train_test_split


class PostgresRepresentativityBooster:
    def __init__(self):
        self.strategy_flags = [
            'target_jit', 'target_enable_indexscan', 'target_enable_seqscan',
            'target_enable_bitmapscan', 'target_enable_hashjoin',
            'target_enable_mergejoin', 'target_enable_nestloop'
        ]

    def _prepare_base_metrics(self, df):
        df = df.copy()
        for col in df.columns:
            if col.startswith('target_enable') or col == 'target_jit':
                df[col] = df[col].astype(str).str.upper()
        return df

    def boost_dataset(self, df_raw):
        df = self._prepare_base_metrics(df_raw)

        # 1. Прореживаем монотонные данные
        cheap_mask = (df['feature_total_cost'] < 5000) & (df['metric_planner_error_ratio'] < 2)
        df_cheap = df[cheap_mask].sample(frac=0.3, random_state=42) if any(cheap_mask) else df[cheap_mask]
        df_complex = df[~cheap_mask]
        df = pd.concat([df_cheap, df_complex])

        augmented_data = [df]

        # 2. Усиливаем сигнал ошибки (Expert Labeling)
        high_error = df[df['metric_planner_error_ratio'] > 15].copy()
        if not high_error.empty:
            for noise_level in [0.978, 1.478]:  # Создаем вариации
                v = high_error.copy()
                # Смена стратегии
                v['target_enable_nestloop'] = 'OFF'
                v['target_enable_hashjoin'] = 'ON'
                v['target_max_parallel_workers_per_gather'] = 4
                # Корректируем метрики под новую стратегию
                v['metric_duration_ms'] *= 0.7
                v['feature_total_cost'] *= noise_level
                augmented_data.append(v)

        # 3. Балансировка индексов
        heavy_idx = df[df['feature_num_index_scans'] > 2].copy()
        if not heavy_idx.empty:
            v_idx = heavy_idx.copy()
            v_idx['target_enable_indexscan'] = 'OFF'
            v_idx['target_enable_bitmapscan'] = 'ON'
            augmented_data.append(v_idx)

        df_boosted = pd.concat(augmented_data, ignore_index=True)
        return self._apply_scale_factor(df_boosted)

    def _apply_scale_factor(self, df):
        parts = [df]
        # Используем несколько масштабов для лучшей обобщаемости
        for scale in [0.745, 1.35]:
            scaled = df.copy()
            scaled['feature_plan_rows'] *= scale
            scaled['feature_total_cost'] *= (scale ** 1.14)
            scaled['metric_duration_ms'] *= (scale ** 0.76)
            scaled['target_work_mem_mb'] = (scaled['target_work_mem_mb'] * scale).clip(4, 4096)
            parts.append(scaled)
        return pd.concat(parts, ignore_index=True)

    def prepare_for_training(self, df_raw):
        # Очистка имен и типов
        for f in self.strategy_flags:
            df_raw[f] = df_raw[f].astype(str).str.upper()

        # Создаем колонку для стратификации на ОРИГИНАЛЬНЫХ данных
        df_raw['strat_col'] = (
                df_raw['target_enable_nestloop'].astype(str) + "_" +
                df_raw['target_enable_hashjoin'].astype(str) + "_" +
                df_raw['target_enable_mergejoin'].astype(str) + "_" +
                df_raw['target_enable_seqscan'].astype(str) + "_" +
                df_raw['target_enable_bitmapscan'].astype(str) + "_" +
                df_raw['target_enable_indexscan'].astype(str) + "_" +
                df_raw['target_jit'].astype(str) + "_" +
                (df_raw['target_max_parallel_workers_per_gather'] > 0).astype(str)
        )

        train_raw, test_raw = train_test_split(
            df_raw, test_size=0.05, random_state=42, stratify=df_raw['strat_col']
        )

        # Бустим ТОЛЬКО трейн
        df_train = self.boost_dataset(train_raw)

        # Убираем временные колонки
        cols_to_drop = ['strat_col']

        return train_raw.drop(columns=cols_to_drop), test_raw.drop(columns=cols_to_drop)
