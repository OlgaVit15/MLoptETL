import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ImpalaDatasetPreparer:
    def __init__(self):
        self.numeric_features = [
            'feature_plan_num_joins', 'feature_plan_num_broadcast_joins',
            'feature_plan_num_scan_nodes', 'feature_plan_num_agg_nodes',
            'feature_plan_num_files', 'feature_plan_total_scan_size_bytes',
            'feature_plan_max_cardinality', 'feature_plan_max_row_size'
        ]
        self.target_col1 = 'target_mem_limit'
        self.target_col2 = 'target_mt_dop'

    def clean_and_sanitize(self, df):
        """1. Первичная очистка данных"""
        df = df.copy()

        # Список колонок, которые ДОЛЖНЫ быть числами
        to_numeric_cols = self.numeric_features + [
            self.target_col1, 'sf', 'target_mt_dop',
            'max_mem_limit_dop0', 'target_num_scanner_threads'
        ]

        for col in to_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Очистка строк (ИСКЛЮЧИЛИ target_mt_dop из категорий, оставляем числом)
        cat_cols = ['target_default_join_distribution_mode', 'target_disable_codegen']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        # Фильтрация только успешных запусков
        df = df[(df[self.target_col1] > 0) | (df[self.target_col2] > 0)]

        # Удаление экстремальных выбросов (>64GB)
        df = df[df[self.target_col1] < 65536]

        return df

    def create_derived_features(self, df):
        """2. Генерация производных признаков (Feature Engineering)"""
        df = df.copy()

        # Плотность данных (байт на строку)
        df['feature_bytes_per_row'] = df['feature_plan_total_scan_size_bytes'] / (
                df['feature_plan_max_cardinality'] + 1)

        # Интенсивность джоинов
        df['feature_join_intensity'] = df['feature_plan_num_joins'] * np.log1p(df['feature_plan_max_cardinality'])

        # Средний объем сканирования на узел
        nodes = df['feature_plan_num_scan_nodes'].replace(0, 1)
        df['feature_avg_scan_per_node'] = df['feature_plan_total_scan_size_bytes'] / nodes

        # Логарифмирование (ТОЛЬКО для исходных величин)
        log_cols = [
            'feature_plan_total_scan_size_bytes', 'feature_plan_max_cardinality',
            'feature_plan_max_row_size', 'feature_avg_scan_per_node'
        ]
        for col in log_cols:
            df[f'log_{col}'] = np.log1p(df[col])

        return df

    def _balance_classes(self, df):
        """Внутренний метод: Выравнивание дисбаланса классов MT_DOP"""
        if 'target_mt_dop' not in df.columns:
            return df

        counts = df['target_mt_dop'].value_counts()
        if len(counts) < 2:
            return df

        max_size = counts.max()
        balanced_parts = []

        for val in counts.index:
            group = df[df['target_mt_dop'] == val]
            if len(group) < max_size:
                # Апсемплинг редких классов (2 и 4) до размера мажоритарного (1)
                group = group.sample(max_size, replace=True, random_state=42)
            balanced_parts.append(group)

        return pd.concat(balanced_parts, ignore_index=True)

    def augment_data(self, df, n_clones=2):
        """3. Улучшенная Аугментация (Независимый шум + Пересчет)"""
        augmented_parts = [df]

        # Список СЫРЫХ признаков для зашумления
        raw_features_to_jitter = [
            'feature_plan_total_scan_size_bytes',
            'feature_plan_max_cardinality',
            'feature_plan_max_row_size'
        ]

        for _ in range(n_clones):
            df_noise = df.copy()

            # 1. Зашумляем каждый сырой признак СВОИМ вектором шума (±5%)
            for col in raw_features_to_jitter:
                if col in df_noise.columns:
                    noise_factor = np.random.uniform(0.95, 1.05, size=len(df_noise))
                    df_noise[col] = df_noise[col] * noise_factor

            # 2. Зашумляем таргет (память) отдельно (±10%)
            if self.target_col1 in df_noise.columns:
                target_noise = np.random.uniform(0.9, 1.1, size=len(df_noise))
                df_noise[self.target_col1] = df_noise[self.target_col1] * target_noise

            # 3. ПЕРЕСЧИТЫВАЕМ производные фичи и ЛОГАРИФМЫ для зашумленных данных
            # Это критично, чтобы не ломать математические связи
            df_noise = self.create_derived_features(df_noise)

            augmented_parts.append(df_noise)

        return pd.concat(augmented_parts, ignore_index=True)

    def prepare_full_pipeline(self, df_raw):
        """Главный метод подготовки с защитой редких классов"""
        # Шаг 1: Очистка
        df = self.clean_and_sanitize(df_raw)

        # Шаг 2: Генерация базовых фич
        df = self.create_derived_features(df)

        # --- ИСПРАВЛЕНИЕ СТРАТИФИКАЦИИ ---
        # Мы должны стратифицировать в первую очередь по target_mt_dop,
        # чтобы сохранить редкие классы (DOP 4) в обоих наборах.

        # Если записей слишком мало для сложной стратификации,
        # используем только mt_dop как ключ.
        df['strat_key'] = df['target_mt_dop'].astype(str)

        # Проверяем, есть ли хотя бы 2 примера каждого класса для сплита
        counts = df['strat_key'].value_counts()
        valid_classes = counts[counts >= 2].index
        df = df[df['strat_key'].isin(valid_classes)]

        print(f"Распределение DOP перед разбиением:\n{df['target_mt_dop'].value_counts()}")

        # Шаг 4: Разделение (80/20) с сохранением пропорций DOP
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['strat_key']  # Гарантирует DOP 1, 2 и 4 в обоих наборах
        )

        # Шаг 5: Балансировка ТРЕНИРОВОЧНОГО набора
        # Теперь, когда DOP 4 точно попал в train_df (около 30 записей),
        # мы размножим его до уровня DOP 1.
        print(f"Train DOP before balancing:\n{train_df['target_mt_dop'].value_counts()}")
        train_df = self._balance_classes(train_df)
        print(f"Train DOP after balancing:\n{train_df['target_mt_dop'].value_counts()}")

        # Шаг 6: Аугментация (добавляем шум, сохраняя корреляции)
        train_df = self.augment_data(train_df, n_clones=3)

        # Удаляем временные колонки
        train_df = train_df.drop(columns=['strat_key'])
        test_df = test_df.drop(columns=['strat_key'])

        return train_df, test_df
