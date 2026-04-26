import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ImpalaDatasetPreparer:
    def __init__(self):
        self.numeric_features = [
            'feature_plan_num_joins', 'feature_plan_num_broadcast_joins',
            'feature_plan_num_scan_nodes', 'feature_plan_num_agg_nodes',
            'feature_plan_num_files', 'feature_plan_total_scan_size_bytes',
            'feature_plan_max_cardinality', 'feature_plan_max_row_size', 'target_mt_dop'
        ]
        self.target_col = 'metric_pmu'

    def clean_and_sanitize(self, df):
        """1. Первичная очистка данных"""
        df = df.copy()

        # Приведение числовых колонок
        for col in self.numeric_features + [self.target_col, 'sf']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Очистка строк
        cat_cols = ['target_default_join_distribution_mode', 'target_disable_codegen']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        # Оставляем только успешные и адекватные запуски
        # df = df[(df['is_success'] == 1) & (df[self.target_col] > 0)]

        # Удаляем явные выбросы (например, более 64GB), если это не целевые данные
        df = df[df[self.target_col] < 65536]

        return df

    def create_derived_features(self, df):
        """2. Генерация производных признаков (Feature Engineering)"""
        # Плотность данных (байт на строку) - ключевой признак для памяти
        df['feature_bytes_per_row'] = df['feature_plan_total_scan_size_bytes'] / (
                    df['feature_plan_max_cardinality'] + 1)

        # Интенсивность джоинов на объем данных
        df['feature_join_intensity'] = df['feature_plan_num_joins'] * np.log1p(df['feature_plan_max_cardinality'])

        # Средний объем сканирования на один узел
        nodes = df['feature_plan_num_scan_nodes'].replace(0, 1)
        df['feature_avg_scan_per_node'] = df['feature_plan_total_scan_size_bytes'] / nodes

        # Логарифмирование (сглаживание Power-Law распределений)
        log_cols = [
            'feature_plan_total_scan_size_bytes', 'feature_plan_max_cardinality',
            'feature_plan_max_row_size', 'feature_avg_scan_per_node'
        ]
        for col in log_cols:
            df[f'log_{col}'] = np.log1p(df[col])

        return df

    def augment_data(self, df, n_clones=2):
        augmented_parts = [df]
        for _ in range(n_clones):
            df_noise = df.copy()
            # Разный шум для разных признаков!
            df_noise['feature_plan_total_scan_size_bytes'] *= np.random.uniform(0.95, 1.05, size=len(df_noise))
            df_noise['feature_plan_max_cardinality'] *= np.random.uniform(0.98, 1.02, size=len(df_noise))

            # Целевую метрику (память) зашумляем чуть сильнее, чтобы модель была устойчива
            df_noise['metric_pmu'] *= np.random.uniform(0.9, 1.2, size=len(df_noise))

            # ВАЖНО: Пересчитываем производные фичи и логарифмы заново!
            df_noise = self.create_derived_features(df_noise)
            augmented_parts.append(df_noise)
        return pd.concat(augmented_parts)

    def prepare_full_pipeline(self, df_raw):
        """Главый метод подготовки"""
        # Шаг 1: Очистка
        df = self.clean_and_sanitize(df_raw)

        # Шаг 2: Фичи
        df = self.create_derived_features(df)

        # Шаг 3: Стратификация (создаем временные корзины для деления)
        # Это гарантирует, что SF=1 и SF=12 будут и в трейне, и в тесте
        df['strat_bin'] = pd.qcut(df[self.target_col], q=5, labels=False, duplicates='drop')

        # Шаг 4: Разделение (80/20)
        # Используем stratify по SF и корзинам памяти
        train_df, test_df = train_test_split(
            df,
            test_size=0.25,
            random_state=42,
            stratify=df[['sf', 'strat_bin', 'target_mt_dop']]
        )

        # Шаг 5: Аугментация ТОЛЬКО тренировочного набора
        # Мы не трогаем тест, чтобы проверка была честной
        train_df = self.augment_data(train_df, n_clones=2)

        # Удаляем временные колонки
        train_df = train_df.drop(columns=['strat_bin'])
        test_df = test_df.drop(columns=['strat_bin'])

        print(f"Dataset Ready: Train Size={len(train_df)}, Test Size={len(test_df)}")
        return train_df, test_df

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
# preparer = ImpalaDatasetPreparer()
# train_data, test_data = preparer.prepare_full_pipeline(df_sql_raw)

# Теперь эти данные можно подавать в модель:
# model.train(train_data)