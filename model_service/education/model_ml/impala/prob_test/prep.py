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

    def create_derived_features(self, df):
        """Продвинутый Feature Engineering для 'вытягивания' точности Join и Threads"""
        df = df.copy()

        # 1. Признаки для Scanner Threads (соотношение файлов и параллелизма)
        nodes = df['feature_plan_num_scan_nodes'].replace(0, 1)
        dop = df['target_mt_dop'].replace(0, 1)
        df['files_per_node'] = df['feature_plan_num_files'] / nodes
        df['load_per_thread'] = df['feature_plan_total_scan_size_bytes'] / (nodes * dop)

        # 2. Признаки для Join Mode (интенсивность джойнов)
        df['join_complexity'] = df['feature_plan_num_joins'] / (df['feature_plan_num_scan_nodes'] + 1)
        # Отношение широковещательных джойнов к общему числу (важно для точности Join Mode)
        df['broadcast_ratio'] = df['feature_plan_num_broadcast_joins'] / (df['feature_plan_num_joins'] + 1)

        # 3. Признаки для Memory (плотность данных)
        df['row_density'] = df['feature_plan_total_scan_size_bytes'] / (df['feature_plan_max_cardinality'] + 1)

        # Логарифмирование ключевых признаков
        for col in ['feature_plan_total_scan_size_bytes', 'feature_plan_max_cardinality', 'load_per_thread']:
            df[f'log_{col}'] = np.log1p(df[col])

        return df

    # def augment_data_smart(self, df, n_clones=3):
    #     augmented_parts = [df]
    #
    #     rare_mask = (df['target_default_join_distribution_mode'] == 'SHUFFLE') | (df['target_mt_dop'] > 4)
    #     df_rare = df[rare_mask]
    #
    #     for i in range(n_clones):
    #         current_df = df_rare.copy() if i > 1 else df.copy()
    #         scale = np.random.uniform(0.7, 1.5, size=len(current_df))
    #
    #         current_df['feature_plan_total_scan_size_bytes'] *= scale
    #         current_df['feature_plan_max_cardinality'] *= scale
    #         current_df['feature_plan_num_files'] = (current_df['feature_plan_num_files'] * scale).astype(int)
    #
    #         # Память растет нелинейно (добавляем небольшой штраф на масштаб)
    #         current_df['metric_pmu'] *= (scale ** 1.1)
    #
    #         augmented_parts.append(current_df)
    #
    #     return pd.concat(augmented_parts, ignore_index=True)

    def augment_data_physics_based(self, df, n_clones=2):
        """
        Размножаем данные, учитывая, что память (target) растет
        при росте объема сканирования и кардинальности.
        """
        augmented_parts = [df]

        for _ in range(n_clones):
            clone = df.copy()

            # Генерируем случайный коэффициент изменения объема данных (от -15% до +20%)
            # Мы не берем огромный разброс, чтобы не выйти за рамки здравого смысла
            scale = np.random.uniform(0.85, 1.2, size=len(clone))

            # Корректируем признаки
            clone['feature_plan_total_scan_size_bytes'] *= scale
            clone['feature_plan_max_cardinality'] *= scale

            # КОРРЕКТИРУЕМ ТАРГЕТ: Память в Impala растет пропорционально данным,
            # но обычно чуть медленнее (корень или логарифм) или почти линейно.
            # Используем коэффициент scale для изменения max_mem_limit_dop0
            clone['max_mem_limit_dop0'] = np.ceil(clone['max_mem_limit_dop0'] * (scale ** 0.9) / 64) * 64

            # Категориальные таргеты (режимы джойна и тд) оставляем без изменений,
            # так как это "свойства" того же самого типа запроса.

            augmented_parts.append(clone)

        return pd.concat(augmented_parts, ignore_index=True)

    def prepare_full_pipeline(self, df_raw):
        # 1. Очистка
        df = df_raw.copy()
        for col in self.numeric_features + ['metric_pmu', 'sf']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 2. Первичные фичи
        df = self.create_derived_features(df)

        # 3. Стратификация по трем осям (SF, Memory, JoinMode)
        # критически важно, чтобы в тесте были ВСЕ типы джойнов
        df['strat_col'] = (
                df['target_default_join_distribution_mode'].astype(str) + "_" +
                pd.qcut(df['metric_pmu'], q=4, labels=False, duplicates='drop').astype(str)
        )

        # 4. Split
        # Используем stratify, чтобы редкие SHUFFLE попали и в тест, и в трейн
        train_df, test_df = train_test_split(
            df,
            test_size=0.05,
            random_state=42,
            stratify=df['strat_col']
        )

        # 5. аугментация только для трейна
        # train_df = self.augment_data_smart(train_df, n_clones=3)

        # Удаляем мусор
        train_df = train_df.drop(columns=['strat_col'])
        test_df = test_df.drop(columns=['strat_col'])

        return train_df, test_df
