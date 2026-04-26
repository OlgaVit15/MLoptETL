import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class ImpalaMLModel:
    def __init__(self, max_groups=60):
        self.model = DecisionTreeRegressor(
            max_leaf_nodes=max_groups,
            random_state=42,
            min_samples_leaf=8,
            ccp_alpha=0.0005
        )
        self.leaf_profiles = {}

        # Список признаков БЕЗ 'sf'. Модель будет "чувствовать" масштаб через размеры скана и строк.
        self.feature_cols = [
            'log_num_joins', 'log_num_broadcast_joins', 'log_num_scan_nodes',
            'log_num_agg_nodes', 'log_num_files', 'feature_plan_has_missing_stats',
            'log_total_scan_size', 'log_max_cardinality', 'log_max_row_size',
            'log_bytes_per_node', 'log_bytes_per_row', 'log_cardinality_per_node'
        ]

    def _prepare_features(self, df):
        X = df.copy()

        # 1. Заполняем пропуски и переводим в числа (важно для продакшена)
        for col in X.columns:
            if col.startswith('feature_plan_') or col.startswith('metric_'):
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        # 2. Базовые логарифмы
        X['log_num_joins'] = np.log1p(X['feature_plan_num_joins'])
        X['log_num_broadcast_joins'] = np.log1p(X['feature_plan_num_broadcast_joins'])
        X['log_num_scan_nodes'] = np.log1p(X['feature_plan_num_scan_nodes'])
        X['log_num_agg_nodes'] = np.log1p(X['feature_plan_num_agg_nodes'])
        X['log_num_files'] = np.log1p(X['feature_plan_num_files'])
        X['log_total_scan_size'] = np.log1p(X['feature_plan_total_scan_size_bytes'])
        X['log_max_cardinality'] = np.log1p(X['feature_plan_max_cardinality'])
        X['log_max_row_size'] = np.log1p(X['feature_plan_max_row_size'])

        # 3. ПРОКСИ-ПРИЗНАКИ МАСШТАБА (Вместо SF)
        # Эти признаки позволяют дереву понять плотность данных без знания SF

        nodes = X['feature_plan_num_scan_nodes'].replace(0, 1)
        # Объем данных на один узел (ключевой фактор для памяти)
        X['log_bytes_per_node'] = np.log1p(X['feature_plan_total_scan_size_bytes'] / nodes)

        cards = X['feature_plan_max_cardinality'].replace(0, 1)
        # Нагрузка по строкам на узел (драйвер памяти для Join/Agg)
        X['log_cardinality_per_node'] = np.log1p(X['feature_plan_max_cardinality'] / nodes)

        # Ширина данных (байт на строку)
        X['log_bytes_per_row'] = np.log1p(X['feature_plan_total_scan_size_bytes'] / cards)

        return X

    def train(self, df_train_raw):
        # Принудительно очищаем таргеты
        df_train_raw['metric_pmu'] = pd.to_numeric(df_train_raw['metric_pmu'], errors='coerce').fillna(0)
        df_train_raw['is_success'] = pd.to_numeric(df_train_raw['is_success'], errors='coerce').fillna(0)

        df_train = self._prepare_features(df_train_raw)
        X_feat = df_train[self.feature_cols]
        y = df_train['metric_pmu']

        # Обучаем дерево. Оно само поймет, что при большом log_max_cardinality нужно больше памяти
        self.model.fit(X_feat, y)

        df_train['leaf_id'] = self.model.apply(X_feat)

        for leaf_id in df_train['leaf_id'].unique():
            leaf_data = df_train[df_train['leaf_id'] == leaf_id]
            opt = leaf_data[leaf_data['is_success'] == 1].copy()

            if not opt.empty:
                # Берем 90-й квантиль для надежности
                pmu_90 = opt['metric_pmu'].quantile(0.9)

                self.leaf_profiles[leaf_id] = {
                    'mem_limit': int(pmu_90 * 1.25),  # Запас увеличен до 25%, так как SF неявный
                    'mem_limit_dop0': int(opt['max_mem_limit_dop0'].median()) if 'max_mem_limit_dop0' in opt else int(
                        pmu_90 * 0.7),
                    'mt_dop': int(pd.to_numeric(opt['target_mt_dop'], errors='coerce').mode()[0]),
                    'num_scanner_threads': int(
                        pd.to_numeric(opt['target_num_scanner_threads'], errors='coerce').median()),
                    'join_mode': opt['target_default_join_distribution_mode'].mode()[0],
                    'disable_codegen': str(opt['target_disable_codegen'].mode()[0]).lower(),
                    'success_rate': leaf_data['is_success'].mean()
                }
            else:
                self.leaf_profiles[leaf_id] = {'mem_limit': 1024, 'success_rate': 0, 'mt_dop': 2}

        return self

    def predict(self, plan_dict):
        # Входящий словарь plan_dict не содержит SF, только данные из EXPLAIN
        df_input = pd.DataFrame([plan_dict])
        df_enriched = self._prepare_features(df_input)
        X_feat = df_enriched[self.feature_cols].fillna(0)

        leaf_id = self.model.apply(X_feat)[0]
        profile = self.leaf_profiles.get(leaf_id)

        if not profile or profile.get('mem_limit', 0) == 0:
            return {'target_mem_limit': "2048mb", 'target_mt_dop': 2, 'group_id': -1, 'reliability_score': 0.0}

        # Баккетизация лимитов памяти по 256МБ (меньше шума для Impala Admission Control)
        raw_mem = profile['mem_limit']
        stable_mem = int(np.ceil(raw_mem / 256.0) * 256)
        stable_mem = max(256, min(stable_mem, 32768))

        return {
            'target_mem_limit': f"{stable_mem}mb",
            'target_mem_limit_exec_group_0': f"{profile.get('mem_limit_dop0', 512)}mb",
            'target_mt_dop': int(profile.get('mt_dop', 2)),
            'target_num_scanner_threads': int(profile.get('num_scanner_threads', 4)),
            'target_default_join_distribution_mode': profile.get('join_mode', 'BROADCAST'),
            'target_disable_codegen': profile.get('disable_codegen', 'false'),
            'group_id': int(leaf_id),
            'reliability_score': round(float(profile['success_rate']), 2)
        }
