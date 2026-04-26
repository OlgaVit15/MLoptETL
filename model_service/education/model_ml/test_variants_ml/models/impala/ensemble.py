import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class ImpalaEnsembleModel:
    def __init__(self, n_trees=100):
        # RandomForest гораздо стабильнее одного дерева
        self.model = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=15,
            min_samples_leaf=15,
            random_state=42
        )
        self.leaf_profiles = {}  # Для ансамбля профили строим через "ведущее" дерево
        self.feature_cols = [
            'log_num_joins', 'log_num_broadcast_joins', 'log_num_scan_nodes',
            'log_num_agg_nodes', 'log_num_files', 'log_total_scan_size',
            'log_max_cardinality', 'log_max_row_size', 'log_bytes_per_node'
        ]

    @staticmethod
    def _prepare_features(df):
        X = df.copy()
        # Приведение входных данных к числам
        for col in X.columns:
            if col.startswith('feature_plan_'):
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        # Создаем ВСЕ колонки, которые перечислены в self.feature_cols
        X['log_num_joins'] = np.log1p(X['feature_plan_num_joins'])
        X['log_num_scan_nodes'] = np.log1p(X['feature_plan_num_scan_nodes'])
        X['log_total_scan_size'] = np.log1p(X['feature_plan_total_scan_size_bytes'])
        X['log_max_cardinality'] = np.log1p(X['feature_plan_max_cardinality'])
        X['log_max_row_size'] = np.log1p(X['feature_plan_max_row_size'])

        # ДОБАВЛЯЕМ ПРОПУЩЕННЫЕ:
        X['log_num_broadcast_joins'] = np.log1p(X.get('feature_plan_num_broadcast_joins', 0))
        X['log_num_agg_nodes'] = np.log1p(X.get('feature_plan_num_agg_nodes', 0))
        X['log_num_files'] = np.log1p(X.get('feature_plan_num_files', 0))

        nodes = X['feature_plan_num_scan_nodes'].replace(0, 1)
        X['log_bytes_per_node'] = np.log1p(X['feature_plan_total_scan_size_bytes'] / nodes)

        return X

    def train(self, df_train_raw):
        df_train = self._prepare_features(df_train_raw)
        X = df_train[self.feature_cols]

        # Предсказываем сразу основной таргет - память
        y = pd.to_numeric(df_train_raw['metric_pmu'], errors='coerce').fillna(1024)
        self.model.fit(X.values, y)

        # Для категориальных настроек (threads, dop) используем
        # группировку по "ведущему" дереву из леса (первому)
        leaf_ids = self.model.estimators_[0].apply(X.values)
        df_train['leaf_id'] = leaf_ids

        for leaf_id in np.unique(leaf_ids):
            leaf_data = df_train_raw.iloc[df_train[df_train['leaf_id'] == leaf_id].index]
            opt = leaf_data[leaf_data['is_success'] == 1]
            if not opt.empty:
                self.leaf_profiles[leaf_id] = {
                    'mt_dop': int(opt['target_mt_dop'].mode()[0]),
                    'threads': int(opt['target_num_scanner_threads'].median()),
                    'join_mode': opt['target_default_join_distribution_mode'].mode()[0],
                    'codegen': str(opt['target_disable_codegen'].mode()[0]).lower(),
                    'success_rate': leaf_data['is_success'].mean()
                }
        return self

    def predict(self, plan_dict):
        df_input = self._prepare_features(pd.DataFrame([plan_dict]))
        X = df_input[self.feature_cols]

        # 1. Предсказание памяти лесом (очень точное)
        pred_pmu = self.model.predict(X.values)[0]

        # 2. Получение группы из ведущего дерева для остальных настроек
        leaf_id = self.model.estimators_[0].apply(X.values)[0]
        profile = self.leaf_profiles.get(leaf_id, {'mt_dop': 2, 'success_rate': 0})

        stable_mem = int(np.ceil((pred_pmu * 1.25) / 256.0) * 256)

        return {
            'target_mem_limit': f"{max(256, stable_mem)}mb",
            'target_mt_dop': int(profile['mt_dop']),
            'target_num_scanner_threads': int(profile.get('threads', 4)),
            'target_default_join_distribution_mode': profile.get('join_mode', 'BROADCAST'),
            'target_disable_codegen': profile.get('codegen', 'false'),
            'reliability_score': round(float(profile['success_rate']), 2),
            'group_id': leaf_id
        }
