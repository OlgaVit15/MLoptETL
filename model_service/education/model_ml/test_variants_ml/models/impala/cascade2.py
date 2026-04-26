import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


class ImpalaCascadeModel:
    def __init__(self, max_groups=60, n_estimators=50, min_samples_leaf_ratio=0.005):
        self.max_groups = max_groups
        self.n_estimators = n_estimators
        self.min_samples_leaf_ratio = min_samples_leaf_ratio

        self.model_base = None
        self.model_strategy = None
        self.grouper = None  # Используем ExtraTrees для более точного разбиения

        self.leaf_profiles = {}
        self.feature_cols = [
            'log_num_joins', 'log_num_broadcast_joins', 'log_num_scan_nodes',
            'log_num_agg_nodes', 'log_num_files', 'log_total_scan_size',
            'log_max_cardinality', 'log_max_row_size', 'log_bytes_per_node',
            'joins_density', 'files_per_node'
        ]

    def _prepare_features(self, df):
        X = df.copy()
        # Преобразование в числа и логирование
        mapping = {
            'log_num_joins': 'feature_plan_num_joins',
            'log_num_broadcast_joins': 'feature_plan_num_broadcast_joins',
            'log_num_scan_nodes': 'feature_plan_num_scan_nodes',
            'log_num_agg_nodes': 'feature_plan_num_agg_nodes',
            'log_num_files': 'feature_plan_num_files',
            'log_total_scan_size': 'feature_plan_total_scan_size_bytes',
            'log_max_cardinality': 'feature_plan_max_cardinality',
            'log_max_row_size': 'feature_plan_max_row_size'
        }
        for new_c, old_c in mapping.items():
            X[new_c] = np.log1p(pd.to_numeric(X.get(old_c, 0), errors='coerce').fillna(0))

        # Дополнительные прецизионные признаки
        nodes = pd.to_numeric(X.get('feature_plan_num_scan_nodes', 1), errors='coerce').replace(0, 1)
        joins = pd.to_numeric(X.get('feature_plan_num_joins', 0), errors='coerce')
        files = pd.to_numeric(X.get('feature_plan_num_files', 0), errors='coerce')

        X['log_bytes_per_node'] = np.log1p(pd.to_numeric(X.get('feature_plan_total_scan_size_bytes', 0)) / nodes)
        X['joins_density'] = joins / nodes
        X['files_per_node'] = files / nodes
        return X

    def train(self, df_train_raw):
        # 1. Динамическая регуляризация
        min_leaf = max(4, int(len(df_train_raw) * self.min_samples_leaf_ratio))

        self.model_base = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=min_leaf,
                                                random_state=42, n_jobs=-1)
        self.model_strategy = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=min_leaf,
                                                    random_state=42, n_jobs=-1)
        self.grouper = ExtraTreesClassifier(n_estimators=1, max_leaf_nodes=self.max_groups, min_samples_leaf=min_leaf,
                                            random_state=42)

        df_train = self._prepare_features(df_train_raw)

        # Оценка базовой памяти
        y_base = pd.to_numeric(df_train_raw['max_mem_limit_dop0'], errors='coerce').fillna(df_train_raw['metric_pmu'])
        self.model_base.fit(df_train[self.feature_cols], y_base)
        df_train['pred_base_pmu'] = self.model_base.predict(df_train[self.feature_cols])

        # Выбор DOP
        strategy_feats = self.feature_cols + ['pred_base_pmu']
        y_dop = pd.to_numeric(df_train_raw['target_mt_dop'], errors='coerce').fillna(2)
        self.model_strategy.fit(df_train[strategy_feats], y_dop)

        # Multi-Target Grouper (DOP + Join Mode) для точности
        combined_target = df_train_raw['target_mt_dop'].astype(str) + "_" + df_train_raw[
            'target_default_join_distribution_mode'].astype(str)
        y_combined = LabelEncoder().fit_transform(combined_target)
        self.grouper.fit(df_train[strategy_feats], y_combined)

        # Получаем leaf_id и выпрямляем в 1D массив
        leaf_ids = self.grouper.apply(df_train[strategy_feats])
        if leaf_ids.ndim > 1:
            leaf_ids = leaf_ids[:, 0]
        df_train['leaf_id'] = leaf_ids

        # Сбор профилей
        for leaf_id in df_train['leaf_id'].unique():
            idx = df_train[df_train['leaf_id'] == leaf_id].index
            leaf_data = df_train_raw.loc[idx]
            opt = leaf_data[leaf_data['is_success'] == 1]
            success_rate = leaf_data['is_success'].mean()

            if not opt.empty:
                # PMU Adj: 90-й квантиль для защиты от OOM
                pmu_ratio = (opt['metric_pmu'] / opt['max_mem_limit_dop0']).replace([np.inf, -np.inf], 1.0).fillna(1.0)
                q_level = 0.85 if success_rate > 0.9 else 0.95  # Адаптивный буфер

                self.leaf_profiles[int(leaf_id)] = {
                    'pmu_adj': float(np.percentile(pmu_ratio, q_level * 100)),
                    'mt_dop': int(opt['target_mt_dop'].mode()[0]),
                    'threads': int(opt['target_num_scanner_threads'].median()),
                    'join_mode': str(opt['target_default_join_distribution_mode'].mode()[0]),
                    'codegen': str(opt['target_disable_codegen'].mode()[0]).lower(),
                    'success_rate': float(success_rate)
                }
        return self

    def predict(self, plan_dict):
        df_input = self._prepare_features(pd.DataFrame([plan_dict]))
        base_pmu = self.model_base.predict(df_input[self.feature_cols])[0]

        df_input['pred_base_pmu'] = base_pmu
        strategy_feats = self.feature_cols + ['pred_base_pmu']

        # ОШИБКА БЫЛА ЗДЕСЬ: фиксим получение скалярного leaf_id
        leaf_id_raw = self.grouper.apply(df_input[strategy_feats])
        leaf_id = int(leaf_id_raw[0][0]) if leaf_id_raw.ndim > 1 else int(leaf_id_raw[0])

        profile = self.leaf_profiles.get(leaf_id)

        # Fallback (Запасной выход)
        if not profile or profile['success_rate'] < 0.6:
            pmu_adj, mt_dop, threads, join_mode, codegen, success_rate = 1.3, 2, 4, 'BROADCAST', 'false', 0.0
        else:
            pmu_adj = profile['pmu_adj']
            mt_dop = profile['mt_dop']
            threads = profile['threads']
            join_mode = profile['join_mode']
            codegen = profile['codegen']
            success_rate = profile['success_rate']

        # Расчет памяти с запасом 25% (снизили с 30% для точности)
        stable_mem = int(np.ceil((base_pmu * pmu_adj * 1.25) / 256.0) * 256)

        # ВОЗВРАЩАЕМ КЛЮЧИ ИЗ ВАШЕГО ОРИГИНАЛЬНОГО КОДА
        return {
            'target_mem_limit': f"{max(256, stable_mem)}mb",
            'target_mt_dop': int(mt_dop),
            'target_num_scanner_threads': int(threads),
            'target_default_join_distribution_mode': join_mode,
            'target_disable_codegen': codegen,
            'reliability_score': round(float(success_rate), 2),
            'group_id': int(leaf_id)
        }

    def save_model(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load_model(cls, filepath):
        return joblib.load(filepath)
