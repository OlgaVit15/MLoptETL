import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class PostgresCascadeEnsemble:
    def __init__(self, max_groups=80, n_estimators=150):
        self.max_groups = max_groups
        self.n_estimators = n_estimators
        self.min_samples_leaf_ratio = 0.005
        self.model_error = None
        self.model_parallel = None
        self.grouper = None

        self.plan_features = [
            'feature_total_cost', 'feature_plan_rows', 'feature_plan_width',
            'feature_num_joins', 'feature_num_scans', 'feature_num_aggs',
            'feature_num_sorts', 'feature_num_filters', 'feature_num_index_scans',
            'feature_num_mem_nodes', 'feature_total_scan_size_bytes', 'feature_max_node_cost'
        ]

        # ЧИСТЫЙ список фич без дублей
        self.feature_cols = [f"log_{col}" for col in self.plan_features] + [
            'idx_ratio', 'mem_density', 'log_cost_per_row', 'log_scan_intensity'
        ]

        self.leaf_profiles = {}

    def _prepare_features(self, df):
        X = df.copy()
        # 1. Логарифмируем базу
        for col in self.plan_features:
            if col in X.columns:
                # Обработка потенциальных ошибок при преобразовании к числовому типу
                val = pd.to_numeric(X.get(col, 0), errors='coerce').fillna(0)
                X[f"log_{col}"] = np.log1p(val)
            else:
                X[f"log_{col}"] = 0  # Если фича отсутствует, присваиваем 0

        # 2. производные фичи
        # Плотность стоимости: log(cost) - log(rows)
        if 'log_feature_total_cost' in X.columns and 'log_feature_plan_rows' in X.columns:
            X['log_cost_per_row'] = X['log_feature_total_cost'] - X['log_feature_plan_rows']
        else:
            X['log_cost_per_row'] = 0

            # Интенсивность сканирования
        if 'log_feature_total_scan_size_bytes' in X.columns and 'log_feature_total_cost' in X.columns:
            X['log_scan_intensity'] = X['log_feature_total_scan_size_bytes'] - X['log_feature_total_cost']
        else:
            X['log_scan_intensity'] = 0

            # Соотношения
        if 'feature_num_scans' in X.columns:
            X['idx_ratio'] = X['feature_num_index_scans'] / (X['feature_num_scans'] + 1e-6)  # Добавим epsilon
        else:
            X['idx_ratio'] = X['feature_num_index_scans']

        if 'feature_num_joins' in X.columns and 'feature_num_scans' in X.columns:
            X['mem_density'] = X['feature_num_mem_nodes'] / (X['feature_num_joins'] + X['feature_num_scans'] + 1e-6)
        else:
            X['mem_density'] = X['feature_num_mem_nodes']

        return X[self.feature_cols] if self.feature_cols[0] in X.columns else X

    def train(self, df_train_raw):
        # Подготовка данных
        X_train = self._prepare_features(df_train_raw)
        calculated_min_leaf = max(5, int(len(df_train_raw) * self.min_samples_leaf_ratio))

        # Модель 1: Ошибка
        self.model_error = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=calculated_min_leaf,
                                                 random_state=42)
        y_error = np.log1p(df_train_raw['metric_planner_error_ratio'].fillna(1.0))
        self.model_error.fit(X_train[self.feature_cols], y_error)

        X_train['pred_error'] = self.model_error.predict(X_train[self.feature_cols])

        # Модель 2: Параллелизм
        self.model_parallel = RandomForestClassifier(n_estimators=self.n_estimators,
                                                     min_samples_leaf=calculated_min_leaf, random_state=42)
        y_parallel = df_train_raw['target_max_parallel_workers_per_gather'].astype(int)
        feat_parallel = self.feature_cols + ['pred_error']
        self.model_parallel.fit(X_train[feat_parallel], y_parallel)

        X_train['pred_parallel'] = self.model_parallel.predict(X_train[feat_parallel])

        # Модель 3: Группировщик (на СТРАТЕГИЮ)
        df_train_raw['strat_id'] = (
                df_train_raw['target_enable_nestloop'].astype(str) + "_" +
                df_train_raw['target_enable_indexscan'].astype(str) + "_" +
                y_parallel.astype(str)
        )
        y_grouper, _ = pd.factorize(df_train_raw['strat_id'])

        self.label_encoder = LabelEncoder()
        y_grouper = self.label_encoder.fit_transform(df_train_raw['strat_id'])
        y_grouper = y_grouper[X_train.index]  # Сопоставляем индексы

        self.grouper = DecisionTreeClassifier(max_leaf_nodes=self.max_groups, min_samples_leaf=calculated_min_leaf,
                                              random_state=42)
        feat_grouper = feat_parallel + ['pred_parallel']
        self.grouper.fit(X_train[feat_grouper], y_grouper)

        leaf_ids = self.grouper.apply(X_train[feat_grouper])

        # Сборка профилей
        join_targets = ['target_enable_hashjoin', 'target_enable_mergejoin', 'target_enable_nestloop']
        index_targets = ['target_enable_indexscan', 'target_enable_seqscan', 'target_enable_bitmapscan']

        for lid in np.unique(leaf_ids):
            mask = (leaf_ids == lid)
            opt_data = df_train_raw[mask]

            # Самая частая комбинация флагов
            best_join = opt_data[join_targets].value_counts().idxmax()
            best_idx = opt_data[index_targets].value_counts().idxmax()

            self.leaf_profiles[lid] = {
                'work_mem_mb': int(np.percentile(opt_data['target_work_mem_mb'], 90)),
                'target_jit': opt_data['target_jit'].mode()[0],
                'target_max_parallel_workers_per_gather': int(
                    opt_data['target_max_parallel_workers_per_gather'].mode()[0]),
                **dict(zip(join_targets, best_join)),
                **dict(zip(index_targets, best_idx))
            }
        return self

    def predict(self, plan_dict):
        df_input = pd.DataFrame([plan_dict])
        X = self._prepare_features(df_input)

        # Каскад
        p_err = self.model_error.predict(X[self.feature_cols])[0]
        X['pred_error'] = p_err

        p_par = self.model_parallel.predict(X[self.feature_cols + ['pred_error']])[0]
        X['pred_parallel'] = p_par

        lid = self.grouper.apply(X[self.feature_cols + ['pred_error', 'pred_parallel']])[0]

        profile = self.leaf_profiles.get(lid, self.leaf_profiles[next(iter(self.leaf_profiles))])

        res = profile.copy()
        res['planner_error_estimate'] = float(np.expm1(p_err))
        res['target_parallel_workers'] = int(p_par)
        res['target_work_mem'] = f"{profile['work_mem_mb']}MB"
        res['group_id'] = int(lid)
        return res

    def save(self, filename):
        joblib.dump(self, filename)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename):
        return joblib.load(filename)