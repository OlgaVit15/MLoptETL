import numpy as np
import pandas as pd
import logging
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


class StrategyCascadeClassifier:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_jit = None
        self.model_index = None
        self.model_hashjoin = None
        self.model_nestloop = None
        self.model_parallel = None

        self.memory_lookup = {}
        self.global_default_mem = None
        self.feature_cols = None
        self.scan_size_bins = None  # Для хранения границ qcut

    @staticmethod
    def _calculate_features_array(data_dict: dict, bins=None) -> np.array:
        """
        Ядро расчетов. Работает как со словарем (инференс),
        так и вызывается из батч-обработки.
        """
        # Извлекаем значения (защита от None)
        cost = float(data_dict.get('feature_total_cost', 0) or 0)
        rows = float(data_dict.get('feature_plan_rows', 0) or 0)
        width = float(data_dict.get('feature_plan_width', 0) or 0)
        err = float(data_dict.get('log_planner_error', 0) or 0)
        scans = float(data_dict.get('feature_num_scans', 0) or 0)
        joins = float(data_dict.get('feature_num_joins', 0) or 0)
        aggs = float(data_dict.get('feature_num_aggs', 0) or 0)
        idx_scans = float(data_dict.get('feature_num_index_scans', 0) or 0)
        scan_bytes = float(data_dict.get('feature_total_scan_size_bytes', 0) or 0)
        sorts = float(data_dict.get('feature_num_sorts', 0) or 0)

        # Математика (векторизуется NumPy автоматически)
        log_cost = np.log10(cost + 1)
        log_rows = np.log10(rows + 1)
        true_log_rows = log_rows + err
        nodes_comp = scans + joins + aggs

        # Вычисляем фичи
        feats = [
            log_cost,  # log_total_cost
            log_rows,  # log_plan_rows
            width,  # plan_width
            err,  # planner_error
            true_log_rows,  # true_log_rows
            log_cost / (true_log_rows + 1),  # cost_density
            true_log_rows + np.log10(width + 1),  # data_volume
            nodes_comp,  # nodes_complexity
            scans - idx_scans,  # seq_scan_weight
            log_cost - 3.0,  # parallel_gain_estimate (log10 1000)
            np.log10(scan_bytes + 1),  # log_scan_bytes
            idx_scans / (scans + 1),  # idx_scan_ratio
            sorts + aggs + joins,  # mem_intensity
            log_cost / 3.001,  # parallel_efficiency
            np.digitize(scan_bytes, bins) if bins is not None else 0,  # scan_size_bin
            np.log10(scan_bytes / (scans + 1) + 1),  # bytes_per_scan_node
            log_cost / (nodes_comp + 1),  # workers_corr
            idx_scans / (true_log_rows + 1),  # index_density
            true_log_rows / (width + 1)  # rows_to_width_ratio
        ]
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизированный батч-процессинг для обучения и тестов."""
        if self.scan_size_bins is None:
            # Вычисляем границы только при обучении
            self.scan_size_bins = np.percentile(df['feature_total_scan_size_bytes'], np.linspace(0, 100, 11))[1:-1]

        # Применяем ядро расчетов ко всему DF
        data_list = df.to_dict('records')
        feat_matrix = np.array([self._calculate_features_array(d, self.scan_size_bins) for d in data_list])

        # Создаем выходной DF (порядок колонок должен быть железным)
        cols = [
            'log_total_cost', 'log_plan_rows', 'plan_width', 'planner_error', 'true_log_rows',
            'cost_density', 'data_volume', 'nodes_complexity', 'seq_scan_weight',
            'parallel_gain_estimate', 'log_scan_bytes', 'idx_scan_ratio', 'mem_intensity',
            'parallel_efficiency', 'scan_size_bin', 'bytes_per_scan_node', 'workers_corr',
            'index_density', 'rows_to_width_ratio'
        ]
        return pd.DataFrame(feat_matrix, columns=cols, index=df.index)

    def _get_base_params(self, is_multiclass=False):
        return {
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 7,
            'l2_leaf_reg': 5,
            'loss_function': 'MultiClass' if is_multiclass else 'Logloss',
            'auto_class_weights': 'Balanced',
            'random_seed': self.random_state,
            'early_stopping_rounds': 100,
            'verbose': False,
            'bootstrap_type': 'MVS'
        }

    def train(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        X_train = self.extract_features(df_train)
        X_test = self.extract_features(df_test)
        self.feature_cols = X_train.columns.tolist()

        # 1. JIT & Index
        y_jit = df_train['target_jit'].apply(clean_boolean)
        self.model_jit = CatBoostClassifier(**self._get_base_params()).fit(X_train, y_jit)

        y_idx = df_train['target_enable_indexscan'].apply(clean_boolean)
        self.model_index = CatBoostClassifier(**self._get_base_params()).fit(X_train, y_idx)

        # 2. Joins (Независимые бинарные модели для HJ и NL)
        # Генерируем мета-признаки (вероятности)
        p_jit_tr = self.model_jit.predict_proba(X_train)[:, 1]
        p_idx_tr = self.model_index.predict_proba(X_train)[:, 1]

        X_join_tr = np.column_stack([X_train.values, p_jit_tr, p_idx_tr])

        y_hj = df_train['target_enable_hashjoin'].apply(clean_boolean)
        self.model_hashjoin = CatBoostClassifier(**self._get_base_params()).fit(X_join_tr, y_hj)

        y_nl = df_train['target_enable_nestloop'].apply(clean_boolean)
        self.model_nestloop = CatBoostClassifier(**self._get_base_params()).fit(X_join_tr, y_nl)

        # 3. Parallel (DOP)
        p_hj_tr = self.model_hashjoin.predict_proba(X_join_tr)[:, 1]
        p_nl_tr = self.model_nestloop.predict_proba(X_join_tr)[:, 1]

        X_par_tr = np.column_stack([X_train.values, p_jit_tr, p_idx_tr, p_hj_tr, p_nl_tr])
        y_par = df_train['target_max_parallel_workers_per_gather'].astype(int)

        self.model_parallel = CatBoostClassifier(**self._get_base_params(True)).fit(X_par_tr, y_par)

        # 4. Memory Lookup
        df_train['strat_key'] = (
                y_jit.astype(str) + "_" + y_hj.astype(str) + "_" + y_nl.astype(str) + "_" + y_par.astype(str)
        )
        self.memory_lookup = df_train.groupby('strat_key')['target_work_mem_mb'].quantile(0.9).to_dict()
        self.global_default_mem = df_train['target_work_mem_mb'].median()

    def predict_single(self, data_dict: dict) -> dict:
        """
        Сверхбыстрый инференс для Airflow.
        На входе - один словарь. На выходе - один словарь.
        """
        # 1. Фичи
        x_base = self._calculate_features_array(data_dict, self.scan_size_bins)
        x_base_2d = x_base.reshape(1, -1)

        # 2. Каскад вероятностей
        p_jit = self.model_jit.predict_proba(x_base_2d)[0, 1]
        p_idx = self.model_index.predict_proba(x_base_2d)[0, 1]

        x_join = np.append(x_base, [p_jit, p_idx]).reshape(1, -1)
        p_hj = self.model_hashjoin.predict_proba(x_join)[0, 1]
        p_nl = self.model_nestloop.predict_proba(x_join)[0, 1]

        x_par = np.append(x_base, [p_jit, p_idx, p_hj, p_nl]).reshape(1, -1)
        res_par = int(self.model_parallel.predict(x_par)[0, 0])

        # 3. Пороги
        f_jit = 1 if p_jit > 0.5 else 0
        f_idx = 1 if p_idx > 0.5 else 0
        f_hj = 1 if p_hj > 0.5 else 0
        f_nl = 1 if p_nl > 0.5 else 0

        # 4. Memory
        key = f"{f_jit}_{f_hj}_{f_nl}_{res_par}"
        mem = self.memory_lookup.get(key, self.global_default_mem)

        return {
            'target_jit': 'on' if f_jit else 'off',
            'target_enable_indexscan': 'on' if f_idx else 'off',
            'target_enable_hashjoin': 'on' if f_hj else 'off',
            'target_enable_nestloop': 'on' if f_nl else 'off',
            'target_max_parallel_workers_per_gather': res_par,
            'pred_work_mem_mb': int(mem)
        }

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Для совместимости с evaluate и ETL батчами."""
        records = df_raw.to_dict('records')
        results = [self.predict_single(r) for r in records]
        return pd.DataFrame(results, index=df_raw.index)
