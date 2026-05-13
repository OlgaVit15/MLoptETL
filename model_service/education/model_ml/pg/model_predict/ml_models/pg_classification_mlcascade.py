import numpy as np
import pandas as pd
import logging
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, r2_score

logger = logging.getLogger(__name__)


def get_strat_label(row):
    jit = clean_boolean(row['target_jit'])
    idx = clean_boolean(row['target_enable_indexscan'])
    hj = clean_boolean(row['target_enable_hashjoin'])
    nl = clean_boolean(row['target_enable_nestloop'])
    dop = int(row['target_max_parallel_workers_per_gather'])
    return f"J{jit}_I{idx}_H{hj}_N{nl}_D{dop}"


def clean_boolean(val):
    """Приведение всех вариантов булевых значений PG к единому виду (0 или 1)."""
    s = str(val).lower().strip()
    return 1 if s in ['on', 'true', '1', '1.0', 't'] else 0


class StrategyCascadeClassifier:
    def __init__(self, random_state: int = 42):
        self.scan_size_bins = None
        self.global_default_mem = None
        self.random_state = random_state
        self.model_jit = None
        self.model_join = None
        self.model_index = None
        self.model_parallel = None
        self.memory_lookup = {}
        self.feature_cols = [
            'log_total_cost',
            'log_plan_rows',
            'plan_width',
            'planner_error',
            'true_log_rows',
            'cost_density',
            'data_volume',
            'nodes_complexity',
            'seq_scan_weight',
            'parallel_gain_estimate',
            'log_scan_bytes',
            'idx_scan_ratio',
            'mem_intensity',
            'parallel_efficiency',
            'scan_size_bin',
            'bytes_per_scan_node',
            'workers_corr',
            'index_density',
            'rows_to_width_ratio',
        ]

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scan_size_bins is None:
            # Если бины еще не созданы (первый запуск в train), создаем их
            self.scan_size_bins = np.unique(np.percentile(
                df['feature_total_scan_size_bytes'].fillna(0),
                np.linspace(0, 100, 11)
            ))
            # Расширяем границы, чтобы покрыть всё, что выйдет за пределы трейна
            self.scan_size_bins[0] = -np.inf
            self.scan_size_bins[-1] = np.inf

        # Извлекаем признаки с помощью метода для строк DataFrame
        features_list = df.apply(self.extract_features_s, axis=1)

        # Преобразуем список признаков обратно в DataFrame
        features_df = pd.DataFrame(features_list.tolist(), index=df.index, columns=self.feature_cols)

        return features_df.replace([np.inf, -np.inf], 0).fillna(0)

    def extract_features_s(self, row: pd.Series) -> list:
        # БАЗОВЫЕ МЕТРИКИ
        log_total_cost = np.log10(row['feature_total_cost'] + 1)
        log_plan_rows = np.log10(row['feature_plan_rows'] + 1)
        plan_width = row['feature_plan_width'] if row['feature_plan_width'] is not None else 0

        # РЕАЛЬНЫЕ ПОКАЗАТЕЛИ (Planner Error)
        planner_error = row.get('log_planner_error', 0)
        true_log_rows = log_plan_rows + planner_error

        # ФИЧИ СПЕЦИАЛЬНО ДЛЯ DOP (Параллелизм)
        cost_density = log_total_cost / (true_log_rows + 1)
        data_volume = true_log_rows + np.log10(plan_width + 1)
        nodes_complexity = (row['feature_num_scans'] +
                            row['feature_num_joins'] +
                            row['feature_num_aggs'])
        seq_scan_weight = row['feature_num_scans'] - row['feature_num_index_scans']
        parallel_gain_estimate = log_total_cost - np.log10(1000)

        # ФИЗИКА
        log_scan_bytes = np.log10(row['feature_total_scan_size_bytes'] + 1)
        idx_scan_ratio = row['feature_num_index_scans'] / (row['feature_num_scans'] + 1)
        mem_intensity = (row['feature_num_sorts'] +
                         row['feature_num_aggs'] +
                         row['feature_num_joins'])

        parallel_efficiency = log_total_cost / (np.log10(1000) + 1)

        # Биннинг для scan_size_bin
        scan_size = row.get('feature_total_scan_size_bytes', 0)
        scan_size_bin = np.digitize(scan_size, bins=self.scan_size_bins) - 1

        bytes_per_scan_node = np.log10(scan_size / (row.get('feature_num_scans', 0) + 1) + 1)
        workers_corr = log_total_cost / (nodes_complexity + 1)

        index_density = row['feature_num_index_scans'] / (true_log_rows + 1)
        rows_to_width_ratio = true_log_rows / (plan_width + 1)

        return [
            log_total_cost,
            log_plan_rows,
            plan_width,
            planner_error,
            true_log_rows,
            cost_density,
            data_volume,
            nodes_complexity,
            seq_scan_weight,
            parallel_gain_estimate,
            log_scan_bytes,
            idx_scan_ratio,
            mem_intensity,
            parallel_efficiency,
            scan_size_bin,
            bytes_per_scan_node,
            workers_corr,
            index_density,
            rows_to_width_ratio
        ]

    def _get_base_params(self, is_multiclass=True):
        params = {
            'iterations': 3000,
            'learning_rate': 0.025,
            'depth': 7,
            'l2_leaf_reg': 6,
            'loss_function': 'MultiClass' if is_multiclass else 'Logloss',
            'auto_class_weights': 'SqrtBalanced',
            'random_seed': self.random_state,
            'early_stopping_rounds': 200,
            'bootstrap_type': 'MVS',
            'boosting_type': 'Ordered',
            'bagging_temperature': 0.2,
            'random_strength': 1.5,
            'verbose': 200,
            'eval_metric': 'AUC',
            'allow_writing_files': False
        }

        return params

    def train(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        X_train_base = self.extract_features(df_train)
        X_test_base = self.extract_features(df_test)
        # self.feature_cols = X_train_base.columns.tolist()

        # --- 1. JIT ---
        y_jit_tr = df_train['target_jit'].apply(lambda x: 1 if str(x).lower() in ['on', 'true', '1'] else 0)
        y_jit_te = df_test['target_jit'].apply(lambda x: 1 if str(x).lower() in ['on', 'true', '1'] else 0)
        self.model_jit = CatBoostClassifier(**self._get_base_params(is_multiclass=False))
        self.model_jit.fit(X_train_base, y_jit_tr, eval_set=(X_test_base, y_jit_te))

        # --- 2. INDEX ---
        y_idx_tr = df_train['target_enable_indexscan'].apply(
            lambda x: 1 if str(x).lower() in ['on', 'true', '1'] else 0)
        y_idx_te = df_test['target_enable_indexscan'].apply(lambda x: 1 if str(x).lower() in ['on', 'true', '1'] else 0)
        self.model_index = CatBoostClassifier(**self._get_base_params(is_multiclass=False))
        self.model_index.fit(X_train_base, y_idx_tr, eval_set=(X_test_base, y_idx_te))

        # feature_imp = pd.Series(self.model_index.get_feature_importance(),
        #                         index=X_train_base.columns)
        # # Строим график
        # plt.figure(figsize=(10, 8))
        # feature_imp.plot(kind='barh')
        # plt.title('Важность признаков CatBoost INDEX')
        # plt.show()

        # --- 3. JOIN ---
        X_train_base['prob_jit'] = self.model_jit.predict_proba(X_train_base)[:, 1]
        X_train_base['prob_idx'] = self.model_index.predict_proba(X_train_base)[:, 1]

        X_test_base['prob_jit'] = self.model_jit.predict_proba(X_test_base)[:, 1]
        X_test_base['prob_idx'] = self.model_index.predict_proba(X_test_base)[:, 1]

        y_join_tr = df_train['target_enable_hashjoin'].apply(
            lambda x: 1 if str(x).lower() in ['on', 'true', '1'] else 0)
        y_join_te = df_test['target_enable_hashjoin'].apply(
            lambda x: 1 if str(x).lower() in ['on', 'true', '1'] else 0)
        self.model_join = CatBoostClassifier(**self._get_base_params())
        self.model_join.fit(X_train_base, y_join_tr, eval_set=(X_test_base, y_join_te))

        feature_imp = pd.Series(self.model_join.get_feature_importance(),
                                index=X_train_base.columns)
        # Строим график
        plt.figure(figsize=(10, 8))
        feature_imp.plot(kind='barh')
        plt.title('Важность признаков CatBoost JOIN')
        plt.show()

        # --- 4. КАСКАД: DOP (Модель параллелизма) ---
        # ВАЖНО: Используем вероятности (soft labels) вместо жестких классов
        X_train_base['prob_join'] = self.model_join.predict_proba(X_train_base)[:, 1]
        X_test_base['prob_join'] = self.model_join.predict_proba(X_test_base)[:, 1]

        y_par_tr = df_train['target_max_parallel_workers_per_gather'].astype(int)
        y_par_te = df_test['target_max_parallel_workers_per_gather'].astype(int)

        self.model_parallel = CatBoostClassifier(**self._get_base_params(True))
        self.model_parallel.fit(X_train_base, y_par_tr, eval_set=(X_test_base, y_par_te))

        # --- 5. MEMORY LOOKUP ---
        # Используем 90-й перцентиль для надежности (Memory Sufficiency)
        df_train['pred_final_strat'] = (
                y_idx_tr.astype(str) + "_" + y_join_tr.astype(str) + "_" + y_par_tr.astype(
            str)
        )
        self.memory_lookup = df_train.groupby('pred_final_strat')['target_work_mem_mb'].quantile(0.90).to_dict()
        self.global_default_mem = df_train['target_work_mem_mb'].median()

        # feature_dop = pd.Series(self.model_parallel.get_feature_importance(),
        #                         index=X_train_base.columns)
        # # Строим график
        # plt.figure(figsize=(10, 8))
        # feature_dop.plot(kind='barh')
        # plt.title('Важность признаков CatBoost DOP')
        # plt.show()

    def predict_s(self, row: dict) -> dict:
        # Извлекаем признаки для одной строки
        X_base = pd.Series(self.extract_features_s(pd.Series(row)))

        # Инференс каскада
        p_jit_prob = self.model_jit.predict_proba([X_base])[:, 1]
        p_idx_prob = self.model_index.predict_proba([X_base])[:, 1]

        # включаем предсказания индексов и jit для джойнов и dop
        X_base = pd.concat([X_base, pd.Series({'p_jit_prob': p_jit_prob[0], 'p_idx_prob': p_idx_prob[0]})])
        p_join_prob = self.model_join.predict_proba([X_base])[:, 1]

        # включаем предсказание джойнов для dop
        X_base = pd.concat([X_base, pd.Series({'p_join_prob': p_join_prob[0]})])

        p_jit = (p_jit_prob[0] > 0.5).astype(int)
        p_join = self.model_join.predict([X_base])[0][0]
        p_par = self.model_parallel.predict([X_base])[0][0]
        p_idx = (p_idx_prob[0] > 0.5).astype(int)

        strat_key = f"{p_idx}_{p_join}_{p_par}"
        mem = self.memory_lookup.get(strat_key, self.global_default_mem)

        return {
            'target_jit': 'on' if p_jit == 1 else 'off',
            'target_enable_hashjoin': 'off' if p_join == 0 else 'on',
            'target_enable_indexscan': 'on' if p_idx == 1 else 'off',
            'target_max_parallel_workers_per_gather': int(p_par),
            'pred_work_mem_mb': int(mem)
        }

    def predict(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in X_raw.iterrows():
            result = self.predict_s(row.to_dict())
            results.append(result)

        return pd.DataFrame(results, index=X_raw.index)

    def evaluate(self, df_test: pd.DataFrame):
        predictions_df = self.predict(df_test)  # Передаем список словарей

        # --- ОЦЕНКА РЕЗУЛЬТАТОВ ---
        print("\n" + "=" * 60)
        print("ИТОГОВЫЕ МЕТРИКИ КЛАССИФИКАТОРА")
        print("=" * 60)

        # 1. JIT Accuracy
        y_true_jit = df_test['target_jit'].apply(clean_boolean)
        y_pred_jit = predictions_df['target_jit'].apply(clean_boolean)
        print(f"JIT Accuracy:             {accuracy_score(y_true_jit, y_pred_jit):.2%}")

        # 1. INDEX Accuracy
        y_true_idx = df_test['target_enable_indexscan'].apply(clean_boolean)
        y_pred_idx = predictions_df['target_enable_indexscan'].apply(clean_boolean)
        print(f"JIT Accuracy:             {accuracy_score(y_true_idx, y_pred_idx):.2%}")

        # 1. Join Accuracy
        y_true_join = df_test['target_enable_hashjoin'].apply(clean_boolean)
        y_pred_join = predictions_df['target_enable_hashjoin'].apply(clean_boolean)
        print(f"Join Strategy Accuracy:   {accuracy_score(y_true_join, y_pred_join):.2%}")

        # 3. DOP Accuracy
        y_true_dop = df_test['target_max_parallel_workers_per_gather'].astype(int)
        y_pred_dop = predictions_df['target_max_parallel_workers_per_gather'].astype(int)
        print(f"DOP Accuracy:             {accuracy_score(y_true_dop, y_pred_dop):.2%}")

        # 4. Memory Sufficiency
        sufficiency = (predictions_df['pred_work_mem_mb'] >= df_test['target_work_mem_mb']).mean()
        print(f'"Memory R2": {r2_score(df_test['target_work_mem_mb'], predictions_df['pred_work_mem_mb'])}')
        print(f"Memory Sufficiency (P85): {sufficiency:.2%}")

        # 5. Full Session Match (Насколько часто угадана вся комбинация флагов целиком)
        # Это самая жесткая метрика
        match_mask = (
                (y_true_jit == y_pred_jit) &
                (y_true_idx == y_pred_idx) &
                (y_true_join == y_pred_join) &
                (y_true_dop == y_pred_dop) &
                (df_test['target_enable_indexscan'].apply(clean_boolean) ==
                 predictions_df['target_enable_indexscan'].apply(clean_boolean))
        )
        print(f"Full Strategy Match:      {match_mask.mean():.2%}")
        print("=" * 60)

        # Детальный отчет по DOP (самый сложный параметр)
        print("\nОтчет по предсказанию DOP:")
        print(classification_report(y_true_dop, y_pred_dop))
        print(classification_report(y_true_join, y_pred_join))
        print(classification_report(y_true_jit, y_pred_jit))
        print(classification_report(y_true_idx, y_pred_idx))
