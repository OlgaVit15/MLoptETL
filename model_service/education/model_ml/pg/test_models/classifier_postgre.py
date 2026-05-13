import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyCascadeClassifier:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.target_column = 'temp_target'
        self.model: Optional[CatBoostClassifier] = None  # Убрали Pipeline

        self.strategy_params_map = {}
        self.memory_map = {}
        self.mem_bins = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.feature_columns: Optional[list] = None

    # @staticmethod
    # def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    #     """Генерация признаков. Упор на отношения (ratios)."""
    #     X = pd.DataFrame(index=df.index)
    #
    #     # 1. СТОИМОСТНЫЕ ФИЧИ (Логарифм по основанию 2 или 10 лучше масштабирует)
    #     X['log_total_cost'] = np.log10(df['feature_total_cost'] + 1)
    #     X['log_plan_rows'] = np.log10(df['feature_plan_rows'] + 1)
    #     X['plan_width'] = df['feature_plan_width'].fillna(0)
    #
    #     # 2. ВАЖНО: Ошибка планировщика
    #     X['planner_error'] = df['metric_max_log_planner_error'].fillna(0)
    #     X['corrected_log_rows'] = X['log_plan_rows'] + X['planner_error']
    #
    #     # 3. ФИЗИЧЕСКИЕ ХАРАКТЕРИСТИКИ И ОТНОШЕНИЯ
    #     X['log_scan_bytes'] = np.log10(df['feature_total_scan_size_bytes'] + 1)
    #
    #     # Стоимость на одну строку (помогает отличить Index Scan от Seq Scan)
    #     X['cost_per_row'] = X['log_total_cost'] / (X['log_plan_rows'] + 1)
    #
    #     # Байт на одну строку (важно для Hash Join vs Merge Join)
    #     X['bytes_per_row'] = df['feature_total_scan_size_bytes'] / (df['feature_plan_rows'] + 1)
    #
    #     X['num_joins'] = df['feature_num_joins'].fillna(0)
    #     X['num_scans'] = df['feature_num_scans'].fillna(0)
    #
    #     # Сложность запроса
    #     X['join_density'] = X['num_joins'] / (X['num_scans'] + 1)
    #     X['index_scan_ratio'] = df['feature_num_index_scans'] / (df['feature_num_scans'] + 1)
    #
    #     # Интенсивность памяти
    #     X['mem_intensity'] = (df['feature_num_sorts'] + df['feature_num_aggs'] + df['feature_num_joins'])
    #
    #     return X.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)

        # 1. БАЗОВЫЕ МЕТРИКИ
        X['log_total_cost'] = np.log10(df['feature_total_cost'] + 1)
        X['log_plan_rows'] = np.log10(df['feature_plan_rows'] + 1)
        X['plan_width'] = df['feature_plan_width'].fillna(0)

        # 2. РЕАЛЬНЫЕ ПОКАЗАТЕЛИ (Planner Error)
        X['planner_error'] = df['log_planner_error'].fillna(0)
        X['true_log_rows'] = X['log_plan_rows'] + X['planner_error']

        # 3. ФИЧИ СПЕЦИАЛЬНО ДЛЯ DOP (Параллелизм)
        # Отношение стоимости к строкам (Плотность работы)
        X['cost_density'] = X['log_total_cost'] / (X['true_log_rows'] + 1)
        # Объем данных (физический)
        X['data_volume'] = X['true_log_rows'] + np.log10(X['plan_width'] + 1)
        # Нагрузка на узлы
        X['nodes_complexity'] = df['feature_num_scans'] + df['feature_num_joins'] + df['feature_num_aggs']
        # Вероятность параллельного скана (Seq Scan без индексов)
        X['seq_scan_weight'] = df['feature_num_scans'] - df['feature_num_index_scans']
        # Оценка преимущества параллелизма (Cost Setup Setup)
        X['parallel_gain_estimate'] = X['log_total_cost'] - np.log10(1000)  # 1000 - дефолтный setup cost в PG

        # 4. ФИЗИКА
        X['log_scan_bytes'] = np.log10(df['feature_total_scan_size_bytes'] + 1)
        X['idx_scan_ratio'] = df['feature_num_index_scans'] / (df['feature_num_scans'] + 1)
        X['mem_intensity'] = (df['feature_num_sorts'] + df['feature_num_aggs'] + df['feature_num_joins'])

        # X['parallel_gain_estimate'] = X['log_total_cost'] - np.log10(1000)
        # Показывает, насколько стоимость запроса вообще оправдывает запуск параллелизма
        X['parallel_efficiency'] = X['log_total_cost'] / (np.log10(1000) + 1)
        X['scan_size_bin'] = pd.qcut(df['feature_total_scan_size_bytes'], q=10, labels=False, duplicates='drop')
        X['bytes_per_scan_node'] = np.log10(df['feature_total_scan_size_bytes'] / (df['feature_num_scans'] + 1) + 1)

        return X.replace([np.inf, -np.inf], 0).fillna(0)

    # def _get_model_pipeline(self, n_samples: int) -> Pipeline:
    #     """Создание пайплайна с сохранением всех гиперпараметров CatBoost."""
    #     cb_params = {
    #         'iterations': 3000,
    #         'learning_rate': 0.02,
    #         'depth': 7,
    #         'l2_leaf_reg': 3,
    #         'boosting_type': 'Ordered',
    #         'loss_function': 'MultiClass',
    #         # 'auto_class_weights': 'Balanced',
    #         'random_seed': self.random_state,
    #         'bootstrap_type': 'MVS',
    #         'verbose': 500,
    #         'early_stopping_rounds': 200
    #     }
    #
    #     feature_pipe = Pipeline([
    #         ('scaler', RobustScaler()),
    #         ('quantile', QuantileTransformer(
    #             output_distribution='normal',
    #             n_quantiles=min(n_samples, 500),
    #             random_state=self.random_state
    #         ))
    #     ])
    #
    #     ml_models = CatBoostClassifier(**cb_params)
    #     return Pipeline([('preprocessor', feature_pipe), ('clf', ml_models)])
    # def _get_trained_model(self, X_train, y_train, X_val, y_val):
    #     """Прямая настройка CatBoost без Pipeline."""
    #
    #     # Вычисляем веса классов вручную для стабильности,
    #     # либо используем auto_class_weights
    #     ml_models = CatBoostClassifier(
    #         iterations=3000,
    #         learning_rate=0.02,
    #         depth=3,  # Увеличили глубину для захвата сложных комбинаций PG
    #         l2_leaf_reg=3,
    #         boosting_type = 'Ordered',
    #         bootstrap_type='Bernoulli',
    #         loss_function='MultiClass',
    #         auto_class_weights='SqrtBalanced',
    #         random_seed=self.random_state,
    #         # bootstrap_type='MVS',
    #         early_stopping_rounds=200,
    #         use_best_model=True,
    #         task_type='CPU',
    #         verbose=500,
    #         subsample=0.8
    #     )
    #
    #     # ml_models = self._get_model_pipeline(len(X_train))
    #
    #     ml_models.fit(
    #         X_train, y_train,
    #         eval_set=(X_val, y_val),
    #         plot=False
    #     )
    #     return ml_models
    #
    # def train(self, df_tr: pd.DataFrame, df_t: pd.DataFrame):
    #     # 1. Берем только те классы, которые есть в трейне (уже отфильтрованном)
    #     valid_classes = df_tr[self.target_column].unique()
    #
    #     # 2. Очищаем тестовый набор от классов, которых нет в обучении
    #     df_t_filtered = df_t[df_t[self.target_column].isin(valid_classes)].copy()
    #
    #     X_train = self.extract_features(df_tr)
    #     y_train = df_tr[self.target_column]
    #
    #     X_test = self.extract_features(df_t_filtered)
    #     y_test = df_t_filtered[self.target_column]
    #
    #     self.feature_columns = X_train.columns.tolist()
    #
    #     self.ml_models = CatBoostClassifier(
    #         iterations=2000,
    #         learning_rate=0.03,
    #         depth=8,
    #         l2_leaf_reg=10,
    #         loss_function='MultiClass',
    #         random_seed=self.random_state,
    #         early_stopping_rounds=200,
    #         bootstrap_type='Bernoulli',
    #         subsample=0.8,
    #         verbose=200
    #     )
    #
    #     self.ml_models.fit(
    #         X_train, y_train,
    #         eval_set=(X_test, y_test),
    #         use_best_model=True
    #     )
    #
    #     # Сохранение параметров
    #     for label in y_train.unique():
    #         subset = df_tr[df_tr[self.target_column] == label]
    #         self.strategy_params_map[label] = {
    #             'parallel': subset['target_max_parallel_workers_per_gather'].mode()[0],
    #             'hashjoin': subset['target_enable_hashjoin'].mode()[0],
    #             'nestloop': subset['target_enable_nestloop'].mode()[0],
    #             'indexscan': subset['target_enable_indexscan'].mode()[0],
    #             'jit': subset['target_jit'].mode()[0]
    #         }
    #         self.memory_map[label] = np.percentile(subset['target_work_mem_mb'], 85)
    #
    #     return X_test, y_test

    # @staticmethod
    # def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    #     X = pd.DataFrame(index=df.index)
    #
    #     # 1. СТОИМОСТЬ (Логарифмы)
    #     X['log_total_cost'] = np.log10(df['feature_total_cost'] + 1)
    #     X['log_plan_rows'] = np.log10(df['feature_plan_rows'] + 1)
    #
    #     # 2. ПЛАНИРОВЩИК (Важнейшая коррекция)
    #     X['planner_error'] = df['metric_max_log_planner_error'].fillna(0)
    #     X['corrected_log_rows'] = X['log_plan_rows'] + X['planner_error']
    #
    #     # 3. ТРИГГЕРЫ (То, что заставляет PG менять стратегию)
    #     # JIT включается обычно при cost > 100 000
    #     X['jit_trigger'] = (df['feature_total_cost'] > 100000).astype(int)
    #     # Параллелизм (DOP) эффективен на больших сканах
    #     X['log_scan_bytes'] = np.log10(df['feature_total_scan_size_bytes'] + 1)
    #     X['parallel_trigger'] = (df['feature_total_scan_size_bytes'] > 10000000).astype(int)
    #
    #     # 4. ОТНОШЕНИЯ (Ratios)
    #     # Стоимость обработки одной строки (помогает отличить Index от Seq scan)
    #     X['cost_per_row'] = X['log_total_cost'] / (X['log_plan_rows'] + 1)
    #     # Байт на строку (важно для выбора Hash Join)
    #     X['bytes_per_row'] = df['feature_total_scan_size_bytes'] / (df['feature_plan_rows'] + 1)
    #
    #     # Сложность запроса
    #     X['join_density'] = df['feature_num_joins'] / (df['feature_num_scans'] + 1)
    #     X['idx_ratio'] = df['feature_num_index_scans'] / (df['feature_num_scans'] + 1)
    #
    #     # Операции в памяти
    #     X['mem_ops'] = df['feature_num_sorts'] + df['feature_num_aggs'] + df['feature_num_joins']
    #
    #     return X.replace([np.inf, -np.inf], 0).fillna(0)

    def _get_base_params(self, is_multiclass=True):
        lf = None
        if not is_multiclass:
            lf = 'Logloss'
        else:
            lf = 'MultiClassOneVsAll'
        return {
            'iterations': 3000,
            'learning_rate': 0.025,
            'depth': 7,  # Глубина 8 лучше ловит комбинации параметров
            'l2_leaf_reg': 5,  # Сильная регуляризация против переобучения
            'loss_function': lf,
            'auto_class_weights': 'SqrtBalanced',  # ГЛАВНОЕ: балансирует редкие классы, не убивая частые
            'random_seed': self.random_state,
            'early_stopping_rounds': 200,
            'bootstrap_type': 'MVS',
            'boosting_type': 'Ordered',
            'bagging_temperature': 0.2,  # Добавляет немного случайности
            'random_strength': 1.5,
            'verbose': 200,
            'eval_metric': 'AUC'
        }

    def train(self, df_tr: pd.DataFrame, df_t: pd.DataFrame):
        # 1. Важнейшая фильтрация: убираем классы-призраки (менее 10 примеров)
        # Они создают UndefinedMetricWarning и портят обучение
        counts = df_tr[self.target_column].value_counts()
        valid_classes = counts[counts >= 10].index
        df_tr = df_tr[df_tr[self.target_column].isin(valid_classes)].copy()

        # 2. Очищаем тест от классов, которых нет в трейне
        df_t = df_t[df_t[self.target_column].isin(valid_classes)].copy()

        X_train = self.extract_features(df_tr)
        y_train = df_tr[self.target_column]
        X_test = self.extract_features(df_t)
        y_test = df_t[self.target_column]

        # 3. Настройка CatBoost для борьбы с дисбалансом
        self.model = CatBoostClassifier(**self._get_base_params()
            # iterations=4000,
            # learning_rate=0.02,
            # depth=8,  # Глубина 8 лучше ловит комбинации параметров
            # l2_leaf_reg=15,  # Сильная регуляризация против переобучения
            # loss_function='MultiClass',
            # auto_class_weights='SqrtBalanced',  # ГЛАВНОЕ: балансирует редкие классы, не убивая частые
            # random_seed=self.random_state,
            # early_stopping_rounds=300,
            # bootstrap_type='MVS',
            # verbose=200
        )
        # self.ml_models = self._get_trained_model(X_train, y_train,X_test, y_test)

        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

        # 4. Сохранение маппинга
        for label in y_train.unique():
            subset = df_tr[df_tr[self.target_column] == label]
            self.strategy_params_map[label] = {
                'parallel': int(subset['target_max_parallel_workers_per_gather'].mode()[0]),
                'hashjoin': subset['target_enable_hashjoin'].mode()[0],
                'nestloop': subset['target_enable_nestloop'].mode()[0],
                'indexscan': subset['target_enable_indexscan'].mode()[0],
                'jit': subset['target_jit'].mode()[0]
            }
            # Память: 90-й перцентиль для надежности
            self.memory_map[label] = np.percentile(subset['target_work_mem_mb'], 90)

        return X_train

    def predict(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X_feat = self.extract_features(X_raw)
        # Получаем вероятности
        probs = self.model.predict_proba(X_feat)
        classes = self.model.classes_

        res = []
        for row_probs in probs:
            # Берем топ-1 класс
            best_idx = np.argmax(row_probs)
            label = classes[best_idx]

            params = self.strategy_params_map.get(label, {
                'parallel': 0, 'jit': 'off', 'hashjoin': 'on',
                'nestloop': 'on', 'indexscan': 'on'
            })

            res.append({
                'strategy_label': label,
                'pred_work_mem_mb': self.memory_map.get(label, 64),
                'target_max_parallel_workers_per_gather': params['parallel'],
                'target_jit': params['jit'],
                'target_enable_hashjoin': params['hashjoin'],
                'target_enable_nestloop': params['nestloop'],
                'target_enable_indexscan': params['indexscan']
            })

        return pd.DataFrame(res, index=X_raw.index)

    def evaluate(self, X_test_feat: pd.DataFrame, y_test: pd.Series, full_df: pd.DataFrame) -> Dict[str, Any]:
        """Расчет метрик."""
        # Мы принимаем уже извлеченные фичи X_test_feat из train()
        results_pred = self.predict(full_df.loc[y_test.index])
        actuals = full_df.loc[y_test.index]

        acc_strategy = accuracy_score(y_test, results_pred['strategy_label'])

        # Проверка отдельных параметров
        acc_dop = (actuals['target_max_parallel_workers_per_gather'].values ==
                   results_pred['target_max_parallel_workers_per_gather'].values).mean()

        # Метрики памяти
        y_actual_mem = actuals['target_work_mem_mb']
        y_pred_mem = results_pred['pred_work_mem_mb']

        # Достаточность памяти (сколько раз предсказали БОЛЬШЕ или РАВНО чем нужно было)
        sufficiency = (y_pred_mem >= y_actual_mem).mean()

        print(f"\nAccuracy (Full Strategy): {acc_strategy:.2%}")
        print(f"Accuracy (DOP):            {acc_dop:.2%}")
        print(f"Memory Sufficiency:       {sufficiency:.2%}")

        return {"Accuracy": acc_strategy, "Sufficiency": sufficiency}
