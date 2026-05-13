import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

logger = logging.getLogger(__name__)


class PGStrategyCascadeClassifier:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.target_column = 'strategy_label'
        self.model: Optional[Pipeline] = None
        self.strategy_map = {}
        self.memory_config = {}
        self.fallback_label = None
        self.feature_columns = None
        self.mem_bins = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """Генерация физических и каскадных признаков."""
        X = pd.DataFrame(index=df.index)

        # 1. СТОИМОСТНЫЕ ФИЧИ (Логарифмируем, так как разброс в PG огромен)
        X['log_total_cost'] = np.log1p(df['feature_total_cost'].fillna(0))
        X['log_plan_rows'] = np.log1p(df['feature_plan_rows'].fillna(0))

        # 2. ОШИБКА ПЛАНИРОВЩИКА (Внешняя фича, предсказанная другой моделью)
        # Это важнейший параметр для точности 90%+
        X['planner_error_ratio'] = df['planner_error_ratio'].fillna(1.0)

        log10_ratio_error = X['planner_error_ratio']
        coefficient = 10 ** log10_ratio_error
        # СИНТЕТИКА: Исправленное кол-во строк (Scale-фича)
        X['ml_corrected_rows'] = np.log1p(df['feature_plan_rows'] * coefficient)

        # 3. ФИЗИЧЕСКИЕ ХАРАКТЕРИСТИКИ
        X['log_scan_bytes'] = np.log1p(df['feature_total_scan_size_bytes'].fillna(0))
        X['num_joins'] = df['feature_num_joins'].fillna(0)
        X['num_scans'] = df['feature_num_scans'].fillna(0)

        # Плотность данных (Bytes per Row) - критично для выбора Index vs Hash Scan
        X['bytes_per_row'] = df['feature_total_scan_size_bytes'] / (df['feature_plan_rows'] + 1)

        # Сложность структуры
        X['complexity_ratio'] = (df['feature_num_joins'] + df['feature_num_aggs']) / (df['feature_num_scans'] + 1)

        return X.replace([np.inf, -np.inf], 0).fillna(0)

    def _get_model_pipeline(self, n_samples: int) -> Pipeline:
        # Используем параметры, близкие к тем, что работали на Impala
        cb_params = {
            'iterations': 3000,
            'learning_rate': 0.02,
            'depth': 5,
            'l2_leaf_reg': 3,
            'boosting_type': 'Ordered',
            'loss_function': 'MultiClass',
            'auto_class_weights': 'SqrtBalanced',
            'random_seed': self.random_state,
            'bootstrap_type': 'MVS',
            'verbose': 500,
            'early_stopping_rounds': 100
        }

        return Pipeline([
            ('scaler', StandardScaler()),
            ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=min(n_samples, 500))),
            ('clf', CatBoostClassifier(**cb_params))
        ])

    def train(self, df_tr: pd.DataFrame, df_t: pd.DataFrame):
        # Подготовка данных
        # y_train = df_tr[self.target_column]
        # y_test = df_t[self.target_column]
        y_train = df_tr['target_max_parallel_workers_per_gather', 'target_jit', 'target_enable_hashjoin']
        y_test = df_t['target_max_parallel_workers_per_gather', 'target_jit', 'target_enable_hashjoin']

        X_train = self.extract_features(df_tr)
        X_test = self.extract_features(df_t)

        self.feature_columns = X_train.columns.tolist()
        self.fallback_label = y_train.mode()[0]

        # Обучение
        logger.info(f"Начало обучения: {len(y_train.unique())} классов.")
        self.model = self._get_model_pipeline(len(X_train))
        self.model.fit(X_train, y_train)

        # --- МАППИНГ ПО 90-МУ ПРОЦЕНТИЛЮ (Логика как в Impala) ---
        df_stats = df_tr.copy()
        df_stats['label'] = y_train

        self.strategy_map = {}
        self.memory_config = {}

        # for label, group in df_stats.groupby('label'):
        #     row = group.iloc[0]
        #     # Параметры выполнения
        #     self.strategy_map[label] = {
        #         'parallel_workers': int(row['target_max_parallel_workers_per_gather']),
        #         'jit': row['target_jit'],
        #         'enable_hashjoin': row['target_enable_hashjoin'],
        #         'enable_nestloop': row['target_enable_nestloop']
        #     }
        #     # ПАМЯТЬ: Ровно 90-й процентиль этого класса
        #     self.memory_config[label] = int(np.percentile(group['target_work_mem_mb'], 90))

        for label in y_train.unique():
            subset = df_tr[df_tr['strategy_label'] == label]
            self.strategy_map[label] = {
                'parallel': subset['target_max_parallel_workers_per_gather'].mode()[0],
                'jit': subset['target_jit'].mode()[0],
                'hashjoin': subset['target_enable_hashjoin'].mode()[0],
                'nestloop': subset['target_enable_nestloop'].mode()[0],
                'indexscan': subset['target_enable_indexscan'].mode()[0],
            }
            # Расчет памяти: 90-й перцентиль и привязка к бину (4, 8, 16...)
            p90_mem = np.percentile(subset['target_work_mem_mb'], 90)
            self.memory_config[label] = min([b for b in self.mem_bins if b >= p90_mem] or [self.mem_bins[-1]])

        return X_test, y_test

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_input = self.extract_features(X)
        y_pred = self.model.predict(X_input)
        if y_pred.ndim > 1: y_pred = y_pred.flatten()

        results = []
        for i, lbl in enumerate(y_pred):
            target_label = lbl if lbl in self.strategy_map else self.fallback_label
            res = {
                'strategy_label': target_label,
                'pred_work_mem_mb': self.memory_config[target_label],
                **self.strategy_map[target_label]
            }
            results.append(res)
        return pd.DataFrame(results, index=X.index)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, full_df: pd.DataFrame) -> Dict[str, Any]:
        preds = self.predict(full_df.loc[X_test.index])
        actuals = full_df.loc[X_test.index]

        acc_strat = accuracy_score(y_test, preds['strategy_label'])

        print("\n" + "═" * 60)
        print(f"ИТОГОВАЯ ТОЧНОСТЬ МОДЕЛИ: {acc_strat:.2%}")
        print("═" * 60)
        print(classification_report(y_test, preds['strategy_label']))

        # Оценка памяти
        gt_mem = actuals['target_work_mem_mb'].astype(int)
        pr_mem = preds['pred_work_mem_mb']
        sufficiency = (pr_mem >= gt_mem).mean()
        mae = mean_absolute_error(gt_mem, pr_mem)

        print("\n" + "=" * 40)
        print(f"Accuracy Strategy: {acc_strat:.2%}")
        print(f"Memory Sufficiency: {sufficiency:.2%}")
        print(f"Memory MAE:        {mae:.2f} MB")
        print("=" * 40)

        return {"Accuracy": acc_strat, "Sufficiency": sufficiency}
