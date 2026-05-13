import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class ErrPredictor:
    def __init__(
            self,
            target_column: str = 'log_planner_error',
            random_state: int = 42,
            safe_factor: float = 1.0  # Для ошибки планировщика safe_factor обычно 1.0
    ):
        self.target_column = target_column
        self.random_state = random_state
        self.safe_factor = safe_factor
        self.model = None
        self.features_columns = None

    @staticmethod
    def extract_features_s(row: dict) -> list:

        num_joins = row['feature_num_joins']
        num_filters = row['feature_num_filters']
        num_scans = row['feature_num_scans']

        # Глубина/сложность: произведение джойнов на фильтры часто коррелирует с ошибкой
        complexity_index = (num_joins + 1) * (num_filters + 1)

        # 2. Относительные показатели (Density)
        cost_per_row = row['feature_total_cost'] / (row['feature_plan_rows'] + 1)
        width_per_row = row['feature_plan_width']

        # 3. Индексная стратегия
        # Какую долю сканов составляют индексные? (0 - только SeqScan, 1 - только IndexScan)
        index_scan_ratio = row['feature_num_index_scans'] / (row['feature_num_scans'] + 1e-6)

        # 4. Логарифмические признаки (масштабирование порядков PG)
        # В PG оценки cost и rows распределены логнормально
        log_total_cost = np.log1p(row['feature_total_cost'])
        log_plan_rows = np.log1p(row['feature_plan_rows'])
        log_scan_bytes = np.log1p(row['feature_total_scan_size_bytes'])

        # 5. Специфичные признаки памяти
        mem_nodes_count = row['feature_num_mem_nodes']

        return [
            num_joins,
            num_filters,
            num_scans,
            complexity_index,
            cost_per_row,
            width_per_row,
            index_scan_ratio,
            log_total_cost,
            log_plan_rows,
            log_scan_bytes,
            mem_nodes_count
        ]

    def _prepare_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:

        # Извлекаем признаки с помощью метода для строк DataFrame
        features_list = df.apply(self.extract_features_s, axis=1)

        # Преобразуем список признаков обратно в DataFrame
        features_df = pd.DataFrame(features_list.tolist(), index=df.index)

        return features_df.replace([np.inf, -np.inf], 0).fillna(0)

    def _get_model_pipeline(self) -> CatBoostRegressor:
        cb_params = {
            'iterations': 4000,
            'learning_rate': 0.02,
            'depth': 7,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE',
            'eval_metric': 'MAE',
            'bootstrap_type': 'MVS',
            'random_seed': self.random_state,
            'verbose': 500,
            'early_stopping_rounds': 200,
            'allow_writing_files': False
        }

        base_model = CatBoostRegressor(**cb_params)

        return base_model

    def train(self, df_tr: pd.DataFrame, df_t: pd.DataFrame):
        df_tr['log_planner_error'] = df_tr['metric_max_log_planner_error']
        df_tr = df_tr.drop(columns='metric_max_log_planner_error')
        df_t['log_planner_error'] = df_t['metric_max_log_planner_error']
        df_t = df_t.drop(columns='metric_max_log_planner_error')
        X_train = self._prepare_advanced_features(df_tr)
        y_train = df_tr[self.target_column]
        X_test = self._prepare_advanced_features(df_t)
        y_test = df_t[self.target_column]

        self.features_columns = X_train.columns.tolist()

        logger.info(f"Запуск обучения модели ошибки планировщика на {len(X_train)} образцах...")
        self.model = self._get_model_pipeline()
        self.model.fit(X_train, y_train)

        logger.info("Обучение завершено.")
        return X_test, y_test

    def predict(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in X_raw.iterrows():
            result = self.predict_s(row.to_dict())
            results.append(result)

        return pd.DataFrame(results, index=X_raw.index)

    def predict_s(self, row: dict) -> dict:
        if self.model is None:
            raise ValueError("Модель не обучена!")

        X_input = pd.Series(self.extract_features_s(row))

        y_pred = self.model.predict(X_input)

        return y_pred

    def evaluate(self, df_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        y_pred = self.predict(df_test)

        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "Mean_Error_Ratio": np.mean(y_pred),
            "Max_Error_Ratio": np.max(y_pred),
            "Min_Error_Ratio": np.min(y_pred)
        }

        print("\n--- РЕЗУЛЬТАТЫ МОДЕЛИ Catboost Regressor (PG Planner Error) ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics
