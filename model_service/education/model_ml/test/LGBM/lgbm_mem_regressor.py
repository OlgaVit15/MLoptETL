import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

from lightgbm import LGBMRegressor
from pandas import Series
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryLimitPredictor:
    """
    Класс для обучения и предсказания лимитов памяти с использованием CatBoost
    и трансформации целевой переменной.
    """

    def __init__(
            self,
            target_column: str = 'base_pmu',
            random_state: int = 42,
            safe_factor: float = 1.2
    ):
        self.target_column = target_column
        self.random_state = random_state
        self.safe_factor = safe_factor
        self.model = None
        self.features_columns = None

    def _prepare_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_list = df.apply(self.extract_features_s, axis=1)
        features_df = pd.DataFrame(features_list.tolist(), index=df.index)

        return features_df.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def extract_features_s(df: dict) -> list:
        """Внутренний метод подготовки физических фичей."""
        # Базовые
        scans = df['feature_plan_num_scan_nodes']
        joins = df['feature_plan_num_joins']
        aggs = df['feature_plan_num_agg_nodes']
        row_size = df['feature_plan_max_row_size']

        # Физика объема
        vol_theory = df['feature_plan_max_cardinality'] * df['feature_plan_max_row_size']
        log_scan_size = np.log1p(df['feature_plan_total_scan_size_bytes'])
        log_card = np.log1p(df['feature_plan_max_cardinality'])

        # Сложность структуры
        complexity = (joins + 1) * (aggs + 1)
        nodes_efficiency = log_scan_size / (scans + 1)

        # "Сырые" данные
        raw_scan = df['feature_plan_total_scan_size_bytes']
        raw_card = df['feature_plan_max_cardinality']

        return [
            scans,
            joins,
            aggs,
            row_size,
            vol_theory,
            log_scan_size,
            log_card,
            complexity,
            nodes_efficiency,
            raw_scan,
            raw_card
        ]

    def _get_model_pipeline(self, n_samples: int) -> TransformedTargetRegressor:
        lgbm_params = {
            'n_estimators': 4000,
            'learning_rate': 0.02,
            'max_depth': 7,  # Глубина дерева
            # В LightGBM важно настроить num_leaves.
            # Для симметричного дерева глубиной 7 это было бы 2^7 - 1 = 127.
            'num_leaves': 64,  # Можно оставить 31-64 для баланса скорости и точности
            'reg_lambda': 3,  # L2 регуляризация (аналог l2_leaf_reg)
            'objective': 'regression',
            'metric': 'mae',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,  # Меньше логов (0 или -1)
            # Параметры бутстрапа (аналог MVS в CatBoost)
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'feature_fraction': 0.8,
        }

        feature_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('quantile', QuantileTransformer(
                output_distribution='normal',
                n_quantiles=min(n_samples, 1000),
                random_state=self.random_state
            ))
        ])

        base_model = LGBMRegressor(**lgbm_params)

        full_pipe = Pipeline([
            ('preprocessor', feature_pipe),
            ('regressor', base_model)
        ])

        return TransformedTargetRegressor(
            regressor=full_pipe,
            func=np.log1p,
            inverse_func=np.expm1
        )

    def train(self, df_tr: pd.DataFrame, df_t: pd.DataFrame):
        X_train = self._prepare_advanced_features(df_tr)
        y_train = df_tr[self.target_column]
        X_test = self._prepare_advanced_features(df_t)
        y_test = df_t[self.target_column]
        self.features_columns = X_train.columns.tolist()
        """Обучение модели."""
        logger.info(f"Запуск обучения на {len(X_train)} образцах...")

        self.model = self._get_model_pipeline(len(X_train))
        self.model.fit(X_train, y_train)

        logger.info("Обучение завершено.")
        return X_test, y_test

    # Вставьте эти реализации вместо ваших старых predict_s и predict
    def predict_s(self, X: dict, apply_safety_factor: bool = False) -> float:
        """Инференс на СЛОВАРЕ. Максимально быстро."""
        # 1. Получаем список фичей
        feat_list = self.extract_features_s(X)
        # 2. Превращаем в 2D массив (1 строка, N колонок)
        X_input = np.array([feat_list], dtype=np.float32)

        # 3. Предсказываем (модель не ругается на имена, т.к. не знает их)
        y_pred = self.model.predict(X_input)

        res = max(float(y_pred[0]), 1.0)
        return res * self.safe_factor if apply_safety_factor else res

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Батч-предсказание для теста."""
        X_input = self._prepare_advanced_features(X)
        return self.model.predict(X_input)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Оценка качества модели по метрикам из исходного кода."""
        y_pred = np.array(self.predict(X_test)).ravel()
        y_pred_safe = y_pred * self.safe_factor

        errors = np.abs(y_test - y_pred)
        e_raw = y_pred - y_test
        e_safe = y_pred_safe - y_test

        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "Accuracy_1MB": np.mean(errors <= 1.0),
            "Accuracy_5MB": np.mean(errors <= 5.0),
            "OOM_Rate_Raw": np.mean(y_pred < y_test),
            "Max_OOM_Error_Raw": np.min(e_raw),
            "OOM_Rate_Safe": np.mean(y_pred_safe < y_test),
            "R2_Safe": r2_score(y_test, y_pred_safe),
            "Max_OOM_Error_Safe": np.min(e_safe)
        }

        print("\n--- РЕЗУЛЬТАТЫ МОДЕЛИ LGBM (предсказание base_pmu)---")
        for metric, value in metrics.items():
            fmt = ".2%" if "Rate" in metric or "Accuracy" in metric else ".4f"
            print(f"{metric}: {value:{fmt}}")

        return metrics
