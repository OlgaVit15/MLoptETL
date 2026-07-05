import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from lightgbm import LGBMClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)


class StrategyCascadeClassifier:
    # Определяем имена фичей один раз, чтобы избежать ошибок именования
    FEATURE_NAMES = [
        'log_base_pmu', 'scans', 'joins', 'aggs', 'vol_theory',
        'log_scan_size', 'nodes', 'pmu_per_node', 'row_size'
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.target_column = 'strategy_label'
        self.model: Optional[Pipeline] = None
        self.strategy_map: Dict[Any, Dict[str, Any]] = {}
        self.pmu_adjustments: Dict[Any, float] = {}
        self.fallback_label: Optional[Any] = None
        self.label_encoder = LabelEncoder()  # Сохраняем энкодер здесь

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Всегда возвращает массив."""
        features_list = df.apply(self.extract_features_s, axis=1)
        arr = np.array(features_list.tolist(), dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    @staticmethod
    def extract_features_s(df: dict) -> list:
        # Логика извлечения остается прежней
        log_base_pmu = np.log1p(df.get('base_pmu', 0))
        scans = df.get('feature_plan_num_scan_nodes', 0)
        joins = df.get('feature_plan_num_joins', 0)
        aggs = df.get('feature_plan_num_agg_nodes', 0)
        vol_theory = df.get('feature_plan_max_cardinality', 0) * df.get('feature_plan_max_row_size', 0)
        log_scan_size = np.log1p(df.get('feature_plan_total_scan_size_bytes', 0))
        nodes = scans
        pmu_per_node = df.get('base_pmu', 0) / nodes if nodes > 0 else 0
        row_size = df.get('feature_plan_max_row_size', 0)

        return [log_base_pmu, scans, joins, aggs, vol_theory, log_scan_size, nodes, pmu_per_node, row_size]

    def _get_model_pipeline(self, n_samples: int) -> Pipeline:
        feature_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('quantile', QuantileTransformer(
                output_distribution='normal',
                n_quantiles=min(n_samples, 500),
                random_state=self.random_state
            ))
        ])

        lgbm_params = {
            'n_estimators': 2000,
            'learning_rate': 0.02,
            'max_depth': 7,
            'num_leaves': 64,
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
            'reg_lambda': 3
        }
        return Pipeline([('preprocessor', feature_pipe), ('clf', LGBMClassifier(**lgbm_params))])

    def train(self, df_tr: pd.DataFrame, df_t: pd.DataFrame):
        # 1. Готовим признаки
        X_train = self.extract_features(df_tr)

        # 2. Кодируем таргет (сохраняем энкодер в self)
        y_train = self.label_encoder.fit_transform(df_tr[self.target_column])
        y_test = self.label_encoder.transform(df_t[self.target_column])

        # 3. Обучаем модель
        logger.info(f"Обучение классификатора на {len(X_train)} примерах...")
        self.model = self._get_model_pipeline(len(X_train))
        self.model.fit(X_train, y_train)

        # 4. Настраиваем мапы и fallback
        self.fallback_label = pd.Series(y_train).mode()[0]
        self.strategy_map = {}
        self.pmu_adjustments = {}

        for label_idx in np.unique(y_train):
            group = df_tr.iloc[np.where(y_train == label_idx)]
            sample = group.iloc[0]

            self.strategy_map[label_idx] = {
                'mt_dop': int(sample['target_mt_dop']),
                'threads': int(sample['target_num_scanner_threads']),
                'join_mode': sample['target_default_join_distribution_mode'],
                'codegen': sample['target_disable_codegen']
            }
            ratios = group['metric_pmu'] / group['base_pmu']
            self.pmu_adjustments[label_idx] = float(np.percentile(ratios, 70))

        X_test_raw = df_t.drop(columns=[self.target_column])
        return X_test_raw, y_test

    def predict_s(self, row: dict) -> dict:
        """Инференс на одном словаре. Без DataFrame."""
        # 1. Извлекаем фичи в список
        feat_list = self.extract_features_s(row)
        # 2. В 2D массив NumPy
        X_input = np.array([feat_list], dtype=np.float32)
        # Убираем inf/nan
        X_input[~np.isfinite(X_input)] = 0

        # 3. Предсказание (модель ожидает массив, получает массив)
        label_idx = self.model.predict(X_input)[0]

        strat_params = self.strategy_map.get(label_idx, self.strategy_map[self.fallback_label])
        adj = self.pmu_adjustments.get(label_idx, 1.2)
        base_pmu = row.get('base_pmu', 0)

        return {
            'strategy_label': label_idx,
            'pred_mem_limit': base_pmu * adj,
            **strat_params
        }

    def predict(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Батч-предсказание для теста. Работает на массивах."""
        X_input = self.extract_features(X_raw)
        y_pred_indices = self.model.predict(X_input)

        base_pmus = X_raw['base_pmu'].values
        results = []
        for i, idx in enumerate(y_pred_indices):
            params = self.strategy_map.get(idx, self.strategy_map[self.fallback_label])
            results.append({
                'strategy_label': idx,
                'pred_mem_limit': base_pmus[i] * self.pmu_adjustments.get(idx, 1.2),
                **params
            })
        return pd.DataFrame(results, index=X_raw.index)

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray, full_df: pd.DataFrame) -> Dict[str, Any]:
        """Оценка."""
        # Теперь predict работает быстро
        results_pred = self.predict(X_test)

        acc_strategy = accuracy_score(y_test, results_pred['strategy_label'])
        actuals = full_df.loc[X_test.index]
        acc_dop = accuracy_score(actuals['target_mt_dop'], results_pred['mt_dop'])

        y_actual_mem = actuals['metric_pmu']
        y_pred_mem = results_pred['pred_mem_limit']

        r2_mem = r2_score(y_actual_mem, y_pred_mem)
        mae_mem = mean_absolute_error(y_actual_mem, y_pred_mem)
        oom_rate = (y_pred_mem < y_actual_mem).mean()

        print("\n" + "=" * 40)
        print("РЕЗУЛЬТАТЫ LGBM Cascade Classifier")
        print("=" * 40)
        print(f"Accuracy (Стратегия): {acc_strategy:.2%}")
        print(f"Accuracy (DOP):       {acc_dop:.2%}")
        print(f"R2 Memory:            {r2_mem:.4f}")
        print(f"MAE Memory:           {mae_mem:.2f} MB")
        print(f"OOM Rate:             {oom_rate:.2%}")
        print("=" * 40)

        return {
            "Accuracy_Strategy": acc_strategy,
            "Accuracy_DOP": acc_dop,
            "R2_Memory": r2_mem,
            "MAE_Memory": mae_mem,
            "OOM_Rate": oom_rate
        }
