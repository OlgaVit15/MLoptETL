import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyCascadeClassifier:
    """
    Класс-каскад для предсказания стратегии выполнения (DOP, Join Mode и т.д.)
    и вычисления скорректированного лимита памяти.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.target_column = 'strategy_label'
        self.model: Optional[Pipeline] = None

        # Метаданные, вычисляемые во время обучения
        self.strategy_map: Dict[str, Dict[str, Any]] = {}
        self.pmu_adjustments: Dict[str, float] = {}
        self.fallback_label: Optional[str] = None
        self.feature_columns: Optional[list] = None

    @staticmethod
    def _create_strategy_label(df: pd.DataFrame) -> pd.Series:
        """Внутренний метод генерации уникального идентификатора стратегии."""
        return (
                "DOP" + df['target_mt_dop'].astype(int).astype(str) + "_" +
                "TH" + df['target_num_scanner_threads'].astype(int).astype(str) + "_" +
                "JM" + df['target_default_join_distribution_mode'].astype(str).str.upper() + "_" +
                "CG" + df['target_disable_codegen'].astype(str).str.upper()
        )

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_list = df.apply(self.extract_features_s, axis=1)
        features_df = pd.DataFrame(features_list.tolist(), index=df.index)

        return features_df.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def extract_features_s(df: dict) -> list:
        """Генерация каскадных и физических признаков."""
        # 1. КАСКАДНЫЙ ПРИЗНАК
        log_base_pmu = np.log1p(df['base_pmu'])
        # 2. ФИЗИЧЕСКИЕ ПАРАМЕТРЫ ПЛАНА
        scans = df['feature_plan_num_scan_nodes']
        joins = df['feature_plan_num_joins']
        aggs = df['feature_plan_num_agg_nodes']
        vol_theory = df['feature_plan_max_cardinality'] * df['feature_plan_max_row_size']
        log_scan_size = np.log1p(df['feature_plan_total_scan_size_bytes'])
        # 3. ПРОИЗВОДНЫЕ
        nodes = df['feature_plan_num_scan_nodes']
        pmu_per_node = df['base_pmu'] / nodes
        row_size = df['feature_plan_max_row_size']

        return [
            log_base_pmu,
            scans,
            joins,
            aggs,
            vol_theory,
            log_scan_size,
            nodes,
            pmu_per_node,
            row_size
        ]

    def _get_model_pipeline(self, n_samples: int) -> Pipeline:
        """Создание пайплайна с сохранением всех гиперпараметров CatBoost."""
        cb_params = {
            'iterations': 4000,
            'learning_rate': 0.02,
            'depth': 7,
            'l2_leaf_reg': 3,
            'boosting_type': 'Ordered',
            'loss_function': 'MultiClass',
            'auto_class_weights': 'Balanced',
            'random_seed': self.random_state,
            'bootstrap_type': 'MVS',
            'verbose': 500,
            'early_stopping_rounds': 200
        }

        feature_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('quantile', QuantileTransformer(
                output_distribution='normal',
                n_quantiles=min(n_samples, 500),
                random_state=self.random_state
            ))
        ])

        model = CatBoostClassifier(**cb_params)
        return Pipeline([('preprocessor', feature_pipe), ('clf', model)])

    def train(self, df_tr: pd.DataFrame, df_t: pd.DataFrame):
        """
        Полный цикл обучения:
        1. Обучение классификатора.
        2. Формирование карты стратегий и коэффициентов запаса.
        """
        X_train = self.extract_features(df_tr)
        y_train = df_tr[self.target_column]
        y_test = df_t[self.target_column]
        X_test = df_t.drop(columns=self.target_column)

        self.feature_columns = X_train.columns.tolist()

        logger.info(f"Запуск обучения на {len(y_train.unique())} уникальных стратегий...")
        self.model = self._get_model_pipeline(len(X_train))
        self.model.fit(X_train, y_train)

        # --- КАРТА СТРАТЕГИЙ (на основе тренировочных данных) ---
        df_train = df_tr.loc[X_train.index]
        self.fallback_label = y_train.mode()[0]

        self.strategy_map = {}
        self.pmu_adjustments = {}

        for label in y_train.unique():
            group = df_train[df_train[self.target_column] == label]
            sample = group.iloc[0]

            # Параметры сессии
            self.strategy_map[label] = {
                'mt_dop': int(sample['target_mt_dop']),
                'threads': int(sample['target_num_scanner_threads']),
                'join_mode': sample['target_default_join_distribution_mode'],
                'codegen': sample['target_disable_codegen']
            }

            # Коэффициент запаса по памяти
            ratios = group['metric_pmu'] / group['base_pmu']
            self.pmu_adjustments[label] = float(np.percentile(ratios, 70))

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
        X_input = np.array([self.extract_features_s(row)])
        y_pred_label = self.model.predict(X_input)[0][0]

        label = y_pred_label

        # Получаем метаданные из карт (с использованием fallback)
        strat_params = self.strategy_map.get(label, self.strategy_map[self.fallback_label])
        adj = self.pmu_adjustments.get(label, 1.2)

        base_pmu = row['base_pmu']

        pred_mem_limit = base_pmu * adj

        res = {
            'strategy_label': label,
            'pred_mem_limit': pred_mem_limit,
            **strat_params
        }

        return res

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, full_df: pd.DataFrame) -> Dict[str, Any]:
        """Расчет метрик качества каскада."""
        results_pred = self.predict(X_test)
        actuals = full_df.loc[X_test.index]

        # 1. Точность классификации
        acc_strategy = accuracy_score(y_test, results_pred['strategy_label'])
        # acc_dop = (actuals['target_mt_dop'].values == results_pred['mt_dop'].values).mean()
        acc_dop = accuracy_score(actuals['target_mt_dop'], results_pred['mt_dop'])

        # 2. Метрики памяти
        y_actual_mem = actuals['metric_pmu'].round(0).astype(int)
        y_pred_mem = results_pred['pred_mem_limit']

        r2_mem = r2_score(y_actual_mem, y_pred_mem)
        mae_mem = mean_absolute_error(y_actual_mem, y_pred_mem)
        oom_rate = (y_pred_mem < y_actual_mem).mean()

        print("\n" + "=" * 40)
        print("РЕЗУЛЬТАТЫ Catboost Classifier")
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
