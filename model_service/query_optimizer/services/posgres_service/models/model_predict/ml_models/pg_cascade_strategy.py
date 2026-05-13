import os
from datetime import datetime

import joblib
import pandas as pd
import logging

from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

from .err_regressor import ErrPredictor
from .pg_classification_mlcascade import StrategyCascadeClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_boolean(val):
    s = str(val).lower().strip()
    return 1 if s in ['on', 'true', '1', '1.0', 't'] else 0


class PGOptimizationCascade:

    def __init__(self, random_state: int = 42, safe_factor: float = 1.2):
        self.random_state = random_state
        self.regressor = ErrPredictor(safe_factor=safe_factor, random_state=random_state)
        self.classifier = StrategyCascadeClassifier(random_state=random_state)
        self.is_fitted = False

    @staticmethod
    def get_strat_label(row):
        jit = clean_boolean(row['target_jit'])
        idx = clean_boolean(row['target_enable_indexscan'])
        hj = clean_boolean(row['target_enable_hashjoin'])
        nl = clean_boolean(row['target_enable_nestloop'])
        dop = int(row['target_max_parallel_workers_per_gather'])
        return f"J{jit}_I{idx}_H{hj}_N{nl}_D{dop}"

    def train(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        logger.info("--- Этап 1: Обучение Регрессора (Memory) ---")
        # Обучаем регрессор
        X_test, y_test = self.regressor.train(df_train, df_test)
        res1 = self.regressor.evaluate(df_test, y_test)

        logger.info("--- Этап 2: Подготовка данных для Классификатора ---")
        df_train_cascaded = df_train.copy()
        pred_error_train = self.regressor.predict(df_train)
        df_train_cascaded['log_planner_error'] = pred_error_train
        df_test_cascaded = df_test.copy()
        pred_error_test = self.regressor.predict(df_test)
        df_test_cascaded['log_planner_error'] = pred_error_test

        logger.info("--- Этап 3: Обучение Классификатора (Strategy) ---")
        self.classifier.train(df_train_cascaded, df_test_cascaded)
        res2 = self.classifier.evaluate(df_test_cascaded)

        self.is_fitted = True
        return df_test  # Возвращаем оригинальный тест для финальной оценки

    def predict_s(self, row: dict) -> dict:
        if not self.is_fitted:
            raise ValueError("Каскад моделей не обучен!")
        pred_error = self.regressor.predict_s(row)
        row['log_planner_error'] = pred_error
        final_results = self.classifier.predict_s(row)
        final_results['log_planner_error'] = pred_error

        return final_results

    def predict(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in X_raw.iterrows():
            result = self.predict_s(row.to_dict())
            results.append(result)

        return pd.DataFrame(results, index=X_raw.index)

    def evaluate_cascade(self, df_test: pd.DataFrame):
        """Комплексная оценка всей цепочки."""
        logger.info("Запуск финального тестирования каскада...")

        # Получаем итоговые предсказания
        results = self.predict(df_test)

        # Метрики классификации (DOP)
        acc_dop = accuracy_score(df_test['target_enable_hashjoin'], results['target_enable_hashjoin'])
        print(classification_report(df_test['target_mt_dop'], results['mt_dop']))
        print(classification_report(df_test['target_num_scanner_threads'], results['threads']))
        print(classification_report(df_test['target_default_join_distribution_mode'], results['join_mode']))

        # Метрики памяти (Итоговый лимит vs Реальное потребление metric_pmu)
        y_actual = df_test['metric_pmu']
        y_pred = results['pred_mem_limit']

        r2 = r2_score(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        oom_rate = (y_pred < y_actual).mean()

        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ КАСКАДНОЙ МОДЕЛИ (END-TO-END)")
        print("=" * 50)
        print(f"Accuracy (DOP Selection):  {acc_dop:.2%}")
        print(f"R2 Score (Final Memory):   {r2:.4f}")
        print(f"MAE (Memory Error):        {mae:.2f} MB")
        print(f"OOM Risk Rate:             {oom_rate:.2%}")
        print("=" * 50)

        return {
            "acc_dop": acc_dop,
            "r2_memory": r2,
            "mae_memory": mae,
            "oom_rate": oom_rate
        }

    def save(self, folder_path: str = "ml_models/models"):
        """
        Сохраняет весь каскад моделей (регрессор + классификатор + маппинги) в файл.
        """
        if not self.is_fitted:
            raise ValueError("Нельзя сохранить необученную модель!")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(folder_path, f"impala_cascade_{timestamp}.joblib")

        logger.info(f"Сохранение модели в {file_path}...")
        # Сохраняем весь объект self
        joblib.dump(self, file_path)
        logger.info("Модель успешно сохранена.")
        return file_path

    @staticmethod
    def load(file_path: str):
        """
        Статический метод для загрузки обученного каскада из файла.
        Использование: ml_models = ImpalaOptimizationCascade.load("path/to/ml_models.joblib")
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден.")

        logger.info(f"Загрузка модели из {file_path}...")
        model = joblib.load(file_path)

        if not isinstance(model, PGOptimizationCascade):
            raise ValueError("Файл не является экземпляром ImpalaOptimizationCascade")

        logger.info("Модель успешно загружена.")
        return model
