import os
from datetime import datetime

import joblib
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

from model_ml.impala.model_predict.mem_regressor import MemoryLimitPredictor
from model_ml.impala.model_predict.strategy_classifier import StrategyCascadeClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImpalaOptimizationCascade:
    """
    Каскадная модель:
    1. Регрессор предсказывает базовый объем памяти (base_pmu).
    2. Классификатор использует предсказанный base_pmu как фичу для выбора стратегии.
    3. Итоговый лимит корректируется согласно выбранной стратегии.
    """

    def __init__(self, random_state: int = 42, safe_factor: float = 1.2):
        self.random_state = random_state
        self.regressor = MemoryLimitPredictor(safe_factor=safe_factor, random_state=random_state)
        self.classifier = StrategyCascadeClassifier(random_state=random_state)
        self.is_fitted = False

    @staticmethod
    def _create_strategy_label(df: pd.DataFrame) -> pd.Series:
        """Создает метку стратегии (копия логики из вашего кода)."""
        return (
                "DOP" + df['target_mt_dop'].astype(int).astype(str) + "_" +
                "TH" + df['target_num_scanner_threads'].astype(int).astype(str) + "_" +
                "JM" + df['target_default_join_distribution_mode'].astype(str).str.upper() + "_" +
                "CG" + df['target_disable_codegen'].astype(str).str.upper()
        )

    def train(self, df: pd.DataFrame):
        """
        Каскадное обучение:
        1. Обучаем регрессор на реальных данных.
        2. Прогоняем тренировочные данные через регрессор, чтобы классификатор
           обучался на 'предсказанных' значениях (устойчивость к ошибкам первой модели).
        3. Обучаем классификатор.
        """

        logger.info("--- Разделение и стратификация ---")
        # Разделение для обучения каскада
        strat = df['strategy_label']
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=self.random_state, stratify=strat
        )

        logger.info("--- Этап 1: Обучение Регрессора (Memory) ---")
        # Обучаем регрессор
        X_test, y_test = self.regressor.train(df_train, df_test)
        res1 = self.regressor.evaluate(X_test, y_test)

        logger.info("--- Этап 2: Подготовка данных для Классификатора ---")

        # Чтобы классификатор был честным каскадом, мы подменяем реальный base_pmu
        # на предсказанный регрессором (так будет и в продакшене)
        df_train_cascaded = df_train.copy()
        pred_base_pmu_train = self.regressor.predict(df_train)
        df_train_cascaded['base_pmu'] = pred_base_pmu_train
        df_test_cascaded = df_test.copy()
        pred_base_pmu_test = self.regressor.predict(df_test)
        df_test_cascaded['base_pmu'] = pred_base_pmu_test

        logger.info("--- Этап 3: Обучение Классификатора (Strategy) ---")
        # Обучаем классификатор на данных, где base_pmu пришел из первой модели
        X_testl, y_testl = self.classifier.train(df_train_cascaded, df_test_cascaded)
        res2 = self.classifier.evaluate(X_testl, y_testl, df_test_cascaded)

        self.is_fitted = True
        return df_test  # Возвращаем оригинальный тест для финальной оценки

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Полный цикл инференса:
        Raw Data -> Regressor -> Pred Base PMU -> Classifier -> Strategy & Final Memory
        """
        if not self.is_fitted:
            raise ValueError("Каскад моделей не обучен!")

        # 1. Предсказание регрессора
        pred_base_pmu = self.regressor.predict(df_raw)

        # 2. Подготовка данных для классификатора (инъекция предсказания первой модели)
        df_for_clf = df_raw.copy()
        df_for_clf['base_pmu'] = pred_base_pmu

        # 3. Предсказание классификатора
        final_results = self.classifier.predict(df_for_clf)

        # Добавим для наглядности само предсказание первой модели в результат
        final_results['base_pmu'] = pred_base_pmu

        return final_results

    def evaluate_cascade(self, df_test: pd.DataFrame):
        """Комплексная оценка всей цепочки."""
        logger.info("Запуск финального тестирования каскада...")

        # Получаем итоговые предсказания
        results = self.predict(df_test)

        # Метрики классификации (DOP)
        acc_dop = accuracy_score(df_test['target_mt_dop'], results['mt_dop'])

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

    def save(self, folder_path: str = "models"):
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
        Использование: model = ImpalaOptimizationCascade.load("path/to/model.joblib")
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден.")

        logger.info(f"Загрузка модели из {file_path}...")
        model = joblib.load(file_path)

        if not isinstance(model, ImpalaOptimizationCascade):
            raise ValueError("Файл не является экземпляром ImpalaOptimizationCascade")

        logger.info("Модель успешно загружена.")
        return model
