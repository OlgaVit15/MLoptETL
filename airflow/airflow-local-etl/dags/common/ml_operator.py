import logging
import requests
from abc import ABC, abstractmethod
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator


class BaseMLSQLOperator(SQLExecuteQueryOperator, ABC):
    """
    Абстрактный оператор для умного выполнения SQL с использованием ML-сервиса.
    """

    def __init__(
            self,
            *,
            sql: str,
            ml_service_url: str = None,
            configurations: dict = None,
            **kwargs
    ):
        super().__init__(sql=sql, **kwargs)
        self.ml_service_url = ml_service_url
        self.configurations = configurations or {}

    @abstractmethod
    def _get_features(self, cursor, query: str) -> dict:
        """
        Метод должен выполнить EXPLAIN и вызвать парсер для получения фичей.
        """
        pass

    @abstractmethod
    def _apply_session_params(self, cursor, params: dict):
        """
        Метод должен применить полученные параметры к текущей сессии (SET команды).
        """
        pass

    def _get_ml_params(self, features: dict) -> dict:
        """
        Общий метод запроса к ML-сервису.
        """
        if not features or not self.ml_service_url:
            return None

        try:
            response = requests.post(
                f"{self.ml_service_url}/predict",
                json=features,
                timeout=10  # Для ML лучше брать запас
            )
            response.raise_for_status()
            params = response.json()
            logging.info(f"ML Service returned: {params}")
            return params
        except Exception as e:
            logging.error(f"ML Service unavailable: {e}")
            return None

    def execute(self, context):
        hook = self.get_db_hook()

        with hook.get_conn() as conn:
            with conn.cursor() as cursor:
                # 1. Извлекаем фичи через специфичный для БД парсер
                features = self._get_features(cursor, self.sql)

                # 2. Получаем параметры от ML-сервиса
                ml_params = self._get_ml_params(features)

                # 3. Объединяем дефолтные конфиги и ML (ML в приоритете)
                final_params = {**self.configurations, **(ml_params or {})}

                # 4. Применяем параметры к сессии
                if final_params:
                    logging.info(f"Applying session parameters: {final_params}")
                    self._apply_session_params(cursor, final_params)

                # 5. Выполняем основной запрос
                logging.info(f"Executing main query: {self.sql}")
                cursor.execute(self.sql)

                # 6. Обработка результата (хук для специфичных действий БД типа profile)
                return self._post_execute(cursor, context)

    def _post_execute(self, cursor, context):
        """
        Дополнительные действия после запроса (например, логирование профиля).
        По умолчанию просто возвращает результат.
        """
        try:
            return cursor.fetchall()
        except Exception:
            return None
