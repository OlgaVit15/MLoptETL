import asyncpg
import json
import logging


class MLHistoryManager:
    def __init__(self, dsn: str, model_version: str):
        self.dsn = dsn
        self.model_version = model_version
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.dsn)

    async def get_ml_cache(self, q_hash: str):
        """Ищем именно предсказание модели для конкретной версии"""
        if not self.pool: return None
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT ml_prediction FROM public.ml_prediction_cache 
                       WHERE query_hash = $1 AND model_version = $2""",
                    q_hash, self.model_version
                )
                return json.loads(row['ml_prediction']) if row else None
        except Exception as e:
            logging.error(f"Cache Read Error: {e}")
            return None

    async def save_ml_prediction(self, q_hash: str, ml_prediction: dict):
        print(f"q = {q_hash}")
        """Сохраняем только результат модели"""
        if not self.pool:
            print("Connection pool is not initialized")
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO public.ml_prediction_cache (query_hash, model_version, ml_prediction)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (query_hash, model_version) DO NOTHING
                """, q_hash, self.model_version, json.dumps(ml_prediction))
        except Exception as e:
            logging.error(f"Cache Write Error: {e}")
