from fastapi import BackgroundTasks


class QueryOrchestrator:
    def __init__(self, model, collector, balancer, hasher, history_manager=None):
        self.model = model
        self.collector = collector
        self.balancer = balancer
        self.hasher = hasher
        self.history_manager = history_manager

    async def predict_and_balance(self, data: dict, background_tasks: BackgroundTasks):
        # Вычисляем хэш на основе переданных правил
        q_hash = self.hasher.compute_hash(data)

        ml_pred = None

        # 1. Проверяем кэш ML-предсказания
        if self.history_manager:
            ml_pred = await self.history_manager.get_ml_cache(q_hash)

        # 2. Инференс, если не нашли
        if ml_pred is None:
            ml_pred = self.model.predict_s(data)
            if self.history_manager:
                print("hist")
                background_tasks.add_task(
                    self.history_manager.save_ml_prediction,
                    q_hash, ml_pred
                )
            source = "model"
        else:
            source = "cache"

        # 3. Динамическая балансировка (всегда свежая!)
        state = self.collector.get_current_state()
        final_config = self.balancer.balance(ml_pred, state)

        final_config['meta'] = {
            'ml_source': source,
            'q_hash': q_hash
        }

        return final_config
