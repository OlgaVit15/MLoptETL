from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from model_ml.impala.model_predict.mem_strategy_cascade import ImpalaOptimizationCascade
from balancer.collect_service import ImpalaClusterCollector
from balancer.impala_balancer import AdvancedImpalaBalancer

IMPALA_COORD_URL = "http://impalad-1:25000"
POOL_NAME = "default-pool"
TRAIN_NODES = 3

model = ImpalaOptimizationCascade.load("model_ml/impala/model_predict/models/impala_cascade_20260425_175212.joblib")
collector = ImpalaClusterCollector(impala_coord_url=IMPALA_COORD_URL, pool_name=POOL_NAME)
balancer = AdvancedImpalaBalancer(train_nodes=TRAIN_NODES)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting Impala Collector...")
    collector.start()
    yield
    logging.info("Stopping Impala Collector...")
    collector._stop_event.set()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
def predict(data: dict):
    # 1. Получаем предсказание от ML-модели (сложность запроса)
    ml_pred = model.predict(data)
    print(f"ml predict: {ml_pred}")

    # 2. Получаем текущий снимок состояния кластера из памяти
    cluster_snapshot = collector.get_current_state()

    # 3. Корректируем предсказание под реальность кластера
    final_config = balancer.balance(ml_pred, cluster_snapshot)

    # Собираем итоговый ответ
    # Мы объединяем ML-данные (join mode, codegen) и сбалансированные ресурсы
    return {
            'mem_limit': final_config['target_mem_limit'],
            'mt_dop': final_config['target_mt_dop'],
            'num_scanner_threads': final_config['target_num_scanner_threads'],
            'default_join_distribution_mode': ml_pred['target_default_join_distribution_mode'],
            'disable_codegen': ml_pred['target_disable_codegen']
        }
    #     {
    #     'status': 'success',
    #     'config': {
    #         'mem_limit': final_config['target_mem_limit'],
    #         'mt_dop': final_config['target_mt_dop'],
    #         'num_scanner_threads': final_config['target_num_scanner_threads'],
    #         'default_join_distribution_mode': ml_pred['target_default_join_distribution_mode'],
    #         'disable_codegen': ml_pred['target_disable_codegen']
    #     },
    #     'metadata': {
    #         'group_id': ml_pred.get('group_id'),
    #         'cluster_state': final_config.get('balancer_log')
    #     }
    # }
