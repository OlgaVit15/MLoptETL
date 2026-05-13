import os

from fastapi import FastAPI, BackgroundTasks
from collector import ImpalaClusterCollector
from balancer import AdvancedImpalaBalancer
import joblib

from common.database import MLHistoryManager
from common.hasher import FeatureHasher
from common.orchestrator import QueryOrchestrator

dsn_pg = os.getenv("PG_META_URL")
dsn_impala = os.getenv("IMPALA_URL")
MODEL_VERSION = "v1.0.2026_05_09"

app = FastAPI()
model = joblib.load("ml_models/models/impala_cascade_20260510_090620.joblib")
collector = ImpalaClusterCollector(impala_coord_url=dsn_impala, pool_name="default")
balancer = AdvancedImpalaBalancer(train_nodes=3)

history = MLHistoryManager(
    dsn=dsn_pg,
    model_version=MODEL_VERSION
)

impala_hasher = FeatureHasher(
    strict_keys=[
        'feature_plan_num_joins',
        'feature_plan_num_broadcast_joins',
        'feature_plan_num_scan_nodes',
        'feature_plan_num_agg_nodes',
        'feature_plan_num_files',
        'feature_plan_has_missing_stats'
    ],
    fuzzy_keys=[
        'feature_plan_total_scan_size_bytes',
        'feature_plan_max_cardinality',
        'feature_plan_max_row_size'
    ]
)


orchestrator = QueryOrchestrator(model, collector, balancer, impala_hasher, history)


@app.on_event("startup")
async def startup():
    await history.connect()
    collector.start()


@app.post("/predict")
async def predict(data: dict, bg: BackgroundTasks):
    return await orchestrator.predict_and_balance(data, bg)
