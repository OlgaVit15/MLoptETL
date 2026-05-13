import os

from fastapi import FastAPI, BackgroundTasks

from common.database import MLHistoryManager
from common.hasher import FeatureHasher
from common.orchestrator import QueryOrchestrator
from collector import PostgresStateCollector
from balancer import PostgresBalancer
import joblib

dsn_meta = os.getenv("DSN_PG")
dsn_pg = os.getenv("DSN_META")
MODEL_VERSION = "v1.0.2026_05_09"
app = FastAPI()
model = joblib.load("ml_models/models/impala_cascade_20260509_182026.joblib")
collector = PostgresStateCollector(dsn=dsn_pg, interval=5)
balancer = PostgresBalancer(cpu_cores=16)
history = MLHistoryManager(
    dsn=dsn_meta,
    model_version=MODEL_VERSION
)

pg_hasher = FeatureHasher(
    strict_keys=[
        'feature_num_joins',
        'feature_num_scans',
        'feature_num_index_scans',
        'feature_num_aggs',
        'feature_num_sorts',
        'feature_num_filters',
        'feature_plan_width',
        'feature_num_mem_nodes'
    ],
    fuzzy_keys=[
        'feature_total_cost',
        'feature_plan_rows',
        'feature_total_scan_size_bytes',
        'feature_max_node_cost'
    ]
)

orchestrator = QueryOrchestrator(model, collector, balancer, pg_hasher, history)


@app.on_event("startup")
async def startup():
    await history.connect()
    collector.start()


@app.post("/predict")
async def predict(data: dict, bg: BackgroundTasks):
    return await orchestrator.predict_and_balance(data, bg)
