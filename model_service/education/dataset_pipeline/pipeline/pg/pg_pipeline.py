from sqlalchemy import create_engine

from dataset_pipeline.executor.pg.pg_executor import PostgresExecutor
from dataset_pipeline.pipeline.pg.m_npg_pipeline import UltimatePGMLPipeline

EXPLAIN_FEATURE_COLS = [
    'total_cost', 'plan_rows', 'plan_width',
    'num_joins', 'num_scans', 'num_aggs', 'num_sorts', 'num_filters',
    'num_index_scans', 'index_cond_present', 'total_scan_size_bytes'
]


engine = create_engine('postgresql+psycopg2://suser:example@localhost:5432/model_tuning_pg')

engine_w = {"host": "localhost",
            "port": 5432,
            "dbname": "model_edu_pg",
            "user": "suser",
            "password": "example"
            }

csv = "D:/IdeaProjects/Ver1/model_service/education/dataset_pipeline/queries_generator/pg/workload_tpcds_pg.csv"
num_minutes = 300


def pipeline():
    executor1 = PostgresExecutor(engine_w, 'tpcds_big')
    pm1 = UltimatePGMLPipeline(executor1, engine, 30)
    pm1.run(csv, 10)


if __name__ == '__main__':
    pipeline()
