from sqlalchemy import create_engine

from dataset_pipeline.executor.impala.executor import Executor
from dataset_pipeline.pipeline.impala.m_pipeline import UltimateMLPipeline

settings = {
    "host": "localhost",
    "port": 21050,
    "auth_mechanism": 'NOSASL'
}

engine = create_engine('postgresql+psycopg2://suser:example@localhost:5432/model_tuning')
executor1 = Executor(settings, 's3_tpcds_medium')
csv = "D:\IdeaProjects\Ver1\model_service\education\dataset_pipeline\queries_generator\impala\workload_tpcds_extended.csv"
sf1 = 5
num_minutes = 300


def pipeline():
    pm1 = UltimateMLPipeline(executor1, engine)
    pm1.run(csv, sf1, num_minutes)


if __name__ == '__main__':
    pipeline()
