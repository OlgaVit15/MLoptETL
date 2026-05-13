from datetime import datetime
from airflow.models.dag import dag

from common.postgres_execute_operator import PostgresMLExecuteOperator


@dag(start_date=datetime(2026, 3, 15), schedule="@daily")
def create_pg_dag():
    test_pg = PostgresMLExecuteOperator(task_id="test_pg", conn_id="pgc",
                                     sql="select * from public.ml_prediction_cache",
                                     ml_service_url="http://host.docker.internal:8001")


create_pg_dag()
