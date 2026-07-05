from datetime import datetime
from airflow.models.dag import dag

from common.impala_operator import ImpalaExecuteOperator


@dag(start_date=datetime(2026, 3, 15), schedule="@daily")
def create_dag():
    test = ImpalaExecuteOperator(task_id="test_with_bao", conn_id="impala",
                                 sql="select max(sr_customer_sk), count(distinct sr_customer_sk) from "
                                     "s3_tpcds_small.store_returns",
                                 configurations={"mem_limit": "4g", "request_pool": "default", "mt_dop": "2"},
                                 ml_service_url="http://host.docker.internal:4040")


create_dag()
