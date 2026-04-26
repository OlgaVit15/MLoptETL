import datetime

from airflow.sdk.definitions.decorators import dag
from airflow.sdk.definitions.decorators import task

from common.impala_operator import ImpalaExecuteOperator
from airflow import DAG


@dag(start_date=datetime.datetime(2021, 1, 1), schedule="@daily")
def create_dag():
    @task(task_id="test1")
    def ex():
        ex = ImpalaExecuteOperator
        print(ex.execute("select coordinator()"))
        print("created")

    ex = ex

    [ex]


create_dag()
