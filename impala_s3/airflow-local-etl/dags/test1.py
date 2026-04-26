from airflow.decorators import task, task_group
from airflow.exceptions import AirflowSkipException
from airflow.models.dag import dag
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from pendulum import datetime

from common.impala_operator import ImpalaExecuteOperator


def t(l: SQLExecuteQueryOperator):
    l.execute("select 1")


def ex(l: SQLExecuteQueryOperator):
    t(l)
    print("created")
    y = None
    sql = "select 1"
    if y is None:
        raise AirflowSkipException
    return sql


def tsk(l: SQLExecuteQueryOperator):
    l.execute(ex(l))


@dag(start_date=datetime(2021, 1, 1), schedule="@daily")
def create_dag():

    test = SQLExecuteQueryOperator(conn_id="impala-1", task_id="test1", sql="select 1")


create_dag()
