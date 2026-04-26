from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

conn_id = "impala_1"


class ImpalaExecuteOperator(SQLExecuteQueryOperator):
    def __init__(self, *, sql: str | list[str], **kwargs):
        super().__init__(sql=sql, **kwargs)
        self.conn_id = conn_id
