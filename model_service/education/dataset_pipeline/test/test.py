import json
import re

import regex
from impala.dbapi import connect

from dataset_pipeline.executor.impala.executor import Executor


def create_table_imp():
    conn = {"host":'localhost',
            "port":21050,
            "auth_mechanism":'NOSASL'
            }
    cur = connect(host=conn["host"], port=conn["port"], auth_mechanism=conn["auth_mechanism"]).cursor()
    # cur = Executor(conn, "s3_tpcds_small")
    try:
        # print(cur.execute(1, f"SELECT count(*) FROM store_returns WHERE sr_item_sk > 0"))
        cur.execute("SET EXEC_TIME_LIMIT_S=2;")
        cur.execute(f"SELECT * FROM s3_tpcds_bmedium.store_returns, s3_tpcds_bmedium.item WHERE sr_item_sk > 0")
        print(cur.fetchall())
        # with open('example.json', 'a', encoding='utf-8') as file:
        #     file.write(json.dumps(cur.fetchall()))
    except Exception as e:
        print(f"!failed: {e}")


if __name__ == "__main__":
    create_table_imp()
