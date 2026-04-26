import json
import time

from impala.dbapi import connect

from dataset_pipeline.executor import Executor
from dataset_pipeline.explain_paser import ExplainParser
from dataset_pipeline.profile_extract import ProfilerParser


def save(path, content):
    with open(path, 'w', encoding='utf8') as f:
        f.write(content)


def query():
    settings = {
        "host": "localhost",
        "port": 21050,
        "auth_mechanism": 'NOSASL'
    }
    ex = Executor(settings)
    q = "select count(*) from s3_tpcds.store_returns"
    print(ex.execute(1, q))


if __name__ == "__main__":
    query()
