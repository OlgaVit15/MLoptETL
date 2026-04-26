import time

from impala.dbapi import connect

from dataset_pipeline.executor.impala.explain_paser import ExplainParser
from dataset_pipeline.executor.impala.profile_extract import ProfilerParser


class Executor:
    def __init__(self, settings: dict, db_schema: str):
        self.conn = connect(host=settings["host"], port=settings["port"], auth_mechanism=settings["auth_mechanism"])
        self.cur = self.conn.cursor()
        self.db = db_schema

    @staticmethod
    def get_plan(cur, query, q_metrics):
        ep_parser = ExplainParser()
        q_plan = f"explain {query}"
        cur.execute(q_plan)
        q_metrics["plan_metrics"] = ep_parser.parse(cur.fetchall())

    @staticmethod
    def get_profile(cur, duration, q_metrics):
        pp = ProfilerParser()
        profile = cur.get_profile(profile_format=3)
        pc = pp.execute_profile(profile, duration)
        q_metrics["profile_metrics"] = pc

    @staticmethod
    def set_params(cur, params):
        for key, val in params.items():
            q_set = f"set {key}={val}"
            cur.execute(q_set)

    def execute(self, query_id, query: str, params=None) -> dict:
        q_metrics = {"id": query_id, "session_params": params, "errors": [], "plan_metrics": None, "proffile_metrics": None}

        cur = self.cur

        duration = 0
        try:
            cur.execute(f"use {self.db};")
            cur.execute("SET EXEC_TIME_LIMIT_S = 250;")
            self.get_plan(cur, query, q_metrics)
            if params is None:
                start = time.perf_counter()
                cur.execute(query)
                end = time.perf_counter()
            else:
                self.set_params(cur, params)
                start = time.perf_counter()
                cur.execute(query)
                end = time.perf_counter()
            res = cur.fetchall()
            duration = end - start
            self.get_profile(cur, duration, q_metrics)
        except Exception as e:
            q_metrics["errors"].append(e)

        return q_metrics

