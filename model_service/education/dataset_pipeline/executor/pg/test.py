from dataset_pipeline.executor.pg.pg_executor import PostgresExecutor
from dataset_pipeline.executor.pg.pg_plan_parser import PostgresPlanParser


def main():
    cons = {"host": "localhost",
            "port": 5432,
            "dbname": "model_edu_pg",
            "user": "suser",
            "password": "example"
            }
    pe = PostgresExecutor(cons, "tpcds_medium")
    pp = PostgresPlanParser()
    params = {
        "statement_timeout": "1ms"
    }
    sql = "select max(item.i_rec_end_date), min(item.i_rec_start_date), count(*) from item join web_sales ON web_sales.ws_item_sk = item.i_item_sk;"
    res = pe.execute(sql, params)
    print(res)
    print(type(res))
    if 'errors' in str(res).lower() and 'timeout' in str(res).lower():
        print(f"  [!] Discovery timed out (> 1ms). Skipping heavy query.")
    f, m = pp.extract_features_and_metrics(res)
    r = {
        "f": f,
        "m": m
    }
    print(f"f = {r["f"]} r = {r["m"]}")


if __name__ == '__main__':
    main()
