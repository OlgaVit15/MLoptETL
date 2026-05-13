from dataset_pipeline.executor.pg.pg_executor import PostgresExecutor
from dataset_pipeline.executor.pg.pg_plan_parser import PostgresPlanParser


def main():
    cons = {"host": "localhost",
            "port": 5432,
            "dbname": "model_edu_pg",
            "user": "suser",
            "password": "example"
            }
    pe = PostgresExecutor(cons, "tpcds_big")
    pp = PostgresPlanParser()
    params = {
        # "statement_timeout": "1ms",
        # "enable_parallel_append": "on",
        "work_mem": "4MB",
        "max_parallel_workers_per_gather": 0,
        "jit": "off",
        "enable_hashjoin": "off",
        "enable_indexscan": "on"
    }
    sql = ("SELECT store_sales.ss_sales_price, store_sales.ss_wholesale_cost, customer_address.ca_county, customer_address.ca_state, catalog_sales.cs_ship_addr_sk, catalog_sales.cs_call_center_sk, customer.c_first_name, customer.c_last_name, item.i_product_name, item.i_formulation FROM catalog_sales INNER JOIN customer ON catalog_sales.cs_bill_customer_sk = customer.c_customer_sk INNER JOIN item ON catalog_sales.cs_item_sk = item.i_item_sk INNER JOIN store_sales ON store_sales.ss_customer_sk = customer.c_customer_sk INNER JOIN customer_address ON customer.c_current_addr_sk = customer_address.ca_address_sk WHERE 1 = 1 AND NOT customer.c_customer_id IS NULL LIMIT 155")
    res = pe.execute(sql, params)
    print(res)
    # print(type(res))
    # if 'errors' in str(res).lower() and 'timeout' in str(res).lower():
    #     print(f"  [!] Discovery timed out (> 1ms). Skipping heavy query.")
    f, m = pp.extract_features_and_metrics(res)
    r = {
        "f": f,
        "m": m
    }
    print(f"f = {r["f"]} r = {r["m"]}")


if __name__ == '__main__':
    main()
