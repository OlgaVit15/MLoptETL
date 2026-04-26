import json

from impala.dbapi import connect


def create_table_imp():
    conn = connect(host='localhost',
                   port=21050,
                   auth_mechanism='NOSASL')
    cur = conn.cursor()
    try:
        cur.execute("set log_level =3")
        cur.execute(f"select count(*) from s3_tpcds.store_returns cross join s3_tpcds.promotions cross_join s3_tpcds.items")
        # cur.execute(f"select count(*) s3_tpcds.store_sales")
        cur.execute("profile")
        print(cur.fetchall())
        # er = cur.fetchall()
        # print(json.dumps(er))
        # ep = ExplainParser()
        # print(ep.parse(explain_rows=er))
        # i = ImpalaEnv()
        # print(i.get_cluster_state())
    except Exception as e:
        print(f"!failed: {e}")
    finally:
        if cur is not None:
            cur.close()


if __name__ == "__main__":
    create_table_imp()
