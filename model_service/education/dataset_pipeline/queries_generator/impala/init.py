import duckdb
from impala.dbapi import connect

bucket = "tpcds"
db_test = "s3_tpcds_test"
db = "s3_tpcds_medium"


def get_conn_url(b: str, d: str):
    return f"s3a://{b}/{d}"


def create_table_imp(name: str, path: str):
    conn = connect(host='localhost',
                   port=21050,
                   auth_mechanism='NOSASL')
    cur = conn.cursor()
    location_url = get_conn_url(bucket, db)
    try:
        cur.execute(f"create database if not exists {db} LOCATION '{location_url}'")
        cur.execute(f"use {db}")
        cur.execute(f"drop table if exists {db}.{name} purge")
        cur.execute(
            f"create external table {db}.{name} like parquet '{path}' stored as parquet location '{location_url}/{name}'")
        cur.execute(f"invalidate metadata {db}.{name}")
        cur.execute(f"compute stats  {db}.{name}")
    except Exception as e:
        print(f"!failed to create impala table {name}: {e}")
    finally:
        if cur is not None:
            cur.close()


def generate_tpcds_to_s3(scale_factor=1, s3_bucket='tpcds'):
    # 1. Создаем соединение (в памяти)
    conn = duckdb.connect(database=':memory:')
    try:
        print(f"--- Загрузка расширений ---")
        conn.execute("INSTALL tpcds;")
        conn.execute("LOAD tpcds;")
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")

        # 2. Настройка подключения к MinIO (S3)
        print(f"--- Настройка S3 (MinIO) ---")
        s3_config = f"""
        SET s3_endpoint='localhost:9000';
        SET s3_access_key_id='minioadmin';
        SET s3_secret_access_key='minioadmin';
        SET s3_use_ssl=false;
        SET s3_url_style='path';
        """
        conn.execute(s3_config)

        # 3. Генерация данных TPC-DS (Scale Factor 1 ≈ 1GB)
        print(f"--- Генерация данных TPC-DS (SF={scale_factor}) в памяти ---")
        # создаем таблицы в схеме main
        conn.execute(f"CALL dsdgen(sf={scale_factor});")

        # 4. Получаем список всех созданных таблиц
        tables = conn.execute("SHOW TABLES;").fetchall()

        print(f"--- Экспорт таблиц в Parquet на S3 ---")
        for (table_name,) in tables:
            s3_path = get_conn_url(bucket, db)
            path = f"{s3_path}/{table_name}/{table_name}.parquet"
            print(f"Экспорт {table_name} -> {s3_path}...")

            # Экспортируем каждую таблицу в формате Parquet
            # Используем компрессию snappy
            conn.execute(f"COPY {table_name} TO '{path}' (FORMAT PARQUET, COMPRESSION 'snappy');")
            create_table_imp(table_name, path)
        print("--- Готово! Данные на S3 ---")
    except Exception as e:
        print(f"!failed: {e}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    generate_tpcds_to_s3(scale_factor=5)
