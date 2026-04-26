import duckdb
import psycopg2
import time
import logging

# --- НАСТРОЙКИ ---
pg_schema = "tpcds_big"  # схема в Postgres

# Параметры подключения к Postgres
PG_CONN_STR = "host=localhost port=5432 dbname=model_edu_pg user=suser password=example"

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Словарь для определения индексов ---
# Ключ: имя таблицы TPC-DS
# Значение: список колонок, которые нужно индексировать
# Можно расширить до tuple ('col1', 'col2') для композитных индексов
TPCDS_INDEXES = {
    # Фактовые таблицы (продажи) - критично для JOIN и фильтрации по дате/клиенту/товару
    'store_sales': [
        'ss_item_sk', 'ss_customer_sk', 'ss_sold_date_sk', 'ss_h_hour_sk',
        ('ss_sold_date_sk', 'ss_item_sk')  # Композитный
    ],
    'catalog_sales': [
        'cs_item_sk', 'cs_bill_customer_sk', 'cs_sold_date_sk', 'cs_call_center_sk',
        ('cs_sold_date_sk', 'cs_item_sk')
    ],
    'web_sales': [
        'ws_item_sk', 'ws_bill_customer_sk', 'ws_sold_date_sk', 'ws_warehouse_sk',
        ('ws_sold_date_sk', 'ws_item_sk')
    ],
    'store_returns': [
        'sr_item_sk', 'sr_customer_sk', 'sr_returned_date_sk'
    ],
    'catalog_returns': [
        'cr_item_sk', 'cr_return_customer_sk', 'cr_returned_date_sk'
    ],
    'web_returns': [
        'wr_item_sk', 'wr_customer_sk', 'wr_returned_date_sk'
    ],

    # Измерения (Dimensions) - первичные ключи и некоторые атрибуты для JOIN
    'item': ['i_item_sk', 'i_category', 'i_brand'],
    'customer': ['c_customer_sk', 'c_customer_id', 'c_current_h_hour_sk'],
    'customer_address': ['ca_address_sk', 'ca_city', 'ca_gmt_offset'],
    'customer_demographics': ['cd_demo_sk', 'cd_marital_status', 'cd_education_status'],
    'date_dim': ['d_date_sk', 'd_year', 'd_month', 'd_date'],  # Часто используется в WHERE
    'time_dim': ['t_time_sk', 't_hour', 't_minute'],
    'household_demographics': ['hd_demo_sk', 'hd_buy_potential'],
    'income_band': ['ib_income_band_sk'],
    'call_center': ['cc_call_center_sk', 'cc_city'],
    'catalog_page': ['cp_catalog_page_sk', 'cp_catalog_number'],
    'web_site': ['web_site_sk', 'web_name'],
    'web_page': ['wp_web_page_sk', 'wp_type'],
    'warehouse': ['w_warehouse_sk', 'w_warehouse_name'],
    'store': ['s_store_sk', 's_store_id'],
    'reason': ['r_reason_sk'],
    'promotion': ['p_promo_sk']
    # 'inventory': ['inv_item_sk', 'inv_warehouse_sk', 'inv_date_sk'],  # Композитные индексы
}


def setup_postgres_schema():
    """Создает схему в Postgres, если её нет"""
    with psycopg2.connect(PG_CONN_STR) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {pg_schema};")
            conn.commit()
    logging.info(f"--- Схема {pg_schema} готова ---")


def create_indexes_for_tpcds_table(table_name, schema_name, conn_str):
    """Создает индексы для указанной таблицы TPC-DS в PostgreSQL"""
    if table_name not in TPCDS_INDEXES:
        logging.info(f"[{table_name}] Нет определенных индексов. Пропускаем.")
        return

    columns_to_index = TPCDS_INDEXES[table_name]
    with psycopg2.connect(conn_str) as conn:
        conn.autocommit = True  # Для выполнения CREATE INDEX
        with conn.cursor() as cur:
            for col_def in columns_to_index:
                if isinstance(col_def, str):
                    # Простой индекс
                    index_name = f"idx_{table_name}_{col_def}"
                    create_idx_sql = f"CREATE INDEX {index_name} ON {schema_name}.{table_name} ({col_def});"
                elif isinstance(col_def, tuple):
                    # Композитный индекс
                    cols_str = ', '.join(col_def)
                    index_name = f"idx_{table_name}_{'_'.join(col_def)}"
                    create_idx_sql = f"CREATE INDEX {index_name} ON {schema_name}.{table_name} ({cols_str});"
                else:
                    logging.warning(f"[{table_name}] Неизвестный формат определения индекса: {col_def}")
                    continue

                try:
                    start_idx_t = time.time()
                    cur.execute(create_idx_sql)
                    logging.info(
                        f"[{table_name}] Создан индекс '{index_name}' за {round(time.time() - start_idx_t, 2)} сек.")
                except psycopg2.Error as e:
                    logging.error(f"[{table_name}] Ошибка при создании индекса '{index_name}': {e}")
                    conn.rollback()  # Откатываем на случай сбоя CREATE INDEX CONCURRENTLY


def generate_tpcds_to_pg(scale_factor=1):
    # 1. Создаем соединение DuckDB
    conn = duckdb.connect(database=':memory:')

    try:
        logging.info(f"--- Загрузка расширений DuckDB ---")
        conn.execute("INSTALL tpcds; LOAD tpcds;")
        conn.execute("INSTALL postgres; LOAD postgres;")

        # 2. Подключение к PostgreSQL через DuckDB
        logging.info(f"--- Подключение к PostgreSQL ---")
        setup_postgres_schema()
        # Присоединяем базу Postgres к DuckDB
        conn.execute(f"ATTACH '{PG_CONN_STR}' AS pg_db (TYPE POSTGRES, SCHEMA '{pg_schema}');")

        # 3. Генерация данных TPC-DS в памяти DuckDB
        logging.info(f"--- Генерация данных TPC-DS (SF={scale_factor}) ---")
        conn.execute(f"CALL dsdgen(sf={scale_factor});")

        # 4. Получаем список таблиц
        tables = conn.execute("SHOW TABLES;").fetchall()

        for (table_name,) in tables:
            start_t = time.time()
            logging.info(f"[{table_name}] Перекачка данных -> PostgreSQL ({pg_schema}.{table_name})...")

            # Сначала удаляем, если есть
            with psycopg2.connect(PG_CONN_STR) as pg_conn:
                pg_conn.autocommit = True
                with pg_conn.cursor() as pg_cur:
                    pg_cur.execute(
                        f"DROP TABLE IF EXISTS {pg_schema}.{table_name} CASCADE;")  # CASCADE удалит зависимые индексы/FK

            # Создаем и копируем из DuckDB в Postgres
            conn.execute(f"CREATE TABLE pg_db.{table_name} AS SELECT * FROM main.{table_name};")

            # 6. СОЗДАНИЕ ИНДЕКСОВ
            create_indexes_for_tpcds_table(table_name, pg_schema, PG_CONN_STR)

            # 5. Сбор статистики (ANALYZE)
            logging.info(f"[{table_name}] Сбор статистики (ANALYZE)...")
            with psycopg2.connect(PG_CONN_STR) as pg_conn:
                pg_conn.autocommit = True
                with pg_conn.cursor() as pg_cur:
                    pg_cur.execute(f"ANALYZE {pg_schema}.{table_name};")

            end_t = time.time()
            logging.info(f"[{table_name}] Завершено за {round(end_t - start_t, 2)} сек.")

        logging.info("\n--- ГОТОВО! Данные и индексы в PostgreSQL ---")

    except Exception as e:
        logging.error(f"!!! Ошибка: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    generate_tpcds_to_pg(scale_factor=7)
