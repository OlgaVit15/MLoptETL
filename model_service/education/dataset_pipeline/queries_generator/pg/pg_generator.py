import duckdb
import pandas as pd
import random
import logging
from sqlglot import parse_one, exp, transpile


class PGSQLGenerator:
    def __init__(self, scale_factor=1):
        self.con = duckdb.connect(':memory:')
        self.con.execute(f"INSTALL tpcds; LOAD tpcds; CALL dsdgen(sf={scale_factor});")
        self.schema = self._get_schema()
        # Карта связей TPC-DS (исправленная под реальные префиксы)
        self.tpcds_links = self._get_tpcds_links()

        raw_queries = self.con.execute("SELECT query FROM tpcds_queries()").fetchall()
        self.base_templates = [q[0] for q in raw_queries if q[0] and len(q[0]) > 50]

    def _get_schema(self):
        schema = {}
        tables = self.con.execute("SHOW TABLES").fetchall()
        for (tbl,) in tables:
            cols = self.con.execute(f"PRAGMA table_info('{tbl}')").fetchall()
            schema[tbl] = [c[1].lower() for c in cols]
        return schema

    def _get_tpcds_links(self):
        """Хардкод связей TPC-DS, так как имена колонок имеют разные префиксы"""
        return [
            ('store_sales', 'ss_item_sk', 'item', 'i_item_sk'),
            ('store_sales', 'ss_customer_sk', 'customer', 'c_customer_sk'),
            ('store_sales', 'ss_sold_date_sk', 'date_dim', 'd_date_sk'),
            ('store_sales', 'ss_store_sk', 'store', 's_store_sk'),
            ('catalog_sales', 'cs_item_sk', 'item', 'i_item_sk'),
            ('catalog_sales', 'cs_bill_customer_sk', 'customer', 'c_customer_sk'),
            ('catalog_sales', 'cs_sold_date_sk', 'date_dim', 'd_date_sk'),
            ('web_sales', 'ws_item_sk', 'item', 'i_item_sk'),
            ('web_sales', 'ws_bill_customer_sk', 'customer', 'c_customer_sk'),
            ('web_sales', 'ws_sold_date_sk', 'date_dim', 'd_date_sk'),
            ('inventory', 'inv_item_sk', 'item', 'i_item_sk'),
            ('inventory', 'inv_date_sk', 'date_dim', 'd_date_sk'),
            ('customer', 'c_current_addr_sk', 'customer_address', 'ca_address_sk')
        ]

    def _clean_dangling_references(self, tree, removed_aliases):
        if not removed_aliases:
            return tree

        def prune(node):
            if isinstance(node, exp.Column):
                if node.table.lower() in removed_aliases:
                    return None
            return node

        return tree.transform(prune)

    def mutate_tree(self, tree):
        removed_aliases = set()

        # 1. Удаление JOIN-ов
        joins = list(tree.find_all(exp.Join))
        if len(joins) > 1 and random.random() < 0.3:
            for _ in range(random.randint(1, max(1, len(joins) // 2))):
                target = joins.pop(random.randint(0, len(joins) - 1))
                removed_aliases.add(target.alias_or_name.lower())
                target.pop()

        tree = self._clean_dangling_references(tree, removed_aliases)

        # 2. МУТАЦИЯ КОНСТАНТ (Решение проблемы Substring и типов)
        for node in tree.walk():
            if isinstance(node, exp.Literal) and node.is_number:
                val = float(node.this)

                # Проверяем контекст: если это аргумент функции substring, limit, offset
                # или это колонка типа _sk (суррогатный ключ), нам нужен только INT
                parent = node.parent
                is_integer_required = isinstance(parent, (exp.Limit, exp.Offset, exp.Substring))

                # Также проверяем, не является ли это аргументом в функции (2-й и 3-й аргумент substring)
                if not is_integer_required and isinstance(parent, exp.Anonymous):
                    if parent.this.upper() == 'SUBSTRING':
                        is_integer_required = True

                if is_integer_required:
                    # Для индексов и длин строк меняем только на целые числа в небольшом диапазоне
                    new_val = int(max(1, val + random.randint(-1, 2)))
                    node.replace(exp.Literal.number(new_val))
                else:
                    # Для цен и коэффициентов мутируем как обычно
                    new_val = round(val * random.uniform(7, 13), 2)
                    node.replace(exp.Literal.number(new_val))

        return tree

    def gen_sqlsmith_safe(self):
        try:
            # Выбираем стартовую таблицу (факт)
            fact_tables = ['store_sales', 'catalog_sales', 'web_sales', 'inventory']
            fact = random.choice(fact_tables)
            selected_tables = {fact}
            joins = []

            # Пытаемся добавить 2-3 связанных таблицы
            for _ in range(random.randint(2, 4)):
                # Ищем возможные связи для уже выбранных таблиц
                possible = [
                    lnk for lnk in self.tpcds_links
                    if (lnk[0] in selected_tables and lnk[2] not in selected_tables) or
                       (lnk[2] in selected_tables and lnk[0] not in selected_tables)
                ]

                if not possible: break

                t1, c1, t2, c2 = random.choice(possible)
                t_new = t2 if t1 in selected_tables else t1
                joins.append(f"INNER JOIN {t_new} ON {t1}.{c1} = {t2}.{c2}")
                selected_tables.add(t_new)

            # Выбираем колонки с префиксом таблицы
            cols = []
            for t in selected_tables:
                # Берем 1-2 случайные колонки из каждой таблицы
                target_cols = random.sample(self.schema[t], k=min(2, len(self.schema[t])))
                for c in target_cols:
                    cols.append(f"{t}.{c}")

            if not cols: return None

            sql = f"SELECT {', '.join(cols)} FROM {fact} {' '.join(joins)} WHERE 1=1 "

            # Добавляем случайный фильтр для изменения плана
            filter_t = random.choice(list(selected_tables))
            filter_c = random.choice(self.schema[filter_t])
            sql += f" AND {filter_t}.{filter_c} IS NOT NULL LIMIT {random.randint(100, 1000)}"

            return sql
        except Exception as e:
            logging.error(f"SQLSmith error: {e}")
            return None

    def generate_all(self, count_per_tpl=5, smith_count=500):
        final_queries = set()

        # 1. Обработка шаблонов TPC-DS
        for tpl in self.base_templates:
            try:
                # используем sqlglot для исправления диалекта
                tree = parse_one(tpl, read='duckdb')

                # Добавляем оригинал
                final_queries.add(tree.sql(dialect='postgres'))

                for _ in range(count_per_tpl):
                    mutated = self.mutate_tree(tree.copy())
                    # Принудительная типизация при генерации SQL
                    sql = mutated.sql(dialect='postgres')
                    final_queries.add(sql)
            except:
                continue

        # 2. SQLsmith (с исправленной логикой префиксов)
        for _ in range(smith_count):
            raw = self.gen_sqlsmith_safe()
            if raw:
                try:
                    # Прогоняем через transpile для унификации стиля
                    sql = transpile(raw, read='postgres', write='postgres')[0]
                    final_queries.add(sql)
                except:
                    continue

        return list(final_queries)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = PGSQLGenerator()
    # Уменьшаем кол-во для теста, чтобы убедиться в качестве
    results = gen.generate_all(count_per_tpl=5, smith_count=1000)

    df = pd.DataFrame(results, columns=['sql_text'])
    # Удаляем пустые селекты и артефакты
    df = df[df['sql_text'].str.contains("SELECT .", case=False, regex=True)]
    # убираем точки с запятой в середине, если они есть
    df['sql_text'] = df['sql_text'].str.replace(';', '')

    df.to_csv("workload_tpcds_pg.csv", index=True, index_label='query_id', sep=';', encoding='utf8')
    print(f"Generated {len(df)} valid queries for PostgreSQL.")
