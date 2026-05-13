import duckdb
import pandas as pd
import random
import logging

import regex
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
        """Генерация с нуля с гарантированными ON-clause."""
        try:
            fact = random.choice(['call_center', 'catalog_page', 'catalog_returns',
                                  'catalog_sales', 'customer', 'customer_address', 'customer_demographics',
                                  'household_demographics', 'income_band', 'inventory',
                                  'item', 'promotion', 'reason', 'ship_mode',
                                  'store', 'store_returns', 'store_sales', 'time_dim', 'warehouse', 'web_page',
                                  'web_returns',
                                  'web_sales', 'web_site'])
            selected_tables = {fact}
            joins = []

            for _ in range(2):
                # range(random.randint(2, 4)):
                possible = [r for r in self.relations if (r[0] in selected_tables and r[1] not in selected_tables) or (
                        r[1] in selected_tables and r[0] not in selected_tables)]
                # print(f"relations {self.relations}")
                if not possible:
                    break
                # else:
                # print(f"possible!")
                t1, t2, col = random.choice(possible)
                t_new = t2 if t1 in selected_tables else t1
                join_type = ["INNER JOIN", "LEFT JOIN", "LEFT JOIN [SHUFFLE]", "LEFT ANTI JOIN",
                             'FULL JOIN']
                t1c = f'{"inv" if t1 == "inventory" else "".join(regex.findall(r'(?<=^|_)\w', t1))}_{col}'
                t2c = f'{"inv" if t2 == "inventory" else "".join(regex.findall(r'(?<=^|_)\w', t2))}_{col}'
                joins.append(f"{join_type[random.randint(0, len(join_type))]} {t_new} ON {t1}.{t1c} = {t2}.{t2c}")
                selected_tables.add(t_new)

            cols = []
            for t in selected_tables:
                cols.append(f"{t}.{random.choice(self.schema[t])}")
            expr_cols = ', '.join(cols[:5])
            if random.randint(1, 100) % 2 == 0:
                sql = f"SELECT {expr_cols} FROM {fact} {' '.join(joins)} LIMIT {random.randint(50, 500)}"
            else:
                sql = f"SELECT {expr_cols}, count(*) FROM {fact} {' '.join(joins)} GROUP BY {expr_cols} LIMIT {random.randint(50, 500)}"
            return sql
        except:
            return None

    def generate_all(self, count_per_tpl=5, smith_count=500, add_orig=True):
        final_queries = set()

        # 1. Обработка шаблонов TPC-DS
        for tpl in self.base_templates:
            try:
                # используем sqlglot для исправления диалекта
                tree = parse_one(tpl, read='duckdb')

                # Добавляем оригинал
                if add_orig:
                    final_queries.add(tree.sql(dialect='postgres'))

                # for _ in range(count_per_tpl):
                #     mutated = self.mutate_tree(tree.copy())
                #     # Принудительная типизация при генерации SQL
                #     sql = mutated.sql(dialect='postgres')
                #     final_queries.add(sql)
            except:
                continue

        # 2. SQLsmith (с исправленной логикой префиксов)
        # for _ in range(smith_count):
        #     raw = self.gen_sqlsmith_safe()
        #     if raw:
        #         try:
        #             # Прогоняем через transpile для унификации стиля
        #             sql = transpile(raw, read='postgres', write='postgres')[0]
        #             final_queries.add(sql)
        #         except:
        #             continue

        return list(final_queries)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = PGSQLGenerator()
    # Уменьшаем кол-во для теста, чтобы убедиться в качестве
    results = gen.generate_all(count_per_tpl=12, smith_count=50000)

    # while len(results) < 50000:
    #     results.extend(gen.generate_all(count_per_tpl=12, smith_count=10000, add_orig=False))

    df = pd.DataFrame(results, columns=['sql_text'])
    # Удаляем пустые селекты и артефакты
    df = df[df['sql_text'].str.contains("SELECT .", case=False, regex=True)]
    # убираем точки с запятой в середине, если они есть
    df['sql_text'] = df['sql_text'].str.replace(';', '')

    df.to_csv("test_workload_tpcds_pg.csv", index=True, index_label='query_id', sep=';', encoding='utf8')
    print(f"Generated {len(df)} valid queries for PostgreSQL.")
