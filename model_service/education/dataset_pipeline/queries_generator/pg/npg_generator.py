import random
import pandas as pd
import duckdb
import hashlib
from sqlglot import parse_one, exp, transpile


class PostgresCorrectGenerator:
    def __init__(self, scale_factor=1):
        self.con = duckdb.connect(':memory:')
        self.con.execute(f"INSTALL tpcds; LOAD tpcds; CALL dsdgen(sf={scale_factor});")
        self.schema = self._get_schema()
        # Карта связей TPC-DS
        self.fk_map = {
            'store_sales': {'date_dim': 'ss_sold_date_sk', 'item': 'ss_item_sk', 'customer': 'ss_customer_sk',
                            'store': 'ss_store_sk'},
            'catalog_sales': {'date_dim': 'cs_sold_date_sk', 'item': 'cs_item_sk', 'customer': 'cs_bill_customer_sk',
                              'warehouse': 'cs_warehouse_sk'},
            'web_sales': {'date_dim': 'ws_sold_date_sk', 'item': 'ws_item_sk', 'customer': 'ws_bill_customer_sk',
                          'web_site': 'ws_web_site_sk'},
            'store_returns': {'item': 'sr_item_sk', 'customer': 'sr_customer_sk', 'date_dim': 'sr_returned_date_sk'},
            'inventory': {'item': 'inv_item_sk', 'warehouse': 'inv_warehouse_sk', 'date_dim': 'inv_date_sk'},
            'web_returns': {'item': 'wr_item_sk', 'customer': 'wr_refunded_customer_sk',
                            'date_dim': 'wr_returned_date_sk'}
        }
        self.fact_tables = {'store_sales', 'catalog_sales', 'web_sales', 'inventory', 'store_returns', 'web_returns'}

        raw_queries = self.con.execute("SELECT query FROM tpcds_queries()").fetchall()
        self.base_templates = [q[0] for q in raw_queries if q[0] and len(q[0]) > 50]

    def _get_schema(self):
        schema = {}
        tables = self.con.execute("SHOW TABLES").fetchall()
        for (tbl,) in tables:
            cols = self.con.execute(f"PRAGMA table_info('{tbl}')").fetchall()
            schema[tbl] = [c[1].lower() for c in cols]
        return schema

    def _is_safe_query(self, tree):
        """
        Глубокая проверка на 'картезианские бомбы' и опасные Self-Join.
        """
        try:
            # 1. Проверяем все таблицы в FROM и JOIN
            tables = list(tree.find_all(exp.Table))
            table_names = [t.name.lower() for t in tables]

            # ЗАЩИТА: Запрещаем Self-Join больших таблиц (как в твоем примере ws1, ws2)
            for ft in self.fact_tables:
                if table_names.count(ft) > 1:
                    return False

            # 2. Проверяем количество джойнов
            # Если таблиц много, а условий JOIN (ON) мало - это Cross Join
            joins = list(tree.find_all(exp.Join))
            if len(table_names) > 1 and len(joins) < (len(table_names) - 1):
                # Проверяем, нет ли скрытых джойнов в WHERE (запятые)
                where = tree.find(exp.Where)
                if not where:
                    return False  # Много таблиц без WHERE и без JOIN ON = Смерть

            # 3. Проверка на вложенные CTE (как ws_wh в твоем примере)
            # Если CTE используется в IN (SELECT * FROM CTE) более 1 раза - это риск
            ctes = list(tree.find_all(exp.With))
            if ctes:
                in_subqueries = list(tree.find_all(exp.In))
                if len(in_subqueries) > 1:
                    return False  # Слишком сложная логика зависимых подзапросов

            return True
        except:
            return False

    def _fix_implicit_joins(self, tree):
        """
        Трансформирует 'FROM a, b WHERE a.id = b.id' в явные 'JOIN'.
        Это гарантирует, что Postgres не выберет Cross Join случайно.
        """
        return tree.transform(lambda node: transpile(node.sql(), read=None, write='postgres')[0])

    def _scrub_references(self, tree, removed_aliases):
        if not removed_aliases:
            return tree
        for col in tree.find_all(exp.Column):
            if col.table.lower() in removed_aliases:
                parent = col.parent
                while parent and not isinstance(parent, (exp.Where, exp.Group, exp.Order, exp.Join, exp.Select)):
                    col = parent
                    parent = parent.parent
                col.pop()
        if not tree.selects:
            tree.select("*", copy=False)
        return tree

    def mutate_tree(self, tree):
        try:
            new_tree = tree.copy()
            removed_aliases = set()

            # Удаление джойнов
            joins = list(new_tree.find_all(exp.Join))
            if len(joins) > 1 and random.random() < 0.4:
                target = joins.pop(random.randint(0, len(joins) - 1))
                removed_aliases.add(target.alias_or_name.lower())
                target.pop()

            new_tree = self._scrub_references(new_tree, removed_aliases)

            # Мутация констант
            for literal in new_tree.find_all(exp.Literal):
                if literal.is_number:
                    val = float(literal.this)
                    new_val = int(val * random.uniform(0.5, 1.5)) if val > 1 else val
                    literal.replace(exp.Literal.number(new_val))

            return new_tree
        except:
            return None

    def gen_sqlsmith_safe(self):
        """Генерация с нуля через построитель, исключающий Cross Join."""
        try:
            fact = random.choice(list(self.fk_map.keys()))
            dims = self.fk_map[fact]
            selected_dims = random.sample(list(dims.keys()), k=random.randint(1, 2))

            # Строим через явные JOIN ON
            query = exp.select(f"{fact}.*").from_(fact)
            for d in selected_dims:
                fk = dims[d]
                pk = [c for c in self.schema[d] if c.endswith('_sk')][0]
                query = query.join(d, on=f"{fact}.{fk} = {d}.{pk}")

            query = query.limit(random.randint(50, 500))
            return query
        except:
            return None

    def generate_all(self, target_count=1000):
        final_queries = set()

        # 1. Из шаблонов
        for tpl in self.base_templates:
            try:
                # Читаем duckdb, пишем postgres (это убирает специфичные запятые)
                tree = parse_one(tpl, read='duckdb')

                # Проверяем оригинал на безопасность
                if self._is_safe_query(tree):
                    final_queries.add(tree.sql(dialect='postgres'))

                for _ in range(10):
                    mutated = self.mutate_tree(tree)
                    if mutated and self._is_safe_query(mutated):
                        sql = mutated.sql(dialect='postgres')
                        final_queries.add(sql)
            except:
                continue

        # 2. Добиваем SQLSmith-ом до нужного количества
        attempts = 0
        while len(final_queries) < target_count and attempts < target_count * 5:
            attempts += 1
            smith_tree = self.gen_sqlsmith_safe()
            if smith_tree and self._is_safe_query(smith_tree):
                final_queries.add(smith_tree.sql(dialect='postgres'))

        return list(final_queries)


if __name__ == "__main__":
    gen = PostgresCorrectGenerator()
    # Генерируем 5000 чистых запросов
    results = gen.generate_all(target_count=15000)

    df = pd.DataFrame(results, columns=['sql_text'])
    # Финальный фильтр: убираем всё, где есть запятая во FROM (старый синтаксис)
    # Это гарантирует, что остались только явные JOIN ON
    df = df[~df['sql_text'].str.contains(r'FROM\s+\w+\s*,\s*\w+', case=False, regex=True)]

    # Добавляем хеш для идентификации
    df['query_hash'] = df['sql_text'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())

    df.to_csv("workload_tpcds_pg.csv", index=True, index_label='query_id', sep=';')
    print(f"Генерация завершена. Создано {len(df)} безопасных запросов.")
