import re
import duckdb
import pandas as pd
import random

import regex
from sqlglot import parse_one, exp, transpile


class ImpalaCorrectGenerator:
    def __init__(self, scale_factor=1):
        self.con = duckdb.connect(':memory:')
        self.con.execute(f"INSTALL tpcds; LOAD tpcds; CALL dsdgen(sf={scale_factor});")
        self.schema = self._get_schema()
        self.relations = self._generate_relations()
        raw_queries = self.con.execute("SELECT query FROM tpcds_queries()").fetchall()
        self.base_templates = [q[0] for q in raw_queries if q[0] and len(q[0]) > 50]

    def _get_schema(self):
        schema = {}
        tables = self.con.execute("SHOW TABLES").fetchall()
        for (tbl,) in tables:
            cols = self.con.execute(f"PRAGMA table_info('{tbl}')").fetchall()
            schema[tbl] = [c[1].lower() for c in cols]
        return schema

    def _generate_relations(self):
        rels = []
        # Набор связей для TPC-DS (на основе именования колонок _sk)
        all_tables = list(self.schema.keys())
        for i, t1 in enumerate(all_tables):
            for t2 in all_tables[i + 1:]:
                pattern = re.compile(r'^[^_]*_')  # Компилируем паттерн для скорости
                cleaned_t1 = [pattern.sub('', item) for item in self.schema[t1]]
                cleaned_t2 = [pattern.sub('', item) for item in self.schema[t2]]
                common = set(cleaned_t1) & set(cleaned_t2)
                for col in common:
                    if col.endswith('_sk'):
                        rels.append((t1, t2, col))
        print(f"rels {rels}")
        return rels

    def _clean_dangling_references(self, tree, removed_aliases):
        """Удаляет все выражения (Column, Filter, Group), использующие удаленные таблицы."""
        if not removed_aliases:
            return tree

        # Удаляем колонки из SELECT, WHERE, GROUP BY, ORDER BY
        for col in tree.find_all(exp.Column):
            if col.table.lower() in removed_aliases:
                # Находим родительский узел, который можно безопасно удалить
                # Например, если это часть AND, удаляем все условие
                parent = col.parent
                while parent and not isinstance(parent,
                                                (exp.Select, exp.Where, exp.Group, exp.Order, exp.Join, exp.Window)):
                    col = parent
                    parent = parent.parent
                col.pop()
        return tree

    def mutate_tree(self, tree):
        removed_aliases = set()

        # 1. Удаление JOIN-ов (с вычищением ссылок)
        joins = list(tree.find_all(exp.Join))
        if len(joins) > 2 and random.random() < 0.3:
            # Удаляем 1-2 случайных джойна
            for _ in range(random.randint(1, 2)):
                if not joins: break
                target = joins.pop(random.randint(0, len(joins) - 1))
                alias = target.alias_or_name.lower()
                removed_aliases.add(alias)
                target.pop()

        # Очистка дерева от удаленных алиасов
        tree = self._clean_dangling_references(tree, removed_aliases)

        # 2. Безопасная мутация констант и функций
        for node in tree.walk():
            # Исправляем аргументы функций (типа SUBSTRING)
            if isinstance(node, exp.Substring):
                for arg_key in ['start', 'length']:
                    arg = node.args.get(arg_key)
                    if isinstance(arg, exp.Literal):
                        arg.replace(exp.Literal.number(int(float(arg.this))))

            # Мутация чисел
            elif isinstance(node, exp.Literal) and node.is_number:
                val = float(node.this)
                # Если предок требует INT (Limit, Offset, Substring)
                if isinstance(node.parent, (exp.Limit, exp.Offset, exp.Substring)):
                    new_val = int(val)
                else:
                    # Для цен и коэффициентов
                    new_val = round(val * random.uniform(0.8, 1.2), 2)
                node.replace(exp.Literal.number(new_val))

            # Смена типа JOIN (только если есть ON clause)
            elif isinstance(node, exp.Join):
                if node.args.get("on") and random.random() < 0.2:
                    node.set("kind", random.choice(["LEFT", "INNER"]))

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

    def generate_all(self, count_per_tpl=12, smith_count=3000, add_orig=True):
        final_queries = set()

        # Обработка шаблонов
        for tpl in self.base_templates:
            try:
                # Добавляем оригинал
                if add_orig:
                    clean_orig = transpile(tpl, read='duckdb', write='hive')[0]
                    final_queries.add(clean_orig)

                for _ in range(count_per_tpl):
                    tree = parse_one(tpl, read='duckdb')
                    mutated = self.mutate_tree(tree)
                    # Транспиляция в диалект Hive (наиболее близок к Impala)
                    sql = transpile(mutated.sql(), read='duckdb', write='hive')[0]
                    if "ON" in sql.upper() or "JOIN" not in sql.upper():
                        final_queries.add(sql)
            except:
                continue

        # SQLsmith
        for _ in range(smith_count):
            raw = self.gen_sqlsmith_safe()
            if raw:
                try:
                    sql = transpile(raw, read='duckdb', write='hive')[0]
                    final_queries.add(sql)
                except:
                    continue

        return list(final_queries)


if __name__ == "__main__":
    gen = ImpalaCorrectGenerator()
    results = gen.generate_all(count_per_tpl=12, smith_count=2500)

    while len(results) < 2500:
        results.extend(gen.generate_all(count_per_tpl=12, smith_count=3000, add_orig=False))

    # Финальная очистка от артефактов
    df = pd.DataFrame(results, columns=['sql_text'])
    # Убираем запросы, которые могут вызвать ошибку из-за пустых SELECT после удаления колонок
    df = df[df['sql_text'].str.contains("SELECT .", case=False, regex=True)]

    df.to_csv("workload_tpcds_extended.csv", index=True, index_label='query_id', sep=';')
    print(f"Generated {len(df)} valid queries for Impala.")
