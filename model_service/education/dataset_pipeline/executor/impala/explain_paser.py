import re


class ExplainParser:
    def __init__(self):
        # Оставляем только те регексы, которые нужны для выбранных 7 признаков
        self.re_cardinality = re.compile(r'cardinality=([\d\.KMBT]+)')
        self.re_size = re.compile(r'size=([\d\.KMBT]+B)')
        self.re_max_row_size = re.compile(r'row-size=([\d\.KMBT]*B?\d*)')
        self.re_missing_stats = re.compile(r'stats=unavailable')
        self.re_files = re.compile(r'files=(\d+)')

    @staticmethod
    def _convert_to_bytes(size_str):
        if not size_str or size_str == '0': return 0
        size_str = str(size_str).upper().strip()

        # Математика: используем 1024**N
        units = {'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, 'TB': 1024 ** 4}

        # Если на входе чистое число (байт)
        if size_str.isdigit():
            return int(size_str)

        match = re.match(r"([\d\.]+)\s*([KMBGT]*B?)", size_str)
        if not match: return 0

        number, unit = match.groups()
        if unit and not unit.endswith('B'): unit += 'B'
        if not unit or unit not in units: unit = 'B'

        return int(float(number) * units[unit])

    @staticmethod
    def _convert_cardinality(card_str):
        if not card_str:
            return 0
        card_str = str(card_str).upper().strip()
        # Математика: 10**N
        multipliers = {'K': 10 ** 3, 'M': 10 ** 6, 'B': 10 ** 9, 'T': 10 ** 12}

        match = re.match(r"([\d\.]+)([KMBT]?)", card_str)
        if not match:
            return 0
        number, unit = match.groups()
        return int(float(number) * multipliers.get(unit, 1))

    def parse(self, explain_rows):
        """
        Вход: результат cursor.fetchall() после EXPLAIN
        Выход: словарь с 7 ключевыми признаками
        """
        # Объединяем строки в единый текст для поиска
        full_text = "\n".join([row[0] if isinstance(row, (list, tuple)) else str(row) for row in explain_rows])

        features = {}

        # 1. num_joins (Сложность связей)
        features['num_joins'] = full_text.count('JOIN')

        # 2. num_broadcast_joins (Риск OOM)
        features['num_broadcast_joins'] = full_text.count('BROADCAST')

        features['num_scan_nodes'] = full_text.count('SCAN S3') + full_text.count('SCAN HDFS')  # Учитываем оба типа
        features['num_agg_nodes'] = full_text.count('AGGREGATE')

        # 3. num_files (Потенциал параллелизма)
        features['num_files'] = sum([int(x) for x in self.re_files.findall(full_text)])

        # 4. has_missing_stats (Фактор неопределенности)
        features['has_missing_stats'] = 1 if self.re_missing_stats.search(full_text) else 0

        # 5. total_scan_size_bytes (Объем данных)
        sizes = self.re_size.findall(full_text)
        features['total_scan_size_bytes'] = sum([self._convert_to_bytes(s) for s in sizes])

        # 6. max_cardinality (Нагрузка на память / хеш-таблицы)
        cards = self.re_cardinality.findall(full_text)
        features['max_cardinality'] = max([self._convert_cardinality(c) for c in cards]) if cards else 0

        # 7. max_row_size (Вес строки / ширина данных)
        row_sizes = self.re_max_row_size.findall(full_text)
        features['max_row_size'] = max([self._convert_to_bytes(r) for r in row_sizes]) if row_sizes else 0

        return features
