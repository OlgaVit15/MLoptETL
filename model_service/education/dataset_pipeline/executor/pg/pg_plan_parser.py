import json
import re
import logging


class PostgresPlanParser:
    def __init__(self):
        # Добавлены новые фичи для оценки сложности
        self.feature_cols = [
            'total_cost', 'plan_rows', 'plan_width',
            'num_joins', 'num_scans', 'num_aggs', 'num_sorts', 'num_filters',
            'num_index_scans', 'num_mem_nodes', 'total_scan_size_bytes', 'max_node_cost'
        ]
        self.metric_cols = [
            'duration_ms', 'actual_rows', 'temp_written_blocks',
            'shared_hit_blocks', 'shared_read_blocks', 'peak_memory_mb'
        ]

    @staticmethod
    def parse_mem_to_mb(val):
        """Парсинг памяти: Postgres может вернуть '25kB' или '10MB'"""
        if not val or str(val) == '0': return 0.0
        val = str(val).lower()
        match = re.search(r'(\d+)', val)
        if not match: return 0.0
        num = float(match.group(1))
        if 'gb' in val: return num * 1024.0
        if 'kb' in val: return num / 1024.0  # KB в MB
        return num

    def extract_features_and_metrics(self, explain_json_str):
        try:
            data = explain_json_str
            if isinstance(data, list):
                data = data[0]
            elif not isinstance(data, dict):
                data = json.loads(explain_json_str)
        except Exception as e:
            print(f"!!! Error parsing JSON: {e}")
            return {}, {}

        features = {col: 0 for col in self.feature_cols}
        metrics = {col: 0 for col in self.metric_cols}

        plan = data.get('Plan', {})
        metrics['duration_ms'] = data.get('Execution Time', 0)

        # Корневые фичи
        features['total_cost'] = plan.get('Total Cost', 0)
        features['plan_rows'] = plan.get('Plan Rows', 0)
        features['plan_width'] = plan.get('Plan Width', 0)

        self._walk(plan, features, metrics)
        return features, metrics

    def _walk(self, node, features, metrics):
        ntype = node.get('Node Type', '')

        # 1. Считаем "Мем-узлы" (Причина: каждый такой узел ест свой work_mem)
        mem_node_types = ['Sort', 'Hash', 'Materialize', 'Aggregate', 'WindowAgg']
        if any(m in ntype for m in mem_node_types):
            features['num_mem_nodes'] += 1

        # 2. Считаем типы узлов
        if 'Join' in ntype or 'Nested Loop' in ntype: features['num_joins'] += 1
        if 'Scan' in ntype:
            features['num_scans'] += 1
            if 'Index Scan' in ntype: features['num_index_scans'] += 1
        if 'Aggregate' in ntype: features['num_aggs'] += 1
        if 'Sort' in ntype: features['num_sorts'] += 1
        if 'Filter' in node: features['num_filters'] += 1

        # 3. Самый тяжелый узел (помогает выявить перекосы)
        features['max_node_cost'] = max(features['max_node_cost'], node.get('Total Cost', 0))

        # 4. Метрики блоков и памяти
        metrics['temp_written_blocks'] += node.get('Temp Written Blocks', 0)
        metrics['shared_hit_blocks'] += node.get('Shared Hit Blocks', 0)
        metrics['shared_read_blocks'] += node.get('Shared Read Blocks', 0)

        # Postgres пишет 'Peak Memory Usage' в планах Sort/Hash
        if 'Peak Memory Usage' in node:
            metrics['peak_memory_mb'] += self.parse_mem_to_mb(node['Peak Memory Usage'])

        # Объем сканирования
        node_reads = node.get('Shared Read Blocks', 0) + node.get('Local Read Blocks', 0)
        features['total_scan_size_bytes'] += node_reads * 8192  # 8KB block size

        # Находим максимальный Actual Rows по дереву
        metrics['actual_rows'] = max(metrics['actual_rows'], node.get('Actual Rows', 0))

        for child in node.get('Plans', []):
            self._walk(child, features, metrics)
