import json
import logging
import re


class ProfilerParser:
    @staticmethod
    def parse_to_bytes(val):
        if val is None or val == '': return 0
        val = str(val).upper().strip()
        # Импала может возвращать "1.50 GB (1610612736)" - берем то, что в скобках или до скобок
        # Но чаще просто "1.50 GB".
        units = {'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, 'TB': 1024 ** 4}

        # Ищем число (дробное) и единицу измерения
        match = re.search(r"([\d\.]+)\s*([KMBGT]*B)", val)
        if match:
            number, unit = match.groups()
            return int(float(number) * units.get(unit, 1))

        # Если это просто число в строке
        digits = re.findall(r"\d+", val)
        return int(digits[0]) if digits else 0

    @staticmethod
    def parse_to_ms(val):
        if val is None or val == '' or val == '0' or val == '0s': return 0.0
        val = str(val).lower().strip()

        # Регулярка для "1h2m3s4ms5us6ns"
        pattern = re.compile(r"([\d\.]+)(ns|us|ms|s|m|h)")
        matches = pattern.findall(val)

        total_ms = 0.0
        multipliers = {'ns': 1e-6, 'us': 1e-3, 'ms': 1.0, 's': 1000.0, 'm': 60000.0, 'h': 3600000.0}

        if matches:
            for num, unit in matches:
                total_ms += float(num) * multipliers.get(unit, 0)
            return total_ms

        # Если пришло чистое число (обычно наносекунды)
        try:
            return float(val) / 1_000_000.0
        except:
            return 0.0

    def _recursive_search(self, node, results):

        # 1. Парсим info_strings (здесь лежат PMU и Timeline)
        info_list = node.get('info_strings', [])
        for item in info_list:
            key = item.get('key')
            val = item.get('value')
            if not key or not val: continue

            if key == 'Per Node Peak Memory Usage':
                # Вытаскиваем все значения памяти из строки типа "1.2 GB (node1), 500 MB (node2)"
                mem_vals = re.findall(r"([\d\.]+\s*[KMBGT]*B)", val)
                if mem_vals:
                    current_max = max([self.parse_to_bytes(v) for v in mem_vals])
                    results['pmu'] = max(results['pmu'], current_max / (1024 ** 2))  # в MB

        # 2. Парсим counters (CpuTime, IO, Spill и т.д.)
        counters_list = node.get('counters', [])
        for c in counters_list:
            name = c.get('counter_name')
            val = c.get('value')
            if not name or val is None: continue

            if name == 'TotalCpuTime':
                results['cpu_ms'] += self.parse_to_ms(val)
            elif name == 'ScannerIoWaitTime':
                results['io_wait_ms'] += self.parse_to_ms(val)
            elif name == 'CodegenTotalWallClockTime':
                results['codegen_ms'] += self.parse_to_ms(val)
            elif name == 'TotalNetworkSendTime':
                results['network_send_ms'] += self.parse_to_ms(val)
            elif name == 'TotalNetworkReceiveTime':
                results['network_recieve_ms'] += self.parse_to_ms(val)
            elif name == 'BytesRead':
                results['bytes_read_s3'] += self.parse_to_bytes(val)
            elif name == 'ScratchBytesWritten':
                results['spill_bytes'] += self.parse_to_bytes(val)
            elif name == 'PeakScannerThreadConcurrency':
                try:
                    # Убираем лишние символы, если они есть
                    clean_val = re.findall(r"[\d\.]+", str(val))[0]
                    results['scan_concurrency'] = max(results['scan_concurrency'], float(clean_val))
                except:
                    pass

        # 3. Рекурсия по дочерним профилям
        children = node.get('child_profiles', [])
        for child in children:
            self._recursive_search(child, results)

    def get_full_metrics(self, profile_content) -> dict:
        try:
            if isinstance(profile_content, str):
                data = json.loads(profile_content)
            else:
                data = profile_content

            root = data.get('contents', data)

            results = {
                "duration_ms": 0.0, "pmu": 0.0, "spill_bytes": 0.0,
                "cpu_ms": 0.0, "io_wait_ms": 0.0,
                "codegen_ms": 0.0, "scan_concurrency": 0.0,
                "network_send_ms": 0.0, "network_recieve_ms": 0.0, "bytes_read_s3": 0.0
            }

            self._recursive_search(root, results)
            return results
        except Exception as e:
            logging.error(f"Error parsing JSON profile: {e}")
            return {}

    def execute_profile(self, profile_content, duration) -> dict:

        metrics = self.get_full_metrics(profile_content)
        metrics["duration_ms"] = duration*1000
        return metrics
