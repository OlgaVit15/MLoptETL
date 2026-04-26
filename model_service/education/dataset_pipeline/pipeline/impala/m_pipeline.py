import re
import pandas as pd
import time


class UltimateMLPipeline:
    def __init__(self, executor, engine):
        self.executor = executor
        self.engine = engine
        self.MAX_MEM_PC = 16000
        self.MAX_CORES = 12

        self.TABLE_OPTIMAL = 'ml_results_optimal'
        self.TABLE_FAILURES = 'ml_results_failures'

    @staticmethod
    def parse_mem_to_mb(val):
        if not val or str(val) == '0':
            return 0
        val = str(val).lower()
        match = re.search(r'(\d+)', val)
        if not match:
            return 0
        num = float(match.group(1))
        return num * 1024 if 'gb' in val else num

    def calculate_score(self, row, is_discovery=False):
        error_str = str(row.get('errors', "")).lower()
        if "minimum memory reservation" in error_str:
            return 'bad', 'rejected_limit'
        if "out of memory" in error_str or "oom" in error_str:
            return 'bad', 'oom'
        if row.get('errors'):
            return 'bad', 'execution_error'

        r = row.get('profile_metrics')
        if not r:
            return 'bad', 'no_metrics'

        dur = r['duration_ms'] + 1
        pmu = r['pmu']
        limit_mb = self.parse_mem_to_mb(row['session_params'].get('mem_limit', 0))
        curr_mt_dop = row['session_params'].get('mt_dop', 0)

        if r['spill_bytes'] > 0:
            return 'bad', 'spill'

        if not is_discovery:
            if limit_mb > 256 and pmu < 0.5 * limit_mb:
                return 'overhead', 'mem_waste'
            if (r['network_recieve_ms'] / dur) > 0.4 and dur > 500:
                return 'bad', 'network_congestion'
            if dur < 3000 and (r['codegen_ms'] / dur) > 0.4:
                return 'overhead', 'codegen_heavy'
            # Параллелизм
            if curr_mt_dop > 2 and r['scan_concurrency'] < (curr_mt_dop * 0.3):
                return 'overhead', 'parallelism_waste'

        return 'good', 'optimal'

    def adjust_settings(self, row, score, reason):
        new_set = row['session_params'].copy()
        r = row.get('profile_metrics', {})
        curr_limit = self.parse_mem_to_mb(new_set.get('mem_limit', '1024mb'))
        pmu = r.get('pmu', curr_limit)

        if score == 'bad':
            if reason in ['rejected_limit', 'oom', 'spill']:
                if reason == 'rejected_limit':
                    match = re.search(r"at least ([\d\.]+) MB", str(row['errors']))
                    new_limit = int(float(match.group(1)) * 1.15) if match else curr_limit * 1.5
                else:
                    new_limit = int(max(pmu, curr_limit) * 1.25)
                new_set['mem_limit'] = f"{min(new_limit, self.MAX_MEM_PC)}mb"
            elif reason == 'network_congestion':
                mode = new_set.get('default_join_distribution_mode', 'SHUFFLE')
                new_set['default_join_distribution_mode'] = 'BROADCAST' if mode == 'SHUFFLE' else 'SHUFFLE'
            else:
                new_set['mt_dop'] = min(int(new_set.get('mt_dop', 1)) + 2, self.MAX_CORES)
        elif score == 'overhead':
            if reason == 'mem_waste':
                new_set['mem_limit'] = f"{max(int(pmu * 1.2), 128)}mb"
            elif reason == 'codegen_heavy':
                new_set['disable_codegen'] = 'true'
            elif reason == 'parallelism_waste':
                new_set['mt_dop'] = max(int(r.get('scan_concurrency', 1)), 1)
                new_set['num_scanner_threads'] = 4
        return new_set

    def run(self, csv_path, sf, max_minutes):
        df_main = pd.read_csv(csv_path, sep=";", encoding='utf8')
        df_shuffled = df_main.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df_shuffled
        # df_shuffled.iloc[:5000]
        # df_remaining = df_shuffled.iloc[600:]
        # df_remaining.to_csv(csv_path, index=True, encoding='utf8', sep=";")
        start_time = time.time()

        print(f"🚀 Starting SMART Pipeline. Goals: Efficiency + Clean Labels.")
        print(len(df))

        for idx, row in df.iterrows():
            if (time.time() - start_time) / 60 > max_minutes: break

            q_id = row['query_id']
            sql = row['sql_text']

            # --- ФАЗА 1: DISCOVERY (mt_dop=0) ---
            params_max = {
                'mt_dop': 0, 'mem_limit': f"{self.MAX_MEM_PC}mb",
                'num_scanner_threads': 8, 'default_join_distribution_mode': 'SHUFFLE',
                'disable_codegen': 'false'
            }
            res_max = self.executor.execute(q_id, sql, params=params_max)
            score_max, _ = self.calculate_score({**res_max, 'session_params': params_max}, is_discovery=True)

            if score_max == 'bad':
            #     # self._save(row, res_max, params_max, 'failed_discovery', 'discovery_error', self.TABLE_FAILURES, sf)
                continue

            # Метрики из "идеального" прогона
            max_pmu_dop0 = res_max['profile_metrics']['pmu']
            duration_dop0 = res_max['profile_metrics']['duration_ms']

            # --- ФАЗА 2: SMART OPTIMIZATION ---
            # Принимаем решение о параллелизме на основе длительности
            if duration_dop0 < 500:
                initial_dop = 1
                print(f"[{q_id}] Ultra-light query detected ({duration_dop0}ms). Keep mt_dop=1")
            elif duration_dop0 < 2000:
                initial_dop = 2
                print(f"[{q_id}] Medium query detected ({duration_dop0}ms). Try mt_dop=2")
            else:
                initial_dop = 4
                print(f"[{q_id}] Heavy query detected ({duration_dop0}ms). Try mt_dop=4")

            params_opt = params_max.copy()
            params_opt.update({
                'mt_dop': initial_dop,
                'mem_limit': f"{int(max_pmu_dop0 * 1.3 + 32)}mb"  # +32MB запас на системные нужды
            })

            for attempt in range(3):
                res_opt = self.executor.execute(q_id, sql, params=params_opt)
                score_opt, reason_opt = self.calculate_score({**res_opt, 'session_params': params_opt})
                res_opt['max_mem_limit_dop0'] = max_pmu_dop0

                if score_opt == 'good':
                    self._save(row, res_opt, params_opt, 'good', 'optimal', self.TABLE_OPTIMAL, sf)
                    break
                else:
                    if reason_opt not in ['rejected_limit', 'oom', 'execution_error'] and attempt == 2:
                        self._save(row, res_opt, params_opt, score_opt, f"iter_{attempt}_{reason_opt}", self.TABLE_FAILURES,
                                   sf)
                    # Если причина - тормоза, а не нехватка памяти, увеличиваем mt_dop
                    if reason_opt == 'slow_general' and int(params_opt['mt_dop']) < self.MAX_CORES:
                        params_opt['mt_dop'] = int(params_opt['mt_dop']) + 2

                    params_opt = self.adjust_settings({**res_opt, 'session_params': params_opt}, score_opt, reason_opt)

    def _save(self, row, res, params, score, reason, table_name, sf):
        full_data = {**row.to_dict(), **res}
        full_data['session_params'] = params
        full_data['score'] = score
        full_data['reason'] = reason

        flat = self.flatten_results(full_data, sf)

        df_to_save = pd.DataFrame([flat])
        df_to_save.to_sql(table_name, self.engine, if_exists='append', index=False)

    def flatten_results(self, res, sf):
        if res is None: res = {}

        output = {
            'query_id': int(res.get('query_id', 0)),
            'score': str(res.get('score', '')),
            'reason': str(res.get('reason', '')),
            'sql_text': str(res.get('sql_text', '')),
            'max_mem_limit_dop0': float(res.get('max_mem_limit_dop0', 0)) if res.get('max_mem_limit_dop0') else 0.0,
            'errors': str(res.get('errors', '[]')),
            'sf': int(sf)
        }

        # 1. Параметры (Targets)
        s = res.get('session_params') or {}  # <--- ЗАЩИТА: если None, берем {}
        output.update({
            'target_mt_dop': str(s.get('mt_dop', '')),
            'target_mem_limit': str(s.get('mem_limit', '')),
            'target_num_scanner_threads': str(s.get('num_scanner_threads', '')),
            'target_default_join_distribution_mode': str(s.get('default_join_distribution_mode', '')),
            'target_disable_codegen': str(s.get('disable_codegen', ''))
        })

        # 2. План (Features)
        p = res.get('plan_metrics') or {}  # <--- ЗАЩИТА: если None, берем {}
        plan_cols = ['num_joins', 'num_broadcast_joins', 'num_scan_nodes', 'num_agg_nodes',
                     'num_files', 'has_missing_stats', 'total_scan_size_bytes',
                     'max_cardinality', 'max_row_size']
        for col in plan_cols:
            output[f'feature_plan_{col}'] = float(p.get(col, 0)) if p.get(col) is not None else 0.0

        # 3. Профиль (Metrics)
        m = res.get('profile_metrics') or {}  # <--- ЗАЩИТА: если None, берем {}
        metric_cols = ['duration_ms', 'pmu', 'spill_bytes', 'cpu_ms', 'io_wait_ms',
                       'codegen_ms', 'scan_concurrency', 'network_send_ms',
                       'network_recieve_ms', 'bytes_read_s3']
        for col in metric_cols:
            output[f'metric_{col}'] = float(m.get(col, 0)) if m.get(col) is not None else 0.0

        return output
