import numpy as np
import math


class PostgresBalancer:
    def __init__(self, cpu_cores=16):
        self.cpu_cores = cpu_cores  # Важно знать кол-во ядер для интерпретации load_avg

    @staticmethod
    def _round_to_power_of_2(mem_mb):
        if mem_mb < 64:
            return 64
        exponent = int(math.log2(mem_mb))
        return 2 ** exponent

    def balance(self, ml_pred, cluster_state):
        # 1. Данные от модели
        pred_mem = float(ml_pred.get('pred_work_mem_mb', 64))
        pred_dop = int(ml_pred.get('target_max_parallel_workers_per_gather', 0))

        # 2. Метрики состояния
        running_q = cluster_state.get('running_queries', 0)
        blocked_q = cluster_state.get('blocked_queries', 0)
        conn_p = cluster_state.get('active_connections_pct', 0)
        load_avg = cluster_state.get('load_avg', 0.0)

        mem_factor = 1.0
        final_dop = pred_dop

        # --- ЛОГИКА 1: КРИТИЧЕСКИЕ БЛОКИРОВКИ ---
        if blocked_q > 0:
            # Чтобы не усугублять Lock-конфликты
            final_dop = 0
            mem_factor = 0.25

            # --- ЛОГИКА 2: НАГРУЗКА НА CPU (LOAD AVG) ---
        # Если load_avg выше количества ядер - система перегружена
        load_ratio = load_avg / self.cpu_cores
        if load_ratio > 1.2:  # Перегрузка 20%+
            final_dop = 0
            mem_factor = min(mem_factor, 0.5)
        elif load_ratio > 0.8:  # Нагрузка высокая (80-120%)
            final_dop = max(0, final_dop // 2)
            mem_factor = min(mem_factor, 0.75)

        # --- ЛОГИКА 3: КОНКУРЕНЦИЯ ЗА СОЕДИНЕНИЯ ---
        if conn_p > 90:
            # Нет свободных слотов для воркеров - отключаем параллелизм
            final_dop = 0
        elif conn_p > 75:
            # Ограничиваем параллелизм, чтобы оставить слоты другим
            final_dop = min(final_dop, 2)

        # --- ЛОГИКА 4: КОЛИЧЕСТВО АКТИВНЫХ ЗАПРОСОВ ---
        # Если слишком много людей хотят памяти одновременно
        if running_q > (self.cpu_cores * 2):
            mem_factor = min(mem_factor, 0.5)

        # --- ИТОГОВЫЙ РАСЧЕТ ---
        # Считаем итоговую память и применяем правило "степени двойки"
        adjusted_mem = pred_mem * mem_factor
        final_mem_mb = self._round_to_power_of_2(adjusted_mem)

        return {
            'work_mem': f"{final_mem_mb}MB",
            'max_parallel_workers_per_gather': int(final_dop),
            "jit": ml_pred["target_jit"],
            "enable_hashjoin": ml_pred["target_enable_hashjoin"],
            "enable_indexscan": ml_pred["target_enable_indexscan"],
            'balancer_log': {
                'load_factor': round(load_ratio, 2),
                'running_q': running_q,
                'blocked': blocked_q,
                'mem_scaled': f"{pred_mem}->{final_mem_mb}",
                'dop_scaled': f"{pred_dop}->{final_dop}"
            }
        }
