import numpy as np


class AdvancedImpalaBalancer:
    def __init__(self, train_nodes=3):
        self.train_nodes = train_nodes

    def _parse_mem(self, mem_val):
        if isinstance(mem_val, (int, float)): return float(mem_val)
        return float(str(mem_val).lower().replace('mb', '').strip())

    def balance(self, ml_output, cluster_state):
        print(f"curr state: {cluster_state}")
        # 1. Данные от ML-модели
        pred_dop = max(1, ml_output['target_mt_dop'])
        mem_dop0 = self._parse_mem(ml_output['target_mem_limit_dop0'])
        mem_mt_dop = self._parse_mem(ml_output['target_mem_limit'])

        # 2. Состояние кластера
        live_nodes = cluster_state.get('live_nodes', self.train_nodes)
        running_q = cluster_state.get('running_queries', 0)
        io_latency = cluster_state.get('io_queue', 0)

        # Считаем нагрузку памяти (%)
        mem_p = (cluster_state.get('admitted_mem_gb', 0) / cluster_state.get('max_mem_gb', 1)) * 100

        # 3. ГИБКАЯ КОРРЕКЦИЯ DOP
        current_dop = pred_dop

        # Оцениваем "тесноту" в кластере: сколько запросов на одну живую ноду
        queries_per_node = running_q / live_nodes

        # Если на каждую ноду приходится больше 10 запросов — снижаем параллелизм
        if queries_per_node > 10.0:
            current_dop = max(1, int(current_dop * 0.4))  # Снижаем на 60%
        elif queries_per_node > 1.5:
            current_dop = max(1, int(current_dop * 0.7))  # Снижаем на 30%

        # Если средняя задержка чтения с диска выше 60 мс — диски перегружены
        if io_latency > 150:  # Критично
            current_dop = max(1, min(current_dop, 1))  # Сбиваем до минимума
        elif io_latency > 60:
            current_dop = max(1, min(current_dop, pred_dop // 2))

        # 4. РАСЧЕТ ПАМЯТИ (DOP-Memory Trade-off)
        # Если мы снизили DOP, нужно вычислить промежуточное значение памяти
        if pred_dop > 1 and current_dop < pred_dop:
            # Линейная интерполяция между mem_dop0 (DOP=1) и mem_mt_dop (DOP=pred_dop)
            ratio = (current_dop - 1) / (pred_dop - 1)
            interpolated_mem_3nodes = mem_dop0 + (mem_mt_dop - mem_dop0) * ratio
            # Добавим небольшой буфер 10% за то, что запрос будет работать дольше
            interpolated_mem_3nodes *= 1.1
        else:
            interpolated_mem_3nodes = mem_mt_dop

        # 5. МАСШТАБИРОВАНИЕ ПО НОДАМ (Invariance)
        # Суммарная память на 3 нодах должна распределиться на текущее число нод
        final_mem_per_node = (interpolated_mem_3nodes * self.train_nodes) / live_nodes

        # 6. КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ (Safety First)
        # Если в очереди Admission уже висят запросы или память пула > 90%
        if cluster_state.get('queued_queries', 0) > 10 or mem_p > 90:
            current_dop = 1  # Переходим в минимальный режим
            final_mem_per_node = (mem_dop0 * self.train_nodes) / live_nodes

        # Округление (128MB для маленьких, 256MB для больших)
        step = 128 if final_mem_per_node < 512 else 256
        final_mem_mb = int(np.ceil(final_mem_per_node / step) * step)

        return {
            'target_mem_limit': f"{max(256, final_mem_mb)}mb",
            'target_mt_dop': int(current_dop),
            'target_num_scanner_threads': ml_output['target_num_scanner_threads'] if current_dop > 1 else 1,
            'balancer_log': {
                'load_factor': round(queries_per_node, 2),
                'mem_pressure': f"{round(mem_p)}%",
                'dop_scaled': f"{pred_dop}->{current_dop}",
                'nodes': live_nodes
            }
        }
