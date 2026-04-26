import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class PostgresRepresentativityBooster:
    def __init__(self):
        self.strategy_flags = [
            'target_jit', 'target_enable_indexscan', 'target_enable_seqscan',
            'target_enable_bitmapscan', 'target_enable_hashjoin',
            'target_enable_mergejoin', 'target_enable_nestloop'
        ]
        self.original_cols = None  # Для сохранения оригинального набора колонок

    def _prepare_base_metrics(self, df):
        df = df.copy()
        # Преобразуем булевы флаги в числовые 0/1 или ON/OFF для консистентности
        for col in self.strategy_flags:
            if col in df.columns:
                # Убедимся, что есть данные перед преобразованием
                if not df[col].empty:
                    # Сначала пытаемся привести к bool, затем к int
                    try:
                        df[col] = df[col].astype(bool).astype(int)
                    except Exception:
                        # Если прямой каст не удался, попробуем через строки
                        df[col] = df[col].astype(str).str.upper().map(
                            {'ON': 1, 'TRUE': 1, 'YES': 1, 'OFF': 0, 'FALSE': 0, 'NO': 0}).fillna(0).astype(int)
                else:
                    df[col] = 0  # Если колонка пустая, ставим 0
            else:
                df[col] = 0  # Если колонки нет, ставим 0

        # Преобразуем другие числовые признаки, если они еще не числовые
        for col in ['target_max_parallel_workers_per_gather', 'target_work_mem_mb']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def boost_dataset(self, df_raw):
        df = self._prepare_base_metrics(df_raw)

        # 1. Осторожное прореживание скучных данных
        # Уменьшим дробь, чтобы не потерять слишком много реальных данных
        cheap_mask = (df['feature_total_cost'] < 1000) & (df['metric_planner_error_ratio'] < 1.5)
        frac_cheap = 0.2  # Уменьшаем с 0.3 до 0.2
        df_cheap = df[cheap_mask].sample(frac=frac_cheap, random_state=42) if any(cheap_mask) else df[cheap_mask]
        df_complex = df[~cheap_mask]
        df_processed = pd.concat([df_cheap, df_complex], ignore_index=True)

        augmented_data = [df_processed]

        # 2. Усиление сигнала ошибки (более мягко)
        # Фокусируемся на реальных высоких ошибках, не создаем новые
        high_error_mask = df_processed['metric_planner_error_ratio'] > 10
        high_error_data = df_processed[high_error_mask]

        if not high_error_data.empty:
            # Вместо резкого изменения флагов, попробуем немного "подкрутить"
            # метрики, которые могли бы привести к ошибке.
            # Например, если ошибка высока, возможно, cost_per_row был недооценен.
            v_error = high_error_data.copy()
            # Создаем небольшой шум вокруг существующих метрик, а не кардинально меняем
            # Увеличим cost_per_row (косвенно)
            v_error['feature_total_cost'] = v_error['feature_total_cost'] * 1.2
            v_error['feature_plan_rows'] = v_error['feature_plan_rows'] / 1.1  # Предполагаем, что rows были недооценены
            # Вносим небольшие изменения в флаги, но не радикально
            v_error['target_enable_nestloop'] = np.random.choice([0, 1], size=len(
                v_error))  # Случайный выбор, чтобы избежать фиксированного изменения
            augmented_data.append(v_error)

        # 3. Балансировка индексов (более осторожно)
        # Вместо OFF->ON, попробуем добавить кейсы, где bitmapscan или seqscan
        # использовались вместо indexscan при большом количестве index scans
        # (это может быть признаком проблем с индексами или их использования)
        heavy_idx_mask = df_processed['feature_num_index_scans'] > 5
        heavy_idx_data = df_processed[heavy_idx_mask]

        if not heavy_idx_data.empty:
            v_idx = heavy_idx_data.copy()
            # Предположим, что при большом количестве index scans, но при этом
            # был использован seqscan, это может быть "плохой" кейс,
            # который нужно представить.
            seqscan_mask = v_idx['target_enable_seqscan'] == 1
            v_idx_seq = v_idx[seqscan_mask]
            if not v_idx_seq.empty:
                # Добавляем эти данные, но не меняем их характеристики сильно
                # Может быть, просто увеличим их долю
                augmented_data.append(v_idx_seq.sample(frac=0.5, random_state=42))

        df_boosted = pd.concat(augmented_data, ignore_index=True)

        # Убираем строки с NaN после всех преобразований
        df_boosted.dropna(inplace=True)

        return self._apply_scale_factor(df_boosted)

    def _apply_scale_factor(self, df):
        parts = [df]
        # Уменьшим диапазоны масштабирования и сделаем их более плавными
        for scale in [0.8, 1.2]:  # Убираем 2.5, делаем меньше влияние
            scaled = df.copy()
            # Масштабируем признаки, которые скорее всего влияют на стоимость и время
            scaled['feature_plan_rows'] = (scaled['feature_plan_rows'] * scale).clip(1)
            scaled['feature_total_cost'] = (scaled['feature_total_cost'] * (scale ** 1.2)).clip(
                1)  # Слегка усилим влияние масштаба на стоимость
            scaled['metric_duration_ms'] = (scaled['metric_duration_ms'] * (scale ** 0.9)).clip(1)
            # Work_mem не стоит масштабировать так сильно, его лучше предсказывать
            # scaled['target_work_mem_mb'] = (scaled['target_work_mem_mb'] * scale).clip(4, 4096)
            parts.append(scaled)
        return pd.concat(parts, ignore_index=True).drop_duplicates()

    def prepare_for_training(self, df_raw):
        # Сохраняем оригинальные колонки для последующего использования
        self.original_cols = df_raw.columns.tolist()

        # Очистка имен и типов
        # Убедимся, что все флаги представлены как 0/1
        for f in self.strategy_flags:
            if f in df_raw.columns:
                try:
                    df_raw[f] = df_raw[f].astype(bool).astype(int)
                except Exception:
                    df_raw[f] = df_raw[f].astype(str).str.upper().map(
                        {'ON': 1, 'TRUE': 1, 'YES': 1, 'OFF': 0, 'FALSE': 0, 'NO': 0}).fillna(0).astype(int)
            else:
                df_raw[f] = 0  # Если колонки нет, ставим 0

        # Исключаем строки с NaN в целевых переменных
        target_cols = ['metric_planner_error_ratio', 'target_max_parallel_workers_per_gather',
                       'target_enable_hashjoin', 'target_enable_mergejoin', 'target_enable_nestloop',
                       'target_enable_indexscan', 'target_enable_seqscan', 'target_enable_bitmapscan']
        df_raw.dropna(subset=target_cols, inplace=True)

        # Создаем колонку для стратификации на ОРИГИНАЛЬНЫХ данных
        # Упрощаем стратификацию: только наличие NestLoop и Parallel Workers
        df_raw['strat_col'] = (
                (df_raw['target_enable_nestloop'] == 1).astype(str) + "_" +
                (df_raw['target_max_parallel_workers_per_gather'] > 0).astype(str)
        )

        # Разделяем данные
        # Убедимся, что после очистки от NaN у нас еще остались данные
        if df_raw.empty:
            raise ValueError("Нет данных для обучения после очистки NaN.")

        # Проверяем, что в strat_col есть более одной категории, если возможно
        if df_raw['strat_col'].nunique() < 2:
            print(
                "Предупреждение: Недостаточно различных категорий для стратификации. Используется случайное разделение.")
            train_raw, test_raw = train_test_split(
                df_raw, test_size=0.2, random_state=42
            )
        else:
            train_raw, test_raw = train_test_split(
                df_raw, test_size=0.2, random_state=42, stratify=df_raw['strat_col']
            )

        # Бустим ТОЛЬКО трейн
        df_train = self.boost_dataset(train_raw)

        # Убираем временные колонки
        cols_to_drop = ['strat_col']
        df_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        test_raw.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        return df_train, test_raw
