import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from model_ml.pg.model_predict.err_regressor import ErrPredictor
from model_ml.pg.pg_classifier import PGStrategyClassifier

POSTGRES_URI = 'postgresql://suser:example@localhost:5432/model_tuning_pg'
RANDOM_STATE = 42
TABLE_NAME = 'ml_training_dataset'


def augment_and_fix_pg_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Исправляет 'пустые' фичи, заполняя их на основе метрик (восстановление реальности).
    2. Генерирует синтетические вариации для редких/тяжелых запросов.
    3. Корректирует таргеты там, где 'всё включено', но метрики говорят об обратном.
    """
    df_fixed = df.copy()

    # --- ШАГ 1: Восстановление 'пустых' фичей (Feature Imputation) ---
    # Если планировщик выдал 0 cost/rows, но запрос реально что-то читал или шел долго
    # мы прописываем минимальные значения, чтобы логарифм не уходил в бесконечность
    mask_zero_features = (df_fixed['feature_total_cost'] == 0) & (df_fixed['metric_duration_ms'] > 0)

    # Эвристика: если есть прочитанные блоки, восстанавливаем примерную стоимость
    df_fixed.loc[mask_zero_features, 'feature_total_cost'] = (
            df_fixed['metric_shared_read_blocks'] * 1.0 +
            df_fixed['metric_shared_hit_blocks'] * 0.1 +
            0.01
    )
    df_fixed.loc[df_fixed['feature_plan_rows'] == 0, 'feature_plan_rows'] = df_fixed['metric_actual_rows'].clip(lower=1)

    # --- ШАГ 2: Исправление Error Ratio (Самая важная фича) ---
    # Пересчитываем реальную ошибку: actual / planned
    # Если планировщик сказал 0 строк, а пришла 1, ошибка = 1.
    df_fixed['target_planner_error_ratio'] = (
            df_fixed['metric_actual_rows'] / (df_fixed['feature_plan_rows'] + 1e-5)
    ).clip(upper=100000)  # Ограничиваем сверху, чтобы не было бесконечности

    # --- ШАГ 3: Логическая коррекция таргетов (Expert Tuning) ---
    # Пример 1: Если запрос очень быстрый (< 50ms), JIT всегда должен быть OFF (оверхед)
    df_fixed.loc[df_fixed['metric_duration_ms'] < 50, 'target_jit'] = 'OFF'

    # Пример 2: Если реальных строк много (> 50k), а флаг NestLoop ON -
    # это "опасный" пример. Мы искусственно создаем копию, где NestLoop OFF.
    mask_heavy_nl = (df_fixed['metric_actual_rows'] > 50000) & (df_fixed['target_enable_nestloop'] == 'ON')
    df_heavy_nl_fix = df_fixed[mask_heavy_nl].copy()
    df_heavy_nl_fix['target_enable_nestloop'] = 'OFF'
    df_heavy_nl_fix['target_enable_hashjoin'] = 'ON'  # Форсируем альтернативу

    # --- ШАГ 4: Аугментация (Размножение данных) ---
    # Создаем вариации для тяжелых запросов (duration > 1s),
    # чтобы модель лучше их выучила (Oversampling)
    df_heavy = df_fixed[df_fixed['metric_duration_ms'] > 1000].copy()

    # Добавляем небольшой шум в фичи (jitter), сохраняя таргеты
    # Это учит модель быть устойчивой к небольшим изменениям коста
    df_heavy_variant = df_heavy.copy()
    df_heavy_variant['feature_total_cost'] *= np.random.uniform(0.9, 1.1, size=len(df_heavy))
    df_heavy_variant['feature_plan_rows'] *= np.random.uniform(0.95, 1.05, size=len(df_heavy))

    # --- ШАГ 5: Финальная сборка ---
    result_df = pd.concat([df_fixed, df_heavy_nl_fix, df_heavy_variant], ignore_index=True)

    # Очистка строк
    cat_flags = [
        'target_jit', 'target_enable_indexscan', 'target_enable_seqscan',
        'target_enable_bitmapscan', 'target_enable_hashjoin',
        'target_enable_mergejoin', 'target_enable_nestloop'
    ]
    for flag in cat_flags:
        result_df[flag] = result_df[flag].astype(str).str.upper().apply(
            lambda x: 'ON' if 'T' in x or 'ON' in x or '1' in x else 'OFF'
        )

    return result_df

def bucket_parallel_workers(w):
    if w == 0:
        return 0
    elif w <= 2:
        return 2
    elif w <= 4:
        return 4
    else:
        return 8
    # if w < 4:
    #     return 0
    # else:
    #     return 1


def balance_and_filter_labels(df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    """
    1. Удаляет 'шумные' редкие классы (быстрые запросы).
    2. Аугментирует (размножает) редкие, но важные классы (медленные запросы).
    3. Оставшиеся слишком редкие классы удаляет для стабильности обучения.
    """
    df_copy = df.copy()

    # Считаем частоту каждого лейбла
    label_counts = df_copy['strategy_label'].value_counts()

    # Разделяем лейблы на частые и редкие
    rare_labels = label_counts[label_counts < min_samples].index

    # Списки для сбора данных
    final_dfs = []

    # 1. Сразу берем все частые классы
    df_common = df_copy[~df_copy['strategy_label'].isin(rare_labels)]
    final_dfs.append(df_common)

    # 2. Обработка редких классов
    for label in rare_labels:
        df_rare = df_copy[df_copy['strategy_label'] == label]

        # Проверяем "важность" класса: если среднее время выполнения > 500мс
        # или есть Disk Spills (запись во временные блоки)
        is_important = (df_rare['metric_duration_ms'].mean() > 500) or \
                       (df_rare['metric_temp_written_blocks'].sum() > 0)

        if is_important:
            # АУГМЕНТАЦИЯ: Размножаем важный редкий класс
            # Нам нужно довести его количество хотя бы до min_samples
            multiplier = int(np.ceil(min_samples / len(df_rare)))

            for _ in range(multiplier):
                augmented_variant = df_rare.copy()

                # Добавляем "физический шум" (Jitter) в числовые признаки
                # Чтобы модель не просто зазубрила строку, а поняла диапазон
                numeric_cols = [
                    'feature_total_cost', 'feature_plan_rows',
                    'feature_total_scan_size_bytes', 'feature_max_node_cost',
                    'planner_error_ratio'
                ]

                for col in numeric_cols:
                    # Разброс +/- 7%
                    noise = np.random.uniform(0.93, 1.07, size=len(augmented_variant))
                    augmented_variant[col] *= noise

                final_dfs.append(augmented_variant)

            logging.info(f"Label '{label}' augmented (rare but important)")
        else:
            # Если класс редкий и запрос быстрый - просто игнорируем его
            # (он не попадет в final_dfs), тем самым удаляя шум.
            logging.info(f"Label '{label}' dropped (rare and noise)")

    return pd.concat(final_dfs, ignore_index=True)



# def _create_strategy_label(df: pd.DataFrame) -> pd.Series:
#     """Объединение таргетов в уникальную метку стратегии."""
#
#     def get_flag(val):
#         s = str(val).strip().upper()
#         return 'T' if 'TRUE' in s or 'ON' in s else 'F'
#
#     res = (
#         # "W" + df['target_work_mem_mb'].astype(int).astype(str) + "_" +
#             "P" + df['target_max_parallel_workers_per_gather'].astype(int).astype(str) + "_" +
#             "JIT" + df['target_jit'].apply(get_flag) + "_" +
#             "IDX" + df['target_enable_indexscan'].apply(get_flag) + "_" +
#             "SEQ" + df['target_enable_seqscan'].apply(get_flag) + "_" +
#             "HJ" + df['target_enable_hashjoin'].apply(get_flag) + "_" +
#             "NL" + df['target_enable_nestloop'].apply(get_flag)
#     )
#     return res

def _create_strategy_label(df: pd.DataFrame) -> pd.Series:
    """
    Преобразование разрозненных флагов в консолидированную стратегию.
    Это снижает количество классов и повышает точность.
    """

    def get_join_strat(row):
        # Определяем доминирующий тип джойна
        if row['target_enable_hashjoin'].lower() == 'off': return 'HASH'
        if row['target_enable_nestloop'].lower() == 'off': return 'NEST'
        return 'MERGE'

    def get_scan_strat(row):
        # Определяем доминирующий тип сканирования
        if row['target_enable_indexscan'].lower() == 'off': return 'IDX'
        if row['target_enable_bitmapscan'].lower() == 'off': return 'BM'
        return 'SEQ'

    def get_flag(val):
        s = str(val).strip().upper()
        return 'D' if s == 'OFF' else ''

    # Создаем компактный лейбл: Parallel_JIT_Join_Scan
    # Пример: P2_JIT_F_HASH_IDX
    res = (
            "P" + df['target_max_parallel_workers_per_gather'].astype(int).astype(str)
            + "_" +
            "JIT" + df['target_jit'].apply(get_flag)
            + "_" +
            df.apply(get_join_strat, axis=1)
            + "_" +
            df.apply(get_scan_strat, axis=1)
    )
    return res


def load_and_clean_data(engine, table_name: str) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    numeric_cols = [
        'feature_total_cost'
        , 'feature_plan_rows'
        , 'feature_plan_width'
        , 'feature_num_joins'
        , 'feature_num_scans'
        , 'feature_num_aggs'
        , 'feature_num_sorts'
        , 'feature_num_filters'
        , 'feature_num_index_scans'
        , 'feature_num_mem_nodes'
        , 'feature_total_scan_size_bytes'
        , 'feature_max_node_cost'
        , 'target_work_mem_mb'
        , 'target_max_parallel_workers_per_gather'
        , 'target_planner_error_ratio'
        , 'max_mem_limit_dop0'
        , 'metric_duration_ms'
        , 'metric_actual_rows'
        , 'metric_temp_written_blocks'
        , 'metric_shared_hit_blocks'
        , 'metric_shared_read_blocks'
        , 'metric_peak_memory_mb'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['target_work_mem_mb'] = 2 ** np.round(np.log2(df['target_work_mem_mb']))
    print(df['target_work_mem_mb'])

    cat_cols = [
        'target_jit',
        'target_enable_indexscan',
        'target_enable_seqscan',
        'target_enable_bitmapscan',
        'target_enable_hashjoin',
        'target_enable_mergejoin',
        'target_enable_nestloop'

    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    df = augment_and_fix_pg_dataset(df)

    logging.info(f"begin {len(df)}")
    # Фильтрация выбросов
    df['planner_error_ratio'] = df['target_planner_error_ratio']
    df = df.drop(columns=['target_planner_error_ratio'])
    df['target_max_parallel_workers_per_gather'] = df['target_max_parallel_workers_per_gather'].apply(
        bucket_parallel_workers)
    df['strategy_label'] = _create_strategy_label(df)
    # Перемешивание
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['planner_error_ratio'] = df['planner_error_ratio'].clip(upper=10)
    df = df[df['planner_error_ratio'] > 0]
    return df


if __name__ == '__main__':
    engine = create_engine(POSTGRES_URI)
    df = load_and_clean_data(engine, TABLE_NAME)
    # df = balance_and_filter_labels(df, 10)
    # df.to_csv("df.csv", sep=";", encoding='utf8')
    logging.info(f"begin {len(df)}")
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['strategy_label']
    )

    # logging.info("--- Этап 1: Обучение Регрессора (Memory) ---")
    # # Обучаем регрессор
    regressor = ErrPredictor()
    Xr_test, yr_test = regressor.train(df_train, df_test)
    res1 = regressor.evaluate(Xr_test, yr_test)

    # classifier = PGStrategyClassifier()
    # Xc_test, yc_test = classifier.train(df_train, df_test)
    # res2 = classifier.evaluate(Xc_test, yc_test, df_test)
