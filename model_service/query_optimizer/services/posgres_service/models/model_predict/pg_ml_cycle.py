import logging
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from ml_models.pg_cascade_strategy import PGOptimizationCascade

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POSTGRES_URI = 'postgresql://suser:example@localhost:5432/model_tuning_pg'
RANDOM_STATE = 42
TABLE_NAME = 'ml_training_dataset_new'
mem_bins = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def get_strat_label(row):
    jit = clean_boolean(row['target_jit'])
    idx = clean_boolean(row['target_enable_indexscan'])
    hj = clean_boolean(row['target_enable_hashjoin'])
    nl = clean_boolean(row['target_enable_nestloop'])
    dop = int(row['target_max_parallel_workers_per_gather'])
    return f"J{jit}_I{idx}_H{hj}_N{nl}_D{dop}"


def clean_boolean(val):
    """Приведение всех вариантов булевых значений PG к единому виду (0 или 1)."""
    s = str(val).lower().strip()
    return 1 if s in ['on', 'true', '1', '1.0', 't'] else 0


def round_to_list(x, target_list):
    # Находим индекс значения с минимальной абсолютной разницей
    return target_list[np.abs(target_list - x).argmin()]


def load_and_prepare_data(engine, table_name: str):
    logger.info("Загрузка данных из БД...")
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

    # 1. Очистка числовых типов
    numeric_cols = [
        'feature_total_cost', 'feature_plan_rows', 'feature_plan_width',
        'feature_total_scan_size_bytes', 'target_work_mem_mb',
        'target_max_parallel_workers_per_gather', 'metric_max_log_planner_error',
        'feature_num_joins', 'feature_num_scans', 'feature_num_index_scans',
        'feature_num_sorts', 'feature_num_aggs'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Создаем временный технический лейбл только для СТРАТИФИКАЦИИ (сплита)
    df['strat_label'] = df.apply(get_strat_label, axis=1)

    df = augment_index_off_data(df, fraction=0.10)
    df['target_work_mem_mb'] = df['target_work_mem_mb'].apply(lambda x: round_to_list(x, mem_bins))

    # 3. ФИЛЬТРАЦИЯ: Удаляем комбинации, где меньше 10 примеров
    min_samples = 10
    counts = df['strat_label'].value_counts()
    to_keep = counts[counts >= min_samples].index
    df_filtered = df[df['strat_label'].isin(to_keep)].copy()

    logger.info(f"Загружено: {len(df)}. После фильтрации: {len(df_filtered)} строк. "
                f"Уникальных комбинаций: {len(to_keep)}")

    return df_filtered


def augment_index_off_data(df: pd.DataFrame, fraction: float = 0.10) -> pd.DataFrame:
    """
    Создает синтетические примеры, где индексы выключены.
    """
    logger.info(f"Генерация синтетических данных: выключаем индексы для {fraction * 100}% записей...")

    # 1. Берем случайную выборку из существующих данных
    augmented_sample = df.sample(frac=fraction, random_state=42).copy()

    # 2. Меняем таргет
    augmented_sample['target_enable_indexscan'] = 'off'

    # 3. Корректируем фичи, чтобы они соответствовали "миру без индексов"
    # Обнуляем количество индексных сканов
    if 'feature_num_index_scans' in augmented_sample.columns:
        augmented_sample['feature_num_index_scans'] = 0

    # Логично, что без индексов стоимость (cost) должна вырасти
    k = random.uniform(1.1, 1.8)
    augmented_sample['feature_total_cost'] = augmented_sample['feature_total_cost'] * k

    # 4. Пересчитываем технические лейблы, чтобы сплит и обучение их подхватили
    if 'temp_target' in df.columns:
        augmented_sample['temp_target'] = augmented_sample.apply(get_strat_label, axis=1)

    augmented_sample['strat_label'] = augmented_sample.apply(get_strat_label, axis=1)

    # 5. Соединяем с оригиналом
    new_df = pd.concat([df, augmented_sample], ignore_index=True)

    logger.info(f"Аугментация завершена. Было: {len(df)}, Стало: {len(new_df)}")
    return new_df


def run_cascade_pipeline():
    # engine = create_engine(POSTGRES_URI)
    # df = load_and_prepare_data(engine, TABLE_NAME)
    # df.to_csv("pg_ml_train_dataset2.csv", sep=";", encoding="utf8")
    df = pd.read_csv("D:/IdeaProjects/Ver1/model_service/query_optimizer/services/posgres_service/models/model_predict/dataset/pg_ml_train_dataset2.csv",
                     sep=';')

    if len(df) == 0:
        logger.error("Нет данных после фильтрации!")
        return

    # Разделение со стратификацией по тех. лейблу
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df['strat_label']
    )

    df_train = df_train.drop(columns='strat_label')
    df_test = df_test.drop(columns='strat_label')

    logger.info(f"Трейн: {len(df_train)}, Тест: {len(df_test)}")

    model = PGOptimizationCascade()

    model.train(df_train, df_test)
    # ПРЕДСКАЗАНИЕ
    predictions_df = model.predict(df_test)

    # --- ОЦЕНКА РЕЗУЛЬТАТОВ ---
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ КАСКАДНОЙ МОДЕЛИ (PostgreSQL)")
    print("=" * 60)

    # 1. JIT Accuracy
    y_true_jit = df_test['target_jit'].apply(clean_boolean)
    y_pred_jit = predictions_df['target_jit'].apply(clean_boolean)
    print(f"JIT Accuracy:               {accuracy_score(y_true_jit, y_pred_jit):.2%}")
    # 2. Index Accuracy
    y_true_idx = df_test['target_enable_indexscan'].apply(clean_boolean)
    y_pred_idx = predictions_df['target_enable_indexscan'].apply(clean_boolean)
    print(f"IndexScan Accuracy:         {accuracy_score(y_true_idx, y_pred_idx):.2%}")

    # 2. Join Strategy Accuracy
    y_true_join = df_test['target_enable_hashjoin'].apply(clean_boolean)
    y_pred_join = predictions_df['target_enable_hashjoin'].apply(clean_boolean)
    print(f"Join Strategy Accuracy:     {accuracy_score(y_true_join, y_pred_join):.2%}")

    # 3. DOP Accuracy
    y_true_dop = df_test['target_max_parallel_workers_per_gather'].astype(int)
    y_pred_dop = predictions_df['target_max_parallel_workers_per_gather'].astype(int)
    print(f"DOP Accuracy:               {accuracy_score(y_true_dop, y_pred_dop):.2%}")

    # 4. Memory Sufficiency
    sufficiency = (predictions_df['pred_work_mem_mb'] >= df_test['target_work_mem_mb']).mean()
    print(f"Memory Sufficiency (P85):   {sufficiency:.2%}")


    # Детальный отчеты
    print("\nОтчет по предсказанию DOP:")
    print(classification_report(y_true_dop, y_pred_dop))
    print("\nОтчет по предсказанию INDEX:")
    print(classification_report(y_true_idx, y_pred_idx))
    print("\nОтчет по предсказанию JOIN:")
    print(classification_report(y_true_join, y_pred_join))
    print("\nОтчет по предсказанию JIT:")
    print(classification_report(y_true_jit, y_pred_jit))

    # model.save()


if __name__ == '__main__':
    run_cascade_pipeline()
