import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from model_ml.pg.s_classifier import StrategyCascadeClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POSTGRES_URI = 'postgresql://suser:example@localhost:5432/model_tuning_pg'
RANDOM_STATE = 42
TABLE_NAME = 'ml_training_dataset_new'


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
    # Это гарантирует, что редкие комбинации настроек распределятся равномерно
    df['strat_label'] = df.apply(get_strat_label, axis=1)

    df = augment_index_off_data(df, fraction=0.10)

    # 3. ФИЛЬТРАЦИЯ: Удаляем комбинации, где меньше 10 примеров
    # Каскаду нужно хотя бы 10 примеров, чтобы посчитать 85-й перцентиль памяти
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
    fraction: какую долю данных склонировать и изменить (10% обычно достаточно).
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
    # Увеличим ее в 1.5 - 2 раза (имитация Seq Scan вместо Index Scan)
    augmented_sample['feature_total_cost'] = augmented_sample['feature_total_cost'] * 1.8

    # 4. Пересчитываем технические лейблы, чтобы сплит и обучение их подхватили
    # ВАЖНО: вызываем те же функции, что и в основном коде
    if 'temp_target' in df.columns:
        augmented_sample['temp_target'] = augmented_sample.apply(get_strat_label, axis=1)

    # Если вы используете strat_label для train_test_split
    def get_strat_label_local(row):
        # Быстрая копия логики создания лейбла
        jit = 1 if str(row['target_jit']).lower() in ['on', 'true', '1'] else 0
        idx = 0  # Мы их выключили
        hj = 1 if str(row['target_enable_hashjoin']).lower() in ['on', 'true', '1'] else 0
        nl = 1 if str(row['target_enable_nestloop']).lower() in ['on', 'true', '1'] else 0
        dop = int(row['target_max_parallel_workers_per_gather'])
        return f"J{jit}_I{idx}_H{hj}_N{nl}_D{dop}"

    augmented_sample['strat_label'] = augmented_sample.apply(get_strat_label_local, axis=1)

    # 5. Соединяем с оригиналом
    new_df = pd.concat([df, augmented_sample], ignore_index=True)

    logger.info(f"Аугментация завершена. Было: {len(df)}, Стало: {len(new_df)}")
    return new_df


def run_cascade_pipeline():
    engine = create_engine(POSTGRES_URI)
    df = load_and_prepare_data(engine, TABLE_NAME)

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

    # df_train = augment_index_off_data(df_train, fraction=0.10)

    # regressor = ErrPredictor()
    # Xr_test, yr_test = regressor.train(df_train, df_test)
    # res1 = regressor.evaluate(Xr_test, yr_test)

    logger.info(f"Трейн: {len(df_train)}, Тест: {len(df_test)}")

    # Инициализация каскада
    classifier = StrategyCascadeClassifier(random_state=RANDOM_STATE)

    # ОБУЧЕНИЕ (внутри обучит JIT, Joins, Indexes и Parallelism)
    classifier.train(df_train, df_test)

    # ПРЕДСКАЗАНИЕ
    predictions_df = classifier.predict(df_test)

    # --- ОЦЕНКА РЕЗУЛЬТАТОВ ---
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ КАСКАДНОЙ МОДЕЛИ")
    print("=" * 60)

    # 1. JIT Accuracy
    y_true_jit = df_test['target_jit'].apply(clean_boolean)
    y_pred_jit = predictions_df['target_jit'].apply(clean_boolean)
    print(f"JIT Accuracy:             {accuracy_score(y_true_jit, y_pred_jit):.2%}")

    # 2. Join Strategy Accuracy
    def get_join_type(row):
        h = clean_boolean(row['target_enable_hashjoin'])
        n = clean_boolean(row['target_enable_nestloop'])
        if not h: return "NoHash"
        if not n: return "NoNest"
        return "JoinsAll"

    y_true_join = df_test.apply(get_join_type, axis=1)
    # В предикте мы уже возвращаем строки 'JoinsAll', 'NoHash', 'NoNest' или аналоги
    # ВАЖНО: убедитесь, что в predict возвращается колонка для сравнения.
    # Если в predict нет 'join_type', соберем его из флагов:
    y_pred_join = predictions_df.apply(get_join_type, axis=1)
    print(f"Join Strategy Accuracy:   {accuracy_score(y_true_join, y_pred_join):.2%}")

    # 3. DOP Accuracy
    y_true_dop = df_test['target_max_parallel_workers_per_gather'].astype(int)
    y_pred_dop = predictions_df['target_max_parallel_workers_per_gather'].astype(int)
    print(f"DOP Accuracy:             {accuracy_score(y_true_dop, y_pred_dop):.2%}")

    # 4. Memory Sufficiency
    sufficiency = (predictions_df['pred_work_mem_mb'] >= df_test['target_work_mem_mb']).mean()
    print(f"Memory Sufficiency (P85): {sufficiency:.2%}")

    # 5. Full Session Match (Насколько часто угадана вся комбинация флагов целиком)
    # Это самая жесткая метрика
    match_mask = (
            (y_true_jit == y_pred_jit) &
            (y_true_join == y_pred_join) &
            (y_true_dop == y_pred_dop) &
            (df_test['target_enable_indexscan'].apply(clean_boolean) ==
             predictions_df['target_enable_indexscan'].apply(clean_boolean))
    )
    print(f"Full Strategy Match:      {match_mask.mean():.2%}")
    print("=" * 60)

    # Детальный отчет по DOP (самый сложный параметр)
    print("\nОтчет по предсказанию DOP:")
    print(classification_report(y_true_dop, y_pred_dop))
    print(classification_report(y_true_join, y_pred_join))
    print(classification_report(y_true_jit, y_pred_jit))


if __name__ == '__main__':
    run_cascade_pipeline()
