import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from model_ml.pg.classifier_postgre import StrategyCascadeClassifier

# Импортируем ваш исправленный класс

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POSTGRES_URI = 'postgresql://suser:example@localhost:5432/model_tuning_pg'
RANDOM_STATE = 42
TABLE_NAME = 'ml_training_dataset_new'


def clean_boolean(val):
    """Приведение всех вариантов булевых значений PG к единому виду."""
    s = str(val).lower().strip()
    return '1' if s in ['on', 'true', '1', '1.0', 't'] else '0'


def create_robust_target(df: pd.DataFrame) -> pd.Series:
    """
    Создает целевую переменную.
    ВАЖНО: Если какой-то флаг (например JIT) почти всегда 'off',
    лучше исключить его из лейбла, чтобы не дробить классы зря.
    """
    # DOP: Бинаризация (Low vs High)
    dop_label = df['target_max_parallel_workers_per_gather'].apply(lambda x: 'PL' if x < 4 else 'PH')

    # Joins
    def get_join_label(row):
        h = clean_boolean(row['target_enable_hashjoin']) == '1'
        n = clean_boolean(row['target_enable_nestloop']) == '1'
        if not h: return "NoHash"
        if not n: return "NoNest"
        return "JoinsAll"

    join_label = df.apply(get_join_label, axis=1)

    # Индексы и JIT (объединяем в компактный флаг)
    idx_flag = df['target_enable_indexscan'].apply(clean_boolean)
    jit_flag = df['target_jit'].apply(clean_boolean)

    # Итоговый класс: например 'PH_JoinsAll_I1_J0'
    return dop_label + "_" + join_label + "_I" + idx_flag + "_J" + jit_flag


def load_and_prepare_data(engine, table_name: str):
    logger.info("Загрузка данных из БД...")
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

    # 1. Очистка типов
    cols_to_fix = [
        'feature_total_cost', 'feature_plan_rows', 'feature_plan_width',
        'feature_total_scan_size_bytes', 'target_work_mem_mb',
        'target_max_parallel_workers_per_gather', 'metric_max_log_planner_error'
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Создаем таргет
    df['temp_target'] = create_robust_target(df)
    df['log_planner_error'] = df['metric_max_log_planner_error']
    df.drop(columns='metric_max_log_planner_error')
    # 3. ФИЛЬТРАЦИЯ: Убираем классы, в которых слишком мало примеров
    # Для точности 90% модели нужно хотя бы 10-20 примеров на класс

    return df


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
        augmented_sample['temp_target'] = create_robust_target(augmented_sample)

    # 5. Соединяем с оригиналом
    new_df = pd.concat([df, augmented_sample], ignore_index=True)

    logger.info(f"Аугментация завершена. Было: {len(df)}, Стало: {len(new_df)}")
    return new_df


def print_feature_importance(model, model_name, feature_names):
    if model is None:
        print(f"Модель {model_name} не была обучена (константа).")
        return

    # Получаем веса важности
    importances = model.get_feature_importance()

    # Создаем DataFrame для удобства
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print(f"\nВажность признаков для модели [{model_name}]:")
    print(feature_importances.head(10))  # Топ-10 признаков


def run_training_pipeline():
    engine = create_engine(POSTGRES_URI)
    df_raw = load_and_prepare_data(engine, TABLE_NAME)
    df = augment_index_off_data(df_raw)
    min_samples_per_class = 10
    counts = df['temp_target'].value_counts()
    to_keep = counts[counts >= min_samples_per_class].index
    df = df[df['temp_target'].isin(to_keep)].copy()
    logger.info(f"После фильтрации осталось {len(df)} строк и {len(to_keep)} уникальных стратегий.")

    if len(df) == 0:
        logger.error("Нет данных для обучения после фильтрации!")
        return

    # Разделение на Train и Test со стратификацией
    # Это гарантирует, что в тесте будут те же пропорции классов, что и в обучении
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df['temp_target']
    )
    # df_train = augment_index_off_data(df_train)

    logger.info(f"Размер трейна: {len(df_train)}, теста: {len(df_test)}")

    # Инициализация и обучение
    classifier = StrategyCascadeClassifier(random_state=RANDOM_STATE)

    # Метод train возвращает X_test_feat, y_test для удобства
    X_tr = classifier.train(df_train, df_test)

    # Предсказание
    predictions_df = classifier.predict(df_test)

    print_feature_importance(classifier.model, 'classifier', X_tr.columns)

    # --- ОЦЕНКА ---
    y_true = df_test['temp_target']
    y_pred = predictions_df['strategy_label']

    acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"ОБЩАЯ ТОЧНОСТЬ (Accuracy): {acc:.2%}")
    print("=" * 60)

    # Детальный отчет по каждому классу (поможет понять, где модель ошибается)
    print("\nДетальный отчет по стратегиям:")
    print(classification_report(y_true, y_pred))

    # Оценка точности отдельных параметров (DOP, Joins и т.д.)
    for param in ['target_max_parallel_workers_per_gather', 'target_enable_hashjoin', 'target_enable_indexscan']:
        # Приводим к единому виду для сравнения
        p_true = df_test[param].apply(str).apply(clean_boolean) if 'enable' in param or 'jit' in param else df_test[
            param]
        p_pred = predictions_df[param].apply(str).apply(clean_boolean) if 'enable' in param or 'jit' in param else \
            predictions_df[param]

        # Для DOP делаем пороговое сравнение как в таргете
        if param == 'target_max_parallel_workers_per_gather':
            p_true = df_test[param].apply(lambda x: 'PH' if x >= 4 else 'PL')
            p_pred = predictions_df[param].apply(lambda x: 'PH' if x >= 4 else 'PL')

        p_acc = accuracy_score(p_true, p_pred)
        print(f"Accuracy для {param:40}: {p_acc:.2%}")

    # Оценка памяти
    gt_mem = df_test['target_work_mem_mb']
    pr_mem = predictions_df['pred_work_mem_mb']
    sufficiency = (pr_mem >= gt_mem).mean()
    print(f"Memory Sufficiency (Без OOM): {sufficiency:.2%}")
    print("=" * 60)


if __name__ == '__main__':
    run_training_pipeline()
