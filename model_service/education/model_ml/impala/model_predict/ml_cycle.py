import logging
import pandas as pd
from sqlalchemy import create_engine

from model_ml.impala.model_predict.mem_strategy_cascade import ImpalaOptimizationCascade

POSTGRES_URI = 'postgresql://suser:example@localhost:5432/model_tuning'
RANDOM_STATE = 42
TABLE_NAME = 'ml_training_dataset'


def _create_strategy_label(df: pd.DataFrame) -> pd.Series:
    """Создает метку стратегии"""
    return (
            "DOP" + df['target_mt_dop'].astype(int).astype(str) + "_" +
            "TH" + df['target_num_scanner_threads'].astype(int).astype(str) + "_" +
            "JM" + df['target_default_join_distribution_mode'].astype(str).str.upper() + "_" +
            "CG" + df['target_disable_codegen'].astype(str).str.upper()
    )


def load_and_clean_data(self, uri: str, table_name: str) -> pd.DataFrame:
    """Единый метод загрузки и первичной обработки"""
    engine = create_engine(uri)
    logging.info(f"Загрузка данных из {table_name}...")
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

    # Типизация
    numeric_cols = [
        'metric_pmu', 'target_mem_limit_dop0',
        'feature_plan_num_joins', 'feature_plan_num_scan_nodes',
        'feature_plan_num_agg_nodes', 'feature_plan_total_scan_size_bytes',
        'feature_plan_max_cardinality', 'feature_plan_max_row_size',
        'target_mt_dop', 'target_num_scanner_threads'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Очистка строк и создание таргетов
    cat_cols = ['target_default_join_distribution_mode', 'target_disable_codegen']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    df['strategy_label'] = self._create_strategy_label(df)
    df['base_pmu'] = df['target_mem_limit_dop0']  # Цель для регрессора
    df.drop(columns='target_mem_limit_dop0')

    # Фильтрация выбросов (ваши принципы)
    df = df[df['metric_pmu'] > 0]
    q_high = df['base_pmu'].quantile(0.99)
    df = df[df['base_pmu'] < q_high]

    # Перемешивание
    df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
    return df


if __name__ == '__main__':
    # 1. Инициализация каскада
    cascade_model = ImpalaOptimizationCascade(safe_factor=1.2)

    # 2. Загрузка и предобработка данных
    data = cascade_model.load_and_clean_data(POSTGRES_URI, TABLE_NAME)

    # 3. Обучение всего каскада (Модель 1 -> Модель 2)
    test_df = cascade_model.train(data)

    # 4. Оценка итогового качества
    metrics = cascade_model.evaluate_cascade(test_df)

    # 5. Сохранение модели
    cascade_model.save()
