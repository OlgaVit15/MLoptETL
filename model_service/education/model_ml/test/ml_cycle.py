import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from tqdm import tqdm

from RandomForest.ensemble2 import ImpalaCascadeEnsemble
from CatBoost.mem_strategy_cascade import ImpalaOptimizationCascade
from model_ml.test.LGBM.lgbm_mem_strategy_cascade import LGBMImpalaOptimizationCascade
from model_ml.test.XGboost.xgb_mem_strategy_cascade import XGBImpalaOptimizationCascade

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


def load_and_clean_data(uri=None, table_name=None) -> pd.DataFrame:
    """Единый метод загрузки и первичной обработки"""
    # engine = create_engine(uri)
    logging.info(f"Загрузка данных из {table_name}...")
    # df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    path = "D:/ml_training_dataset.csv"
    df = pd.read_csv(path, sep=";")

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

    df['strategy_label'] = _create_strategy_label(df)
    df['base_pmu'] = df['target_mem_limit_dop0']  # Цель для регрессора
    df.drop(columns='target_mem_limit_dop0')

    # Фильтрация выбросов (ваши принципы)
    df = df[df['metric_pmu'] > 0]
    q_high = df['base_pmu'].quantile(0.99)
    df = df[df['base_pmu'] < q_high]

    # Перемешивание
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == '__main__':
    # 1. Инициализация каскада
    cascade_model = ImpalaOptimizationCascade(safe_factor=1.2)
    rf_model = ImpalaCascadeEnsemble()
    xgb_model = XGBImpalaOptimizationCascade()
    lgb_model = LGBMImpalaOptimizationCascade()

    # 2. Загрузка и предобработка данных
    df = load_and_clean_data()
    logging.info("--- Разделение и стратификация ---")
    # Разделение для обучения каскада
    strat = df['strategy_label']
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=strat
    )

    # 3. Обучение всего каскада (Модель 1 -> Модель 2)
    cascade_model.train(df_train)
    rf_model.train(df_train)
    xgb_model.train(df_train)
    lgb_model.train(df_train)

    results_cb = cascade_model.predict(df_test)
    results_cb.to_csv('results_cb.csv', sep=';', index=False)
    results_xgb = xgb_model.predict(df_test)
    results_xgb.to_csv('results_xgb.csv', sep=';', index=False)
    results_lgb = lgb_model.predict(df_test)
    results_lgb.to_csv('results_lgb.csv', sep=';', index=False)
    results = []
    for _, row in df_test.iterrows():
        plan_dict = row.to_dict()
        prediction = rf_model.predict(plan_dict)
        results.append({
            'mt_dop': int(prediction['target_mt_dop']),
            'pred_mem_limit': prediction['target_mem_limit'],
            'threads': int(prediction['target_num_scanner_threads']),
            'join_mode': prediction['target_default_join_distribution_mode'].upper(),
            'codegen': prediction['target_disable_codegen'],
            'base_pmu': prediction['target_mem_limit_dop0']
        })

    results_rf = pd.DataFrame(results,
                              columns=['mt_dop', 'pred_mem_limit', 'threads', 'join_mode', 'codegen', 'base_pmu'])
    results_rf.to_csv('results_rf.csv', sep=';', index=False)
    results_t = pd.DataFrame()
    results_t['mt_dop'] = df_test['target_mt_dop']
    results_t['mem_limit'] = df_test['target_mem_limit']
    results_t['threads'] = df_test['target_num_scanner_threads']
    results_t['join_mode'] = df_test['target_default_join_distribution_mode']
    results_t['codegen'] = df_test['target_disable_codegen']
    results_t['base_pmu'] = df_test['target_mem_limit_dop0']
    results_t.to_csv('results_t.csv', sep=';', index=False)

    cascade_model.save()
    xgb_model.save()
    lgb_model.save()
