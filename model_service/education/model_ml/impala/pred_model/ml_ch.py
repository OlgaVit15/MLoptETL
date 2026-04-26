import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

POSTGRES_URI = 'postgresql://suser:example@localhost:5432/model_tuning'
TABLE_NAME = 'ml_training_dataset'
TARGET_COLUMN = 'target_mem_limit_dop0'
RANDOM_STATE = 42


# --- 1. ЗАГРУЗКА И ФИЗИЧЕСКИЕ ФИЧИ ---
def prepare_advanced_features(df):
    X = pd.DataFrame()
    # Базовые
    X['scans'] = df['feature_plan_num_scan_nodes']
    X['joins'] = df['feature_plan_num_joins']
    X['aggs'] = df['feature_plan_num_agg_nodes']
    X['row_size'] = df['feature_plan_max_row_size']

    # Физика объема (главный драйвер)
    X['vol_theory'] = df['feature_plan_max_cardinality'] * df['feature_plan_max_row_size']
    X['log_scan_size'] = np.log1p(df['feature_plan_total_scan_size_bytes'])
    X['log_card'] = np.log1p(df['feature_plan_max_cardinality'])

    # Сложность структуры
    X['complexity'] = (X['joins'] + 1) * (X['aggs'] + 1)
    X['nodes_efficiency'] = X['log_scan_size'] / (X['scans'] + 1)

    # Добавляем "сырые" байты и кардинальность (CatBoost сам найдет пороги)
    X['raw_scan'] = df['feature_plan_total_scan_size_bytes']
    X['raw_card'] = df['feature_plan_max_cardinality']

    return X.fillna(0)


# --- 2. ОПТИМИЗИРОВАННАЯ МОДЕЛЬ ---
def get_super_cat_model(n_samples):
    # Настройки для "ювелирной" точности
    cb_params = {
        'iterations': 4000,  # Больше итераций для тонкой подстройки
        'learning_rate': 0.02,  # Малый шаг для точности
        'depth': 7,  # Оптимальная глубина для нелинейности
        'l2_leaf_reg': 3,  # Небольшая регуляризация
        'loss_function': 'RMSE',  # На логарифме таргета RMSE работает идеально
        'eval_metric': 'MAE',
        'bootstrap_type': 'MVS',  # Стабильный бутстрап
        'random_seed': 42,
        'verbose': 500,
        'early_stopping_rounds': 200
    }

    # Пайплайн обработки признаков (как в GPR)
    # Это заставит CatBoost видеть структуру данных четче
    feature_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('quantile', QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(n_samples, 1000),
            random_state=42
        ))
    ])

    model = CatBoostRegressor(**cb_params)

    # Обертка через пайплайн признаков
    full_pipe = Pipeline([
        ('preprocessor', feature_pipe),
        ('regressor', model)
    ])

    # Трансформация ТАРГЕТА (самое важное для 1 МБ точности)
    return TransformedTargetRegressor(
        regressor=full_pipe,
        func=np.log1p,
        inverse_func=np.expm1
    )


def _create_strategy_label(df):
    """Создает единую текстовую метку для всей комбинации настроек"""
    return (
            "DOP" + df['target_mt_dop'].astype(int).astype(str) + "_" +
            "TH" + df['target_num_scanner_threads'].astype(int).astype(str) + "_" +
            "JM" + df['target_default_join_distribution_mode'].astype(str).str.upper() + "_" +
            "CG" + df['target_disable_codegen'].astype(str).str.upper()
    )


if __name__ == '__main__':
    # ... загрузка данных через твой engine ...
    engine = create_engine(POSTGRES_URI)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

    # Очистка (не меняем)
    df = df[df[TARGET_COLUMN] > 0]
    q_high = df[TARGET_COLUMN].quantile(0.99)
    df = df[df[TARGET_COLUMN] < q_high]

    X = prepare_advanced_features(df)
    X['strat'] = _create_strategy_label(df)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['strat'])

    X_train = X_train.drop(columns='strat')
    X_test = X_test.drop(columns='strat')
    model = get_super_cat_model(len(X_train))

    print("Запуск обучения Super-Cat...")
    # Так как внутри пайплайна CatBoost, нам нужно передать eval_set через спец. синтаксис
    # или просто обучить (на 10к данных это быстро)
    model.fit(X_train, y_train)

    # Предсказание
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 1.0)

    # --- ОЦЕНКА ---
    errors = np.abs(y_test - y_pred)
    e = y_pred - y_test
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- РЕЗУЛЬТАТЫ ОДНОЙ МОДЕЛИ ---")
    print(f"MAE: {mae:.2f} MB")
    print(f"R2 Score: {r2:.4f}")
    print(f"Точность +/- 1 MB: {np.mean(errors <= 1.0):.2%}")
    print(f"Точность +/- 5 MB: {np.mean(errors <= 5.0):.2%}")
    print(f"OOM Rate: {np.mean(y_pred < y_test):.2%}")
    print(f"Max OOM Error (худший недолет): {np.min(e):.2f}")
    y_pred_safe = y_pred * 1.2  # Просто добавляем 5 МБ сверху или * 1.1
    print(f"nOOM Rate с поправкой +5MB: {np.mean(y_pred_safe < y_test):.2%}")
    nr2 = r2_score(y_test, y_pred_safe)
    print(f"R2 Score: {nr2:.4f}")
    e = y_pred_safe - y_test
    print(f"Max OOM Error (худший недолет): {np.min(e):.2f}")
