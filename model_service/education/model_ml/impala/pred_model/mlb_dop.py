import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report

# --- КОНФИГУРАЦИЯ ---
POSTGRES_URI = 'postgresql://suser:example@localhost:5432/model_tuning'
TARGET_COLUMN = 'target_mt_dop'
RANDOM_STATE = 42


def _create_strategy_label(self, df):
    """Создает уникальный идентификатор стратегии из всех настроек"""
    return (
            "DOP" + df['target_mt_dop'].astype(int).astype(str) + "_" +
            "TH" + df['target_num_scanner_threads'].astype(int).astype(str) + "_" +
            "JM" + df['target_default_join_distribution_mode'].astype(str).str.upper() + "_" +
            "CG" + df['target_disable_codegen'].astype(str).str.upper()
    )


def load_dataset(engine):
    query = "SELECT * FROM ml_training_dataset"
    df = pd.read_sql(query, engine)

    numeric_cols = [
        'target_mem_limit_dop0', 'feature_plan_num_joins',
        'feature_plan_num_scan_nodes', 'feature_plan_num_agg_nodes',
        'feature_plan_total_scan_size_bytes', 'feature_plan_max_cardinality',
        'feature_plan_max_row_size', 'target_mt_dop'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Классификатор требует целых чисел в таргете
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Фильтрация мусора
    df = df[df['feature_plan_total_scan_size_bytes'] >= 0]
    return df


def prepare_advanced_features(df):
    X = pd.DataFrame()

    # 1. КАСКАДНЫЙ ПРИЗНАК (Связь между памятью DOP0 и итоговым DOP)
    # Это самый сильный признак для классификатора
    X['base_pmu'] = df['target_mem_limit_dop0']
    X['log_base_pmu'] = np.log1p(df['target_mem_limit_dop0'])

    # 2. ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
    X['scans'] = df['feature_plan_num_scan_nodes']
    X['joins'] = df['feature_plan_num_joins']
    X['aggs'] = df['feature_plan_num_agg_nodes']

    # Объем данных
    X['vol_theory'] = df['feature_plan_max_cardinality'] * df['feature_plan_max_row_size']
    X['log_scan_size'] = np.log1p(df['feature_plan_total_scan_size_bytes'])

    # 3. УМНЫЕ ВЗАИМОДЕЙСТВИЯ (Инсайты для DOP)
    # Плотность памяти на один узел сканирования
    nodes = df['feature_plan_num_scan_nodes'].replace(0, 1)
    X['pmu_per_node'] = df['target_mem_limit_dop0'] / nodes

    # Сложность джойнов относительно объема памяти
    X['join_complexity'] = df['feature_plan_num_joins'] * X['log_base_pmu']

    # Ширина строки (влияет на то, сколько потоков выгодно запускать)
    X['row_size'] = df['feature_plan_max_row_size']

    return X.fillna(0)


def get_classifier_model(n_samples):
    # ПАРАМЕТРЫ ДЛЯ КЛАССИФИКАЦИИ ВЫСОКОЙ ТОЧНОСТИ
    cb_params = {
        'iterations': 4000,
        'learning_rate': 0.02,  # Очень маленький шаг для "вылизывания" точности
        'depth': 7,  # Глубокие деревья для сложных зависимостей каскада
        'l2_leaf_reg': 3,  # Сильная регуляризация против переобучения
        'boosting_type': 'Ordered',  # КЛЮЧ к точности на малых данных
        'loss_function': 'MultiClass',  # Классификация нескольких уровней DOP
        'auto_class_weights': 'Balanced',  # Если каких-то DOP мало (например DOP 8)
        'random_seed': RANDOM_STATE,
        'bootstrap_type': 'MVS',
        'verbose': 500,
        'early_stopping_rounds': 200
    }

    # Пайплайн обработки
    feature_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('quantile', QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(n_samples, 1000),
            random_state=RANDOM_STATE
        ))
    ])

    model = CatBoostClassifier(**cb_params)

    # Собираем итоговый пайплайн
    return Pipeline([
        ('preprocessor', feature_pipe),
        ('clf', model)
    ])


if __name__ == '__main__':
    engine = create_engine(POSTGRES_URI)
    df = load_dataset(engine)

    X = prepare_advanced_features(df)
    y = df[TARGET_COLUMN]

    # Важно: stratify=y гарантирует, что все уровни DOP попадут и в train, и в test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = get_classifier_model(len(X_train))

    print(f"Запуск обучения CatBoostClassifier (Classes: {y.unique()})...")
    # Обучаем классификатор
    model.fit(X_train, y_train)

    # Предсказание классов (меток DOP)
    y_pred = model.predict(X_test)
    # Превращаем в плоский массив для оценки
    # y_pred = y_pred.flatten()

    # --- ОЦЕНКА ---
    # Хотя это классификатор, мы считаем R2, так как DOP - это числовой порядок
    r2 = r2_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n--- РЕЗУЛЬТАТЫ КЛАССИФИКАТОРА ---")
    print(f"Accuracy (Точное совпадение DOP): {acc:.2%}")
    print(f"R2 Score (как числовой метрики): {r2:.4f}")

    if r2 < 0.90:
        print("\nСОВЕТ: Для достижения 0.90+ проверь важность признака 'base_pmu'.")
        print("Если он в ТОПе, попробуй добавить 'base_pmu' во второй степени (квадрат).")
