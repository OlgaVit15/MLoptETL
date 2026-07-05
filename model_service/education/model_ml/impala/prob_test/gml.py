import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

POSTGRES_URI = 'postgresql://suser:example@localhost:5432/model_tuning'
TABLE_NAME = 'ml_training_dataset'
TARGET_COLUMN = 'target_mem_limit_dop0'
RANDOM_STATE = 42

# --- 1. ЗАГРУЗКА ---
def load_dataset(engine):
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
    cols = ['feature_plan_num_joins', 'feature_plan_num_scan_nodes',
            'feature_plan_total_scan_size_bytes', 'feature_plan_max_cardinality',
            'feature_plan_max_row_size', TARGET_COLUMN]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Очистка от экстремальных выбросов (обязательно для GPR)
    df = df[df[TARGET_COLUMN] > 0]
    q_high = df[TARGET_COLUMN].quantile(0.98)
    df = df[df[TARGET_COLUMN] < q_high]
    return df


# --- 2. ГЕНЕРАЦИЯ ВЗАИМОДЕЙСТВИЙ (Manual Polynomial) ---
def prepare_advanced_features(df):
    X = pd.DataFrame()
    # Базовые
    X['scans'] = df['feature_plan_num_scan_nodes']
    X['joins'] = df['feature_plan_num_joins']
    X['row_size'] = df['feature_plan_max_row_size']

    # Взаимодействия (физика процесса)
    # Главный фактор: сколько данных мы реально "ворочаем"
    X['vol_theory'] = df['feature_plan_max_cardinality'] * df['feature_plan_max_row_size']
    X['scan_density'] = df['feature_plan_total_scan_size_bytes'] / (df['feature_plan_num_scan_nodes'] + 1)

    # Нелинейные комбинации
    X['join_impact'] = X['joins'] * np.log1p(df['feature_plan_max_cardinality'])

    # Логарифмы для сглаживания
    X['log_scan_size'] = np.log1p(df['feature_plan_total_scan_size_bytes'])
    X['log_card'] = np.log1p(df['feature_plan_max_cardinality'])

    return X.fillna(0)


# --- 3. СТРОИМ МОДЕЛЬ ---
def get_expert_model(n_samples):
    # Ядро для Гауссовского процесса:
    # Matern (гибкое) + WhiteKernel (компенсация шума плана)
    kernel = C(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
ф
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        alpha=0.2,  # Регуляризация шума
        random_state=42
    )

    # SVR - мастер "попадания в точку"
    svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')

    # Ансамбль из двух мощных нелинейных моделей
    voting = VotingRegressor([
        ('gpr', gpr),
        ('svr', svr)
    ])

    # Пайплайн обработки
    pipe = Pipeline([
        ('scaler', RobustScaler()),  # Устойчив к выбросам
        ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=min(n_samples, 1000))),
        ('regressor', voting)
    ])

    # Таргет в логарифм для точности на малых значениях
    return TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1
    )


if __name__ == '__main__':
    engine = create_engine(POSTGRES_URI)
    df = load_dataset(engine)

    X = prepare_advanced_features(df)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_expert_model(len(X_train))

    print("Обучение нелинейного ансамбля (это может занять 1-2 минуты)...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 1.0)  # Минимум 1 МБ

    # ОЦЕНКА
    errors = np.abs(y_test - y_pred)
    print(f"\n--- ИТОГИ ЮВЕЛИРНОЙ МОДЕЛИ ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} MB")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Точность +/- 1 MB: {np.mean(errors <= 1.0):.2%}")
    print(f"Точность +/- 5 MB: {np.mean(errors <= 5.0):.2%}")
    print(f"OOM Rate: {np.mean(y_pred < y_test):.2%}")
