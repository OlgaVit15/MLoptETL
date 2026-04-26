import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_absolute_error, r2_score

POSTGRES_URI = 'postgresql://suser:example@localhost:5432/model_tuning'
TABLE_NAME = 'ml_training_dataset'
TARGET_COLUMN = 'target_mem_limit_dop0'
RANDOM_STATE = 42


class ImpalaCascadeEnsemble:
    def __init__(self, max_groups=60, n_estimators=50, min_samples_leaf_ratio=0.005):
        self.max_groups = max_groups
        self.n_estimators = n_estimators
        self.min_samples_leaf_ratio = min_samples_leaf_ratio

        # Модели-ансамбли для точности
        self.model_mem = None  # RandomForest для памяти при 0 степени параллелизма
        self.model_strategy = None  # RandomForest для выбора степени параллелизма
        # Модель-группировщик для стабильного ID
        self.grouper = None

        self.leaf_profiles = {}
        self.feature_cols = [
            'log_num_joins', 'log_num_broadcast_joins', 'log_num_scan_nodes',
            'log_num_agg_nodes', 'log_num_files', 'log_total_scan_size',
            'log_max_cardinality', 'log_max_row_size', 'log_bytes_per_node'
        ]

    # --- 1. ЗАГРУЗКА И ФИЗИЧЕСКИЕ ФИЧИ ---
    def prepare_advanced_features_mem(self, df):
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

    def prepare_advanced_features_dop(self, df):
        X = pd.DataFrame()

        # 1. КАСКАДНЫЙ ПРИЗНАК (Связь между памятью DOP0 и итоговым DOP)
        # Это самый сильный признак для классификатора
        X['base_pmu'] = df['base_pmu']
        X['log_base_pmu'] = np.log1p(df['base_pmu'])

        # 2. ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
        X['scans'] = df['scans']
        X['joins'] = df['joins']
        X['aggs'] = df['aggs']

        # Объем данных
        X['vol_theory'] = df['vol_theory']
        X['log_scan_size'] = df['log_scan_size']

        # 3. УМНЫЕ ВЗАИМОДЕЙСТВИЯ (Инсайты для DOP)
        # Плотность памяти на один узел сканирования
        nodes = df['scans'].replace(0, 1)
        X['pmu_per_node'] = df['base_pmu'] / nodes

        # Сложность джойнов относительно объема памяти
        X['join_complexity'] = df['joins'] * X['log_base_pmu']

        # Ширина строки (влияет на то, сколько потоков выгодно запускать)
        X['row_size'] = df['row_size']

        return X.fillna(0)

   # подготовка моделей
    def get_super_catb_model(n_samples):
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

    def get_classifier_model_dop(self, n_samples):
        # ПАРАМЕТРЫ ДЛЯ КЛАССИФИКАЦИИ ВЫСОКОЙ ТОЧНОСТИ
        cb_params = {
            'iterations': 5000,
            'learning_rate': 0.01,  # Очень маленький шаг для "вылизывания" точности
            'depth': 8,  # Глубокие деревья для сложных зависимостей каскада
            'l2_leaf_reg': 5,  # Сильная регуляризация против переобучения
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

    # обучение моделей
    def train_cb_mem(self, X_train, y_train, X_test, y_test):
        model = self.get_super_catb_model(len(X_train))

        print("Запуск обучения Super-Cat...")
        # Так как внутри пайплайна CatBoost, нам нужно передать eval_set через спец. синтаксис
        model.fit(X_train, y_train)

        # Предсказание
        y_pred = np.maximum(model.predict(X_test), 1.0)

        # оценка
        self.model_score(y_test, y_pred)

        return model



    def predict_cb(self, feature_df):
        return np.maximum(self.model_mem.predict(feature_df), 1.0)

    def model_score(self, y_test, y_pred):
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
        # y_pred_safe = y_pred * 1.2  # Просто добавляем 5 МБ сверху или * 1.1
        # print(f"nOOM Rate с поправкой +5MB: {np.mean(y_pred_safe < y_test):.2%}")
        # nr2 = r2_score(y_test, y_pred_safe)
        # print(f"R2 Score: {nr2:.4f}")
        # e = y_pred_safe - y_test
        # print(f"Max OOM Error (худший недолет): {np.min(e):.2f}")


if __name__ == '__main__':
    # ... загрузка данных через твой engine ...
    engine = create_engine(POSTGRES_URI)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

    # Очистка (не меняем)
    df = df[df[TARGET_COLUMN] > 0]
    q_high = df[TARGET_COLUMN].quantile(0.99)
    df = df[df[TARGET_COLUMN] < q_high]

    X = prepare_advanced_features_mem(df)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
