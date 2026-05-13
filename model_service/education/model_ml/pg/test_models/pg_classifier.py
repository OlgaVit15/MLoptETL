import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from catboost import CatBoostClassifier

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PGStrategyModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.strategy_params_map = {}
        self.memory_map = {}
        self.mem_bins = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.feature_cols = []

    def _create_target_label(self, df: pd.DataFrame) -> pd.Series:
        """Преобразует комбинацию настроек PG в единый классифицируемый признак."""

        # 1. DOP (Parallelism)
        def get_dop_label(x):
            if x <= 0: return "P0"
            if x <= 2: return "P2"
            return "P4H"

        # 2. Joins (что именно включено/выключено)
        def get_join_label(row):
            h = str(row['target_enable_hashjoin']).lower() in ['on', 'true', '1']
            n = str(row['target_enable_nestloop']).lower() in ['on', 'true', '1']
            if not h: return "NoHash"
            if not n: return "NoNest"
            return "JoinsAll"

        # 3. Флаги JIT и IndexScan
        def get_flag(x):
            return "1" if str(x).lower() in ['on', 'true', '1'] else "0"

        return (
                df['target_max_parallel_workers_per_gather'].apply(get_dop_label) + "_" +
                df.apply(get_join_label, axis=1) + "_" +
                "J" + df['target_jit'].apply(get_flag) + "_" +
                "I" + df['target_enable_indexscan'].apply(get_flag)
        )

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерация признаков с учетом специфики Postgres."""
        X = pd.DataFrame(index=df.index)

        # Логарифмирование (убираем огромный разброс стоимостей)
        X['log_total_cost'] = np.log1p(df['feature_total_cost'].fillna(0))
        X['log_plan_rows'] = np.log1p(df['feature_plan_rows'].fillna(0))
        X['log_scan_bytes'] = np.log1p(df['feature_total_scan_size_bytes'].fillna(0))
        X['log_max_node_cost'] = np.log1p(df['feature_max_node_cost'].fillna(0))

        # Коррекция ошибки планировщика (самая важная фича для точности)
        # metric_max_log_planner_error - это логарифм ошибки. Прибавляем его к логу строк.
        X['corrected_log_rows'] = X['log_plan_rows'] + df['metric_max_log_planner_error'].fillna(0)

        # Плотность данных и сложность
        X['plan_width'] = df['feature_plan_width'].fillna(0)
        X['rows_x_width'] = X['log_plan_rows'] * X['plan_width']
        X['num_joins'] = df['feature_num_joins'].fillna(0)
        X['num_scans'] = df['feature_num_scans'].fillna(0)
        X['scans_to_joins'] = X['num_scans'] / (X['num_joins'] + 1)

        # Интенсивность операций в памяти
        X['mem_ops'] = (df['feature_num_sorts'] + df['feature_num_aggs'] + df['feature_num_joins']).fillna(0)
        X['index_scan_ratio'] = df['feature_num_index_scans'] / (df['feature_num_scans'] + 1)

        return X.replace([np.inf, -np.inf], 0).fillna(0)

    def train(self, df_train: pd.DataFrame):
        # Подготовка таргета
        df = df_train.copy()
        df['strategy_label'] = self._create_target_label(df)

        # Фильтруем слишком редкие стратегии (менее 3 вхождений), чтобы не ломать кросс-валидацию
        counts = df['strategy_label'].value_counts()
        df = df[df['strategy_label'].isin(counts[counts >= 3].index)]

        X = self.extract_features(df)
        y = df['strategy_label']
        self.feature_cols = X.columns.tolist()

        # Маппинг для восстановления настроек из лейбла и расчет 90-го перцентиля памяти
        for label in y.unique():
            subset = df[df['strategy_label'] == label]
            self.strategy_params_map[label] = {
                'parallel': subset['target_max_parallel_workers_per_gather'].mode()[0],
                'jit': subset['target_jit'].mode()[0],
                'hashjoin': subset['target_enable_hashjoin'].mode()[0],
                'nestloop': subset['target_enable_nestloop'].mode()[0],
                'indexscan': subset['target_enable_indexscan'].mode()[0],
            }
            # Расчет памяти: 90-й перцентиль и привязка к бину (4, 8, 16...)
            p90_mem = np.percentile(subset['target_work_mem_mb'], 90)
            self.memory_map[label] = min([b for b in self.mem_bins if b >= p90_mem] or [self.mem_bins[-1]])

        # Модель
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            auto_class_weights='Balanced',
            random_seed=self.random_state,
            verbose=200,
            early_stopping_rounds=50
        )

        self.model.fit(X, y)
        logger.info("Модель успешно обучена.")

    def predict(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X_feat = self.extract_features(X_raw)
        preds = self.model.predict(X_feat).flatten()

        res = []
        for p in preds:
            params = self.strategy_params_map.get(p)
            res.append({
                'strategy_label': p,
                'pred_work_mem_mb': self.memory_map.get(p, 64),
                'target_max_parallel_workers_per_gather': params['parallel'],
                'target_jit': params['jit'],
                'target_enable_hashjoin': params['hashjoin'],
                'target_enable_nestloop': params['nestloop'],
                'target_enable_indexscan': params['indexscan']
            })
        return pd.DataFrame(res, index=X_raw.index)


def run_pipeline(uri: str, table_name: str):
    # 1. Загрузка данных
    engine = create_engine(uri)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    logger.info(f"Загружено {len(df)} записей.")

    # 2. Очистка типов
    cols_to_fix = ['target_work_mem_mb', 'target_max_parallel_workers_per_gather',
                   'feature_total_cost', 'feature_plan_rows', 'metric_max_log_planner_error']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Честный сплит
    # Создаем временный лейбл для стратификации
    temp_model = PGStrategyModel()
    df['tmp_label'] = temp_model._create_target_label(df)

    # Убираем синглтоны (классы с 1 записью), они не делятся стратифицированно
    counts = df['tmp_label'].value_counts()
    df = df[df['tmp_label'].isin(counts[counts > 1].index)]

    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df['tmp_label'], random_state=42
    )

    # 4. Обучение
    model = PGStrategyModel()
    model.train(train_df)

    # 5. Оценка
    predictions = model.predict(test_df)

    # Accuracy стратегии
    y_true = test_df['tmp_label']
    y_pred = predictions['strategy_label']
    acc = accuracy_score(y_true, y_pred)

    # Оценка памяти (Sufficiency)
    gt_mem = test_df['target_work_mem_mb'].values
    pr_mem = predictions['pred_work_mem_mb'].values
    sufficiency = (pr_mem >= gt_mem).mean()
    mae_mem = mean_absolute_error(gt_mem, pr_mem)

    print("\n" + "=" * 50)
    print(f"РЕЗУЛЬТАТЫ (Тестовая выборка):")
    print(f"Accuracy (Strategy ID): {acc:.2%}")
    print(f"Memory Sufficiency:     {sufficiency:.2%}")
    print(f"Memory MAE:             {mae_mem:.2f} MB")
    print("=" * 50)

    # Важность признаков
    import matplotlib.pyplot as plt
    feat_imp = pd.Series(model.model.get_feature_importance(), index=model.feature_cols).sort_values()
    print("\nВажность признаков:")
    print(feat_imp.tail(10))


if __name__ == "__main__":
    # Укажите ваши данные подключения
    DB_URI = "postgresql://suser:example@localhost:5432/model_tuning_pg"
    TABLE = "ml_training_dataset_new"

    try:
        run_pipeline(DB_URI, TABLE)
    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")
