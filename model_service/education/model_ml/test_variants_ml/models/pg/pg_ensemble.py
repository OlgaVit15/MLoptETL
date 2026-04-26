import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib


class PostgresCascadeEnsemble:
    def __init__(self, max_groups=60, n_estimators=100):
        self.max_groups = max_groups
        self.n_estimators = n_estimators
        # Модели-ансамбли для точности
        self.model_base = None  # RandomForest для DOP 0
        self.model_strategy = None  # RandomForest для DOP выбор
        self.min_samples_leaf_ratio = 0.005
        # Модель-группировщик для стабильного ID (решает ошибку 2D массива)
        self.grouper = None

        self.leaf_profiles = {}

        # Все 12 фич плана, которые мы будем использовать
        self.plan_features = [
            'feature_total_cost', 'feature_plan_rows', 'feature_plan_width',
            'feature_num_joins', 'feature_num_scans', 'feature_num_aggs',
            'feature_num_sorts', 'feature_num_filters', 'feature_num_index_scans',
            'feature_num_mem_nodes', 'feature_total_scan_size_bytes', 'feature_max_node_cost'
        ]
        # Колонки после логарифмирования для обучения
        self.feature_cols = ([f"log_{col}" for col in self.plan_features] +
                             ['index_ratio', 'mem_node_density', 'index_ratio', 'mem_node_density', 'cost_per_row',
                              'scan_intensity', 'join_complexity'])

    def _prepare_features(self, df):
        X = df.copy()
        # 1. Логарифмирование всех числовых признаков плана (избегаем перекосов из-за больших весов)
        for col in self.plan_features:
            val = pd.to_numeric(X.get(col, 0), errors='coerce').fillna(0)
            X[f"log_{col}"] = np.log1p(val)

        # 2. Производные фичи (соотношения)
        idx_scans = pd.to_numeric(X.get('feature_num_index_scans', 0), errors='coerce')
        total_scans = pd.to_numeric(X.get('feature_num_scans', 0), errors='coerce')
        X['index_ratio'] = idx_scans / (total_scans + 1)

        mem_nodes = pd.to_numeric(X.get('feature_num_mem_nodes', 0), errors='coerce')
        total_nodes = pd.to_numeric(X.get('feature_num_joins', 0), errors='coerce') + total_scans + 1
        X['mem_node_density'] = mem_nodes / total_nodes

        X['cost_per_row'] = X['feature_total_cost'] / (X['feature_plan_rows'] + 1)

        X['scan_intensity'] = X['feature_total_scan_size_bytes'] / (X['feature_num_scans'] + 1)

        X['join_complexity'] = X['feature_num_joins'] * X['log_feature_plan_rows']
        return X

    def train(self, df_train_raw):
        n_samples = len(df_train_raw)
        calculated_min_leaf = max(5, int(n_samples * self.min_samples_leaf_ratio))

        # Инициализация моделей
        self.model_error = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=calculated_min_leaf,
            random_state=42, n_jobs=-1
        )
        self.model_parallel = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=calculated_min_leaf,
            random_state=42, n_jobs=-1
        )
        self.grouper = DecisionTreeClassifier(
            max_leaf_nodes=self.max_groups,
            min_samples_leaf=calculated_min_leaf,
            random_state=42,
            max_depth=15,
            criterion='entropy'  # Энтропия лучше ловит редкие классы, чем Gini
        )

        df_train = self._prepare_features(df_train_raw)

        # ШАГ 1: Обучаем предсказание ошибки планировщика (Таргет: metric_planner_error_ratio)
        y_error = np.log1p(df_train_raw['metric_planner_error_ratio'].fillna(1.0))
        self.model_error.fit(df_train[self.feature_cols], y_error)
        df_train['pred_error'] = self.model_error.predict(df_train[self.feature_cols])

        # ШАГ 2: Обучаем параллелизм (Таргет: target_parallel_workers)
        # Используем предсказанную ошибку как важный признак
        strategy_features = self.feature_cols + ['pred_error']
        y_parallel = df_train_raw['target_max_parallel_workers_per_gather'].fillna(0).astype(int)
        self.model_parallel.fit(df_train[strategy_features], y_parallel)
        df_train['pred_parallel'] = self.model_parallel.predict(df_train[strategy_features])

        # ШАГ 3: Группировщик для профилей
        df_train['composite_strategy'] = (
                df_train_raw['target_max_parallel_workers_per_gather'].astype(str) + "_" +
                df_train_raw['target_enable_nestloop'].astype(str) + "_" +
                df_train_raw['target_enable_hashjoin'].astype(str) + "_" +
                df_train_raw['target_enable_mergejoin'].astype(str) + "_" +
                df_train_raw['target_enable_indexscan'].astype(str) + "_" +
                df_train_raw['target_enable_bitmapscan'].astype(str) + "_" +
                df_train_raw['target_enable_seqscan'].astype(str)
        )
        y_grouper, _ = pd.factorize(df_train['composite_strategy'])
        grouper_features = strategy_features + ['pred_parallel']
        self.grouper.fit(df_train[grouper_features], y_grouper)
        df_train['leaf_id'] = self.grouper.apply(df_train[grouper_features])

        # Группируем таргеты по логике
        join_targets = ['target_enable_hashjoin', 'target_enable_mergejoin', 'target_enable_nestloop']
        # , 'target_enable_indexscan', 'target_enable_seqscan', 'target_enable_bitmapscan']
        index_targets = ['target_enable_indexscan', 'target_enable_seqscan', 'target_enable_bitmapscan']

        for leaf_id in df_train['leaf_id'].unique():
            indices = df_train[df_train['leaf_id'] == leaf_id].index
            opt_data = df_train_raw.loc[indices]

            if opt_data.empty:
                continue

            # Сохраняем самые частые (mode) значения для флагов и медиану для памяти
            safe_mem = np.percentile(opt_data['target_work_mem_mb'], 85) * 1.15

            # 1. СТРАТЕГИЯ ДЖОЙНОВ (ищем самую частую комбинацию из 3-х флагов)
            # value_counts().idxmax() вернет кортеж, например ('ON', 'OFF', 'ON')
            best_join_comb = opt_data[join_targets].value_counts().idxmax()
            join_profile = dict(zip(join_targets, best_join_comb))

            # 2. СТРАТЕГИЯ ИНДЕКСОВ (самая частая комбинация из 3-х флагов)
            best_index_comb = opt_data[index_targets].value_counts().idxmax()
            index_profile = dict(zip(index_targets, best_index_comb))

            profile = {
                'work_mem_mb': int(np.ceil(safe_mem / 4.0) * 4),
                'target_max_parallel_workers_per_gather': int(
                    opt_data['target_max_parallel_workers_per_gather'].mode()[0]),
                'target_jit': opt_data['target_jit'].mode()[0],
                **join_profile
                **index_profile
            }

            self.leaf_profiles[leaf_id] = profile

        return self

    def predict(self, plan_dict):
        # Превращаем входной словарь в DataFrame с префиксами feature_ (как в парсере)
        df_input = self._prepare_features(pd.DataFrame([plan_dict]))

        # 1. Предсказание ошибки
        pred_error = self.model_error.predict(df_input[self.feature_cols])[0]
        df_input['pred_error'] = pred_error

        # 2. Предсказание параллелизма
        strategy_features = self.feature_cols + ['pred_error']
        pred_parallel = self.model_parallel.predict(df_input[strategy_features])[0]
        df_input['pred_parallel'] = pred_parallel

        # 3. Определение листа и профиля
        grouper_features = strategy_features + ['pred_parallel']
        leaf_id_raw = self.grouper.apply(df_input[grouper_features])
        leaf_id = int(leaf_id_raw[0])

        profile = self.leaf_profiles.get(leaf_id, {
            'work_mem_mb': 64, 'target_jit': 'on',
            'target_enable_indexscan': 'on', 'target_enable_seqscan': 'on',
            'target_enable_bitmapscan': 'on', 'target_enable_hashjoin': 'on',
            'target_enable_mergejoin': 'on', 'target_enable_nestloop': 'on'
        })

        # Формируем итоговый набор параметров для сессии Postgres
        return {
            'planner_error_estimate': float(np.expm1(pred_error)),
            'target_parallel_workers': int(pred_parallel),
            'target_work_mem': f"{profile['work_mem_mb']}MB",
            'target_jit': profile['target_jit'],
            'target_enable_indexscan': profile['target_enable_indexscan'],
            'target_enable_seqscan': profile['target_enable_seqscan'],
            'target_enable_bitmapscan': profile['target_enable_bitmapscan'],
            'target_enable_hashjoin': profile['target_enable_hashjoin'],
            'target_enable_mergejoin': profile['target_enable_mergejoin'],
            'target_enable_nestloop': profile['target_enable_nestloop'],
            'group_id': leaf_id
        }
