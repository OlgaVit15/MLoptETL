import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib


class ImpalaCascadeEnsemble:
    def __init__(self, max_groups=60, n_estimators=50, min_samples_leaf_ratio=0.005):
        self.max_groups = max_groups
        self.n_estimators = n_estimators
        self.min_samples_leaf_ratio = min_samples_leaf_ratio

        # Модели-ансамбли для точности
        self.model_base = None  # RandomForest для памяти при 0 степени параллелизма
        self.model_strategy = None  # RandomForest для выбора степени параллелизма

        # Модель-группировщик для стабильного ID
        self.grouper = None

        self.leaf_profiles = {}
        self.feature_cols = [
            'log_num_joins', 'log_num_broadcast_joins', 'log_num_scan_nodes',
            'log_num_agg_nodes', 'log_num_files', 'log_total_scan_size',
            'log_max_cardinality', 'log_max_row_size', 'log_bytes_per_node'
        ]

    def _prepare_features(self, df):
        X = df.copy()
        # преобработка
        plan_map = {
            'log_num_joins': 'feature_plan_num_joins',
            'log_num_broadcast_joins': 'feature_plan_num_broadcast_joins',
            'log_num_scan_nodes': 'feature_plan_num_scan_nodes',
            'log_num_agg_nodes': 'feature_plan_num_agg_nodes',
            'log_num_files': 'feature_plan_num_files',
            'log_total_scan_size': 'feature_plan_total_scan_size_bytes',
            'log_max_cardinality': 'feature_plan_max_cardinality',
            'log_max_row_size': 'feature_plan_max_row_size'
        }

        for new_col, old_col in plan_map.items():
            val = pd.to_numeric(X.get(old_col, 0), errors='coerce').fillna(0)
            X[new_col] = np.log1p(val)

        nodes = pd.to_numeric(X.get('feature_plan_num_scan_nodes', 1), errors='coerce').replace(0, 1)
        size = pd.to_numeric(X.get('feature_plan_total_scan_size_bytes', 0), errors='coerce')
        X['log_bytes_per_node'] = np.log1p(size / nodes)
        return X

    def train(self, df_train_raw):
        # 1. Динамическая регуляризация
        n_samples = len(df_train_raw)
        calculated_min_leaf = max(5, int(n_samples * self.min_samples_leaf_ratio))

        # Инициализация моделей
        self.model_base = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=calculated_min_leaf,
            random_state=42, n_jobs=-1, max_depth=10,
            min_samples_split=5,
            max_features='sqrt',
        )
        self.model_strategy = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=calculated_min_leaf,
            random_state=42, n_jobs=-1
        )
        self.grouper = DecisionTreeClassifier(
            max_leaf_nodes=self.max_groups,
            min_samples_leaf=calculated_min_leaf,
            random_state=42
        )

        df_train = self._prepare_features(df_train_raw)

        # Обучаем Базу (Memory Prediction)
        y_base = pd.to_numeric(df_train_raw['target_mem_limit_dop0'], errors='coerce').fillna(df_train_raw['metric_pmu'])
        self.model_base.fit(df_train[self.feature_cols], y_base)

        # Обучаем Стратегию (DOP Prediction)
        df_train['pred_base_pmu'] = self.model_base.predict(df_train[self.feature_cols])
        strategy_features = self.feature_cols + ['pred_base_pmu']
        y_dop = pd.to_numeric(df_train_raw['target_mt_dop'], errors='coerce').fillna(2)
        self.model_strategy.fit(df_train[strategy_features], y_dop)

        # Обучаем Группировщик для создания стабильных 1D индексов групп
        self.grouper.fit(df_train[strategy_features], y_dop.astype(int))
        df_train['leaf_id'] = self.grouper.apply(df_train[strategy_features])

        # 2. Формируем профили
        for leaf_id in df_train['leaf_id'].unique():
            # Берем индексы строк, попавших в этот лист
            indices = df_train[df_train['leaf_id'] == leaf_id].index
            leaf_data = df_train_raw.loc[indices]

            # Считаем только по успешным запросам в этой группе
            opt = leaf_data
            # opt = leaf_data[leaf_data['is_success'] == 1]
            # success_rate = leaf_data['is_success'].mean()

            if not opt.empty:
                # PMU Adj: Отношение реальной памяти к оценке DOP0 (90-й квантиль для надежности)
                # pmu_ratio = (opt['target_mem_limit'] / opt['target_mem_limit_dop0']).replace([np.inf, -np.inf], 1.0).fillna(1.0)

                actual_pmu = pd.to_numeric(opt['metric_pmu'], errors='coerce').fillna(0)
                base_pmu = pd.to_numeric(opt['target_mem_limit_dop0'], errors='coerce').fillna(1)

                pmu_ratio = (actual_pmu / base_pmu).replace([np.inf, -np.inf], 1.0).fillna(1.0)

                self.leaf_profiles[leaf_id] = {
                    'pmu_adj': float(np.percentile(pmu_ratio, 70)),  # Квантиль 90
                    'mt_dop': int(opt['target_mt_dop'].mode()[0]),
                    'threads': int(opt['target_num_scanner_threads'].median()),
                    'join_mode': str(opt['target_default_join_distribution_mode'].mode()[0]),
                    'codegen': str(opt['target_disable_codegen'].mode()[0]).lower(),
                    # 'success_rate': float(success_rate)
                }
            else:
                # Fallback для "плохих" веток
                self.leaf_profiles[leaf_id] = {
                    'pmu_adj': 1.3, 'mt_dop': 2, 'threads': 4,
                    'join_mode': 'BROADCAST', 'codegen': 'false'
                    , 'success_rate': 0.0
                }
        return self

    def predict(self, plan_dict):
        df_input = self._prepare_features(pd.DataFrame([plan_dict]))

        # 1. Предсказание базовой памяти (DOP 0)
        base_pmu = self.model_base.predict(df_input[self.feature_cols])[0]

        # Добавляем pred_base_pmu как признак для следующей модели (CASCADE)
        df_input['pred_base_pmu'] = base_pmu
        strategy_features = self.feature_cols + ['pred_base_pmu']

        # 2. Предсказание MT_DOP с помощью model_strategy
        ml_dop_prediction = self.model_strategy.predict(df_input[strategy_features])[0]
        # ml_dop_confidence = np.max(self.model_strategy.predict_proba(df_input[strategy_features])[0])

        # 3. Получаем leaf_id через Grouper (для профилирования и FALLBACK)
        leaf_id_raw = self.grouper.apply(df_input[strategy_features])
        leaf_id = int(leaf_id_raw[0][0]) if leaf_id_raw.ndim > 1 else int(leaf_id_raw[0])

        profile = self.leaf_profiles.get(leaf_id)

        # 4. Логика объединения предсказаний и профиля (Гибридный подход)
        if not profile:
            # or profile['success_rate'] < 0.4:  # Если профиль не найден или ненадежен
            # Используем надежные дефолты и более консервативный буфер
            pmu_adj = 1.4
            mt_dop = max(1, int(ml_dop_prediction))
            threads = 4
            join_mode = 'SHUFFLE'
            codegen = 'false'
            reliability_score = 0
        else:
            # Если профиль надежен, используем его параметры, но DOP берем от ML
            pmu_adj = profile['pmu_adj']
            mt_dop = max(1, int(ml_dop_prediction))
            threads = profile['threads']
            join_mode = profile['join_mode']
            codegen = profile['codegen']
            # reliability_score = round(float(profile['success_rate']), 2)

        # 5. Итоговый расчет памяти (с округлением вверх до 256МБ или 128МБ для малых запросов)
        final_mem_raw = base_pmu * pmu_adj
        stable_mem = final_mem_raw
        # if base_pmu < 150:  # Если запрос реально небольшой
        #     stable_mem = int(np.ceil((final_mem_raw * 1.15) / 128.0) * 128)  # Меньший буфер и шаг 128
        # else:
        #     stable_mem = int(np.ceil((final_mem_raw * 1.25) / 256.0) * 256)  # Стандартный буфер и шаг 256

        return {
            'target_mem_limit_dop0': base_pmu,
            'target_mem_limit': f"{stable_mem}mb",
            'target_mt_dop': int(mt_dop),
            'target_num_scanner_threads': int(threads),
            'target_default_join_distribution_mode': join_mode,
            'target_disable_codegen': codegen,
            # 'reliability_score': reliability_score,
            'group_id': int(leaf_id)
        }

    def save(self, filename):
        joblib.dump(self, filename)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename):
        return joblib.load(filename)
