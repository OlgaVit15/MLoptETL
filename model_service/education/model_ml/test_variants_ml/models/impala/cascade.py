import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class ImpalaCascadeModel:
    def __init__(self, max_groups=60):
        # Модель 1: Оценивает базовую сложность (память при DOP 0/1)
        self.model_base = DecisionTreeRegressor(max_leaf_nodes=max_groups, random_state=42)
        # Модель 2: Выбирает оптимальный mt_dop на основе плана и оценки первой модели
        self.model_strategy = DecisionTreeRegressor(max_leaf_nodes=max_groups, random_state=42)

        self.leaf_profiles = {}
        # Фичи плана
        self.feature_cols = [
            'log_num_joins', 'log_num_broadcast_joins', 'log_num_scan_nodes',
            'log_num_agg_nodes', 'log_num_files', 'log_total_scan_size',
            'log_max_cardinality', 'log_max_row_size', 'log_bytes_per_node'
        ]

    @staticmethod
    def _prepare_features(df):
        X = df.copy()
        for col in X.columns:
            if col.startswith('feature_plan_'):
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        X['log_num_joins'] = np.log1p(X['feature_plan_num_joins'])
        X['log_num_broadcast_joins'] = np.log1p(X['feature_plan_num_broadcast_joins'])
        X['log_num_scan_nodes'] = np.log1p(X['feature_plan_num_scan_nodes'])
        X['log_num_agg_nodes'] = np.log1p(X['feature_plan_num_agg_nodes'])
        X['log_num_files'] = np.log1p(X['feature_plan_num_files'])
        X['log_total_scan_size'] = np.log1p(X['feature_plan_total_scan_size_bytes'])
        X['log_max_cardinality'] = np.log1p(X['feature_plan_max_cardinality'])
        X['log_max_row_size'] = np.log1p(X['feature_plan_max_row_size'])

        nodes = X['feature_plan_num_scan_nodes'].replace(0, 1)
        X['log_bytes_per_node'] = np.log1p(X['feature_plan_total_scan_size_bytes'] / nodes)
        return X

    def train(self, df_train_raw):
        print("Распределение таргетов DOP:")
        print(df_train_raw['target_mt_dop'].value_counts())

        # print("\nВариативность признаков (std):")
        # print(df_train_raw[self.feature_cols].std())
        # 1. Обучаем базу (что было бы при DOP 0)
        df_train = self._prepare_features(df_train_raw)
        y_base = pd.to_numeric(df_train_raw['max_mem_limit_dop0'], errors='coerce').fillna(df_train_raw['metric_pmu'])
        self.model_base.fit(df_train[self.feature_cols], y_base)

        # 2. Обучаем выбор DOP
        # Добавляем предсказание базы как фичу для второй модели
        df_train['pred_base_pmu'] = self.model_base.predict(df_train[self.feature_cols])
        strategy_features = self.feature_cols + ['pred_base_pmu']
        y_dop = pd.to_numeric(df_train_raw['target_mt_dop'], errors='coerce').fillna(2)
        self.model_strategy.fit(df_train[strategy_features], y_dop)

        # 3. Сохраняем профили на основе финальных групп (листьев второй модели)
        df_train['leaf_id'] = self.model_strategy.apply(df_train[strategy_features])
        for leaf_id in df_train['leaf_id'].unique():
            leaf_data = df_train_raw.iloc[df_train[df_train['leaf_id'] == leaf_id].index]
            opt = leaf_data[leaf_data['is_success'] == 1]

            if not opt.empty:
                # Считаем коэффициент изменения памяти (насколько параллелизм снизил/увеличил базу)
                adj_factor = (opt['metric_pmu'] / opt['max_mem_limit_dop0']).median()
                self.leaf_profiles[leaf_id] = {
                    'pmu_adj': adj_factor,
                    'mt_dop': int(opt['target_mt_dop'].mode()[0]),
                    'threads': int(opt['target_num_scanner_threads'].median()),
                    'join_mode': opt['target_default_join_distribution_mode'].mode()[0],
                    'codegen': str(opt['target_disable_codegen'].mode()[0]).lower(),
                    'success_rate': leaf_data['is_success'].mean()
                }
        return self

    def predict(self, plan_dict):
        df_input = self._prepare_features(pd.DataFrame([plan_dict]))

        # Шаг 1: Оценка базы
        base_pmu = self.model_base.predict(df_input[self.feature_cols])[0]

        # Шаг 2: Выбор стратегии
        df_input['pred_base_pmu'] = base_pmu
        strategy_features = self.feature_cols + ['pred_base_pmu']
        leaf_id = self.model_strategy.apply(df_input[strategy_features])[0]

        profile = self.leaf_profiles.get(leaf_id, {'pmu_adj': 1.1, 'mt_dop': 2, 'success_rate': 0})

        # Итоговая память = Базовая оценка * Коэффициент из профиля
        final_mem = base_pmu * profile.get('pmu_adj', 1.0)
        # stable_mem = int(np.ceil((final_mem * 1.25) / 256.0) * 256)
        stable_mem = int(np.ceil((final_mem * 1.30) / 256.0) * 256)  # Подняли с 1.25 до 1.30

        return {
            'target_mem_limit_dop0': base_pmu,
            'target_mem_limit': f"{max(256, stable_mem)}mb",
            'target_mt_dop': int(profile['mt_dop']),
            'target_num_scanner_threads': int(profile.get('threads', 4)),
            'target_default_join_distribution_mode': profile.get('join_mode', 'BROADCAST'),
            'target_disable_codegen': profile.get('codegen', 'false'),
            'reliability_score': round(float(profile['success_rate']), 2),
            'group_id': leaf_id
        }
