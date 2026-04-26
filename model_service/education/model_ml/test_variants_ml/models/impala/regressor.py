import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class ImpalaMLModel:
    def __init__(self, max_groups=50):
        # Используем дерево как "умный группировщик"
        # max_leaf_nodes — это по сути количество твоих "идеальных кластеров"
        self.model = DecisionTreeRegressor(
            max_leaf_nodes=max_groups,
            random_state=42,
            min_samples_leaf=5,  # Защита от переобучения на единичных запросах
            ccp_alpha=0.0
        )
        self.leaf_profiles = {}
        self.feature_cols = [
            'log_num_joins', 'log_num_scan_nodes', 'log_num_agg_nodes',
            'feature_plan_has_missing_stats', 'log_total_scan_size',
            'log_max_cardinality', 'log_max_row_size', 'log_bytes_per_node'
        ]

    @staticmethod
    def _prepare_features(df):
        X = df.copy()
        # Используем log1p для обработки нулей
        X['log_num_joins'] = np.log1p(X['feature_plan_num_joins'].fillna(0))
        X['log_num_scan_nodes'] = np.log1p(X['feature_plan_num_scan_nodes'].fillna(0))
        X['log_num_agg_nodes'] = np.log1p(X['feature_plan_num_agg_nodes'].fillna(0))
        X['log_total_scan_size'] = np.log1p(X['feature_plan_total_scan_size_bytes'].fillna(0))
        X['log_max_cardinality'] = np.log1p(X['feature_plan_max_cardinality'].fillna(0))
        X['log_max_row_size'] = np.log1p(X['feature_plan_max_row_size'].fillna(0))

        nodes = X['feature_plan_num_scan_nodes'].replace(0, 1)
        X['log_bytes_per_node'] = np.log1p(X['feature_plan_total_scan_size_bytes'] / nodes)
        return X

    def train(self, df_train_raw):
        # 1. Подготовка
        df_train_raw = df_train_raw.sample(frac=1, random_state=42).reset_index(drop=True)
        df_train = self._prepare_features(df_train_raw)
        X_feat = df_train[self.feature_cols].fillna(0)

        # Целевая переменная для построения дерева - Peak Memory Usage
        # Дерево разделит запросы на группы так, чтобы минимизировать разброс по памяти
        y = df_train['metric_pmu']

        self.model.fit(X_feat, y)

        # 2. Формируем профили для каждого "листа" (группы)
        # apply() возвращает индекс листа для каждой строки
        df_train['leaf_id'] = self.model.apply(X_feat)

        for leaf_id in df_train['leaf_id'].unique():
            leaf_data = df_train[df_train['leaf_id'] == leaf_id]
            # Берем только успешные запросы для формирования эталона
            opt = leaf_data[leaf_data['is_success'] == 1]

            if not opt.empty:
                # ПРАВИЛО НАДЕЖНОСТИ: 90-й квантиль по памяти для защиты от OOM
                pmu_90 = opt['metric_pmu'].quantile(0.9)

                self.leaf_profiles[leaf_id] = {
                    'mem_limit': int(pmu_90 * 1.15),  # Запас 15% сверху
                    'mem_limit_dop0': int(opt[opt['target_mt_dop'] <= 1]['metric_pmu'].quantile(0.9) * 1.1 if not opt[
                        opt['target_mt_dop'] <= 1].empty else pmu_90),
                    'mt_dop': int(opt['target_mt_dop'].mode()[0]),
                    'num_scanner_threads': int(opt['target_num_scanner_threads'].median()),
                    'join_mode': opt['target_default_join_distribution_mode'].mode()[0],
                    'disable_codegen': str(opt['target_disable_codegen'].mode()[0]).lower(),
                    'success_rate': leaf_data['is_success'].mean()
                }
            else:
                # Fallback если в группе нет успешных (бывает на малых данных)
                self.leaf_profiles[leaf_id] = {'mem_limit': 1024, 'success_rate': 0}

            return self

    def predict(self, plan_dict):
        df_input = pd.DataFrame([plan_dict])
        df_enriched = self._prepare_features(df_input)
        X_feat = df_enriched[self.feature_cols].fillna(0)

        # Определяем, в какой лист попадает запрос
        leaf_id = self.model.apply(X_feat)[0]
        profile = self.leaf_profiles.get(leaf_id)

        if not profile or profile.get('mem_limit', 0) == 0:
            return {
                'target_mem_limit': "1024mb",
                'target_mt_dop': 2,
                'target_num_scanner_threads': 4,
                'target_default_join_distribution_mode': 'BROADCAST',
                'target_disable_codegen': 'false',
                'group_id': -1,  # Специальный ID для пустых групп
                'reliability_score': 0.0
            }

        # Финальная стабилизация: Bucketing памяти (чтобы не плодить тысячи уникальных конфигов)
        # Округляем до ближайших 128МБ
        raw_mem = profile['mem_limit']
        stable_mem = int(np.ceil(raw_mem / 128.0) * 128)
        stable_mem = max(128, min(stable_mem, 32768))  # Ограничители [128MB, 32GB]

        return {
            'target_mem_limit': f"{stable_mem}mb",
            'target_mem_limit_exec_group_0': f"{profile['mem_limit_dop0']}mb",
            'target_mt_dop': profile['mt_dop'],
            'target_num_scanner_threads': profile['num_scanner_threads'],
            'target_default_join_distribution_mode': profile['join_mode'],
            'target_disable_codegen': profile['disable_codegen'],
            'group_id': int(leaf_id),
            'reliability_score': round(profile['success_rate'], 2)
        }
