
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # Добавим DecisionTreeRegressor
from sklearn.linear_model import HuberRegressor # Более устойчивый регрессор для ошибок
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Для масштабирования фич, если нужно

class PostgresCascadeEnsemble:
    def __init__(self, max_groups=80, n_estimators=200, min_samples_leaf_ratio=0.005):
        self.max_groups = max_groups
        self.n_estimators = n_estimators
        self.min_samples_leaf_ratio = min_samples_leaf_ratio

        # Обновленный список фич
        self.plan_features = [
            'feature_total_cost', 'feature_plan_rows', 'feature_plan_width',
            'feature_num_joins', 'feature_num_scans', 'feature_num_aggs',
            'feature_num_sorts', 'feature_num_filters', 'feature_num_index_scans',
            'feature_num_mem_nodes', 'feature_total_scan_size_bytes', 'feature_max_node_cost',
            'metric_duration_ms', # Добавим длительность выполнения как фичу
            'metric_planner_error_ratio' # Добавим саму ошибку, чтобы модель могла её "увидеть"
        ]

        # ЧИСТЫЙ список фич без дублей
        self.feature_cols = [f"log_{col}" for col in self.plan_features] + [
            'idx_ratio', 'mem_density', 'log_cost_per_row', 'log_scan_intensity',
            'log_duration_ms', 'log_planner_error_ratio' # Логарифмы новых фич
        ]

        self.leaf_profiles = {}
        self.grouper_model = None
        self.model_error_regressor = None
        self.model_parallel_clf = None
        self.scaler = StandardScaler() # Добавим скейлер

    def _prepare_features(self, df, is_training=False):
        X = df.copy()
        # 1. Логарифмируем базу
        for col in self.plan_features:
            if col in X.columns:
                # Обработка потенциальных ошибок при преобразовании к числовому типу
                val = pd.to_numeric(X.get(col, 0), errors='coerce').fillna(0)
                X[f"log_{col}"] = np.log1p(val)
            else:
                X[f"log_{col}"] = 0 # Если фича отсутствует, присваиваем 0

        # 2. Умные производные фичи (в LOG-шкале!)
        # Плотность стоимости: log(cost) - log(rows)
        if 'log_feature_total_cost' in X.columns and 'log_feature_plan_rows' in X.columns:
            X['log_cost_per_row'] = X['log_feature_total_cost'] - X['log_feature_plan_rows']
        else:
            X['log_cost_per_row'] = 0

        # Интенсивность сканирования
        if 'log_feature_total_scan_size_bytes' in X.columns and 'log_feature_total_cost' in X.columns:
            X['log_scan_intensity'] = X['log_feature_total_scan_size_bytes'] - X['log_feature_total_cost']
        else:
            X['log_scan_intensity'] = 0

        # Соотношения
        if 'feature_num_scans' in X.columns:
            X['idx_ratio'] = X['feature_num_index_scans'] / (X['feature_num_scans'] + 1e-6) # Добавим epsilon
        else:
            X['idx_ratio'] = X['feature_num_index_scans']

        if 'feature_num_joins' in X.columns and 'feature_num_scans' in X.columns:
            X['mem_density'] = X['feature_num_mem_nodes'] / (X['feature_num_joins'] + X['feature_num_scans'] + 1e-6)
        else:
            X['mem_density'] = X['feature_num_mem_nodes']

        # Добавляем логарифмы для новых фич
        if 'log_metric_duration_ms' not in X.columns and 'metric_duration_ms' in X.columns:
            X['log_metric_duration_ms'] = np.log1p(pd.to_numeric(X['metric_duration_ms'], errors='coerce').fillna(0))
        if 'log_metric_planner_error_ratio' not in X.columns and 'metric_planner_error_ratio' in X.columns:
             X['log_metric_planner_error_ratio'] = np.log1p(pd.to_numeric(X['metric_planner_error_ratio'], errors='coerce').fillna(1.0)) # Заполняем 1.0, как в train

        # Отбираем только нужные колонки
        # Убедимся, что все колонки существуют
        present_feature_cols = [col for col in self.feature_cols if col in X.columns]
        X = X[present_feature_cols]

        # Удаляем строки с NaN после всех преобразований
        X.dropna(inplace=True)

        return X

    def train(self, df_train_raw):
        # 1. Подготовка данных
        X_train_raw = self._prepare_features(df_train_raw, is_training=True)
        calculated_min_leaf = max(5, int(len(X_train_raw) * self.min_samples_leaf_ratio))

        # Фичи для разных моделей
        features_for_error = [col for col in self.feature_cols if col.startswith('log_')] # Используем только логарифмированные
        features_for_parallel = features_for_error + ['pred_error']
        features_for_grouper = features_for_parallel + ['pred_parallel']

        # 2. Модель 1: Предсказание ОШИБКИ
        # Используем HuberRegressor как более устойчивый к выбросам, или RandomForestRegressor
        # RandomForestRegressor может быть лучше, если ошибки имеют сложную структуру
        self.model_error_regressor = RandomForestRegressor(n_estimators=self.n_estimators,
                                                         min_samples_leaf=calculated_min_leaf,
                                                         random_state=42,
                                                         n_jobs=-1) # Используем все ядра CPU

        # Целевая переменная для ошибки
        y_error = np.log1p(df_train_raw['metric_planner_error_ratio'].fillna(1.0))
        # Сопоставляем y_error с X_train_raw после _prepare_features
        y_error = y_error[X_train_raw.index] # Убедимся, что индексы совпадают

        self.model_error_regressor.fit(X_train_raw[features_for_error], y_error)

        # Предсказываем ошибку и добавляем ее как новую фичу
        X_train_raw['pred_error'] = self.model_error_regressor.predict(X_train_raw[features_for_error])

        # 3. Модель 2: Предсказание Параллелизма
        self.model_parallel_clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                                       min_samples_leaf=calculated_min_leaf,
                                                       random_state=42,
                                                       n_jobs=-1)

        # Целевая переменная для параллелизма (целочисленная)
        y_parallel = df_train_raw['target_max_parallel_workers_per_gather'].astype(int)
        y_parallel = y_parallel[X_train_raw.index] # Убедимся, что индексы совпадают

        self.model_parallel_clf.fit(X_train_raw[features_for_parallel], y_parallel)

        # Предсказываем параллелизм и добавляем его как новую фичу
        X_train_raw['pred_parallel'] = self.model_parallel_clf.predict(X_train_raw[features_for_parallel])

        # 4. Модель 3: Группировщик (на основе стратегий)
        # Создаем комбинацию стратегий для группировки
        df_train_raw['strat_id'] = (
            df_train_raw['target_enable_nestloop'].astype(str) + "_" +
            df_train_raw['target_enable_indexscan'].astype(str) + "_" +
            df_train_raw['target_max_parallel_workers_per_gather'].astype(str) # Используем реальное значение
        )
        # Используем LabelEncoder для создания числовых меток, это надежнее factorize
        self.label_encoder = LabelEncoder()
        y_grouper = self.label_encoder.fit_transform(df_train_raw['strat_id'])
        y_grouper = y_grouper[X_train_raw.index] # Сопоставляем индексы

        # Можно попробовать DecisionTreeRegressor вместо Classifier, если мы хотим
        # предсказать числовую метку, а не класс. Но здесь, кажется, логичнее
        # классификатор, который будет выбирать "лучший" класс стратегии.
        self.grouper_model = DecisionTreeClassifier(max_leaf_nodes=self.max_groups,
                                                    min_samples_leaf=calculated_min_leaf,
                                                    random_state=42)

        self.grouper_model.fit(X_train_raw[features_for_grouper], y_grouper)

        # Получаем ID листьев для каждого обучающего примера
        leaf_ids = self.grouper_model.apply(X_train_raw[features_for_grouper])

        # 5. Сборка профилей для каждого листа (группы)
        join_targets = ['target_enable_hashjoin', 'target_enable_mergejoin', 'target_enable_nestloop']
        index_targets = ['target_enable_indexscan', 'target_enable_seqscan', 'target_enable_bitmapscan']
        # Дополнительные целевые переменные для профилей
        other_targets = ['target_jit', 'target_max_parallel_workers_per_gather', 'target_work_mem_mb']

        # Собираем все целевые переменные, которые нужно анализировать для профилей
        all_target_cols_for_profiles = join_targets + index_targets + other_targets
        # Убедимся, что все эти колонки существуют в df_train_raw
        all_target_cols_for_profiles = [col for col in all_target_cols_for_profiles if col in df_train_raw.columns]

        for lid in np.unique(leaf_ids):
            mask = (leaf_ids == lid)
            opt_data = df_train_raw[mask]

            if opt_data.empty:
                continue # Пропускаем пустые группы

            # Вычисляем профиль:
            profile_data = {}

            # Для булевых флагов: выбираем наиболее частый (mode)
            for col in join_targets + index_targets:
                if col in opt_data.columns:
                    # Считаем mode, если есть данные
                    mode_val = opt_data[col].mode()
                    if not mode_val.empty:
                        profile_data[col] = mode_val[0]
                    else:
                        profile_data[col] = 0 # Или другое значение по умолчанию
                else:
                    profile_data[col] = 0 # Если колонки нет

            # Для других целевых:
            if 'target_jit' in opt_data.columns:
                mode_val = opt_data['target_jit'].mode()
                profile_data['target_jit'] = mode_val[0] if not mode_val.empty else 0
            if 'target_max_parallel_workers_per_gather' in opt_data.columns:
                mode_val = opt_data['target_max_parallel_workers_per_gather'].mode()
                profile_data['target_max_parallel_workers_per_gather'] = int(mode_val[0]) if not mode_val.empty else 0
            if 'target_work_mem_mb' in opt_data.columns:
                # Вместо 90-го перцентиля, попробуем среднее или медиану,
                # чтобы избежать влияния редких высоких значений.
                # Или 90-й перцентиль, но с проверкой на разумность.
                percentile_90 = np.percentile(opt_data['target_work_mem_mb'], 90)
                profile_data['work_mem_mb'] = int(percentile_90) if not np.isnan(percentile_90) else 1024 # Значение по умолчанию

            # Сохраняем профиль, если он содержит какие-то данные
            if profile_data:
                self.leaf_profiles[lid] = profile_data

        # Если leaf_profiles пустой, это проблема. Нужно добавить хотя бы один профиль.
        if not self.leaf_profiles:
            print("Предупреждение: leaf_profiles пуст. Создаю дефолтный профиль.")
            self.leaf_profiles[0] = {
                'target_jit': 0, 'target_enable_hashjoin': 0, 'target_enable_mergejoin': 0, 'target_enable_nestloop': 1,
                'target_enable_indexscan': 1, 'target_enable_seqscan': 0, 'target_enable_bitmapscan': 0,
                'target_max_parallel_workers_per_gather': 0, 'work_mem_mb': 1024
            }

        # 6. Обучение скейлера
        # Используем фичи, которые будут использоваться для предсказания
        features_for_scaling = [col for col in self.feature_cols if col in X_train_raw.columns]
        self.scaler.fit(X_train_raw[features_for_scaling])

        return self

    def predict(self, plan_dict):
        # Создаем DataFrame из словаря
        df_input = pd.DataFrame([plan_dict])

        # 1. Подготовка фич для входных данных
        X_input = self._prepare_features(df_input)

        # Проверяем, есть ли фичи после подготовки
        if X_input.empty:
            print("Предупреждение: Входные данные после подготовки признаков пусты.")
            # Возвращаем дефолтный или предсказанный на основе большинства профиль
            default_profile_key = next(iter(self.leaf_profiles))
            default_profile = self.leaf_profiles[default_profile_key].copy()
            default_profile['planner_error_estimate'] = 1.0
            default_profile['target_parallel_workers'] = 0
            default_profile['target_work_mem'] = f"{default_profile.get('work_mem_mb', 1024)}MB"
            default_profile['group_id'] = int(default_profile_key)
            return default_profile


        # 2. Предсказание ошибки
        features_for_error = [col for col in self.feature_cols if col.startswith('log_')]
        features_for_error_present = [col for col in features_for_error if col in X_input.columns]

        if not features_for_error_present:
            print("Ошибка: Не найдены фичи для предсказания ошибки.")
            # Возвращаем дефолтный профиль
            default_profile_key = next(iter(self.leaf_profiles))
            default_profile = self.leaf_profiles[default_profile_key].copy()
            default_profile['planner_error_estimate'] = 1.0
            default_profile['target_parallel_workers'] = 0
            default_profile['target_work_mem'] = f"{default_profile.get('work_mem_mb', 1024)}MB"
            default_profile['group_id'] = int(default_profile_key)
            return default_profile

        # Применяем скейлер к входным фичам
        X_scaled = self.scaler.transform(X_input[features_for_error_present])
        X_scaled_df = pd.DataFrame(X_scaled, columns=features_for_error_present, index=X_input.index)


        p_err = self.model_error_regressor.predict(X_scaled_df)[0]
        X_input['pred_error'] = p_err # Добавляем предсказанную ошибку

        # 3. Предсказание параллелизма
        features_for_parallel = features_for_error_present + ['pred_error']
        features_for_parallel_present = [col for col in features_for_parallel if col in X_input.columns]

        # Применяем скейлер к входным фичам для параллелизма
        X_scaled_parallel_df = pd.DataFrame(self.scaler.transform(X_input[features_for_parallel_present]),
                                            columns=features_for_parallel_present, index=X_input.index)


        p_par = self.model_parallel_clf.predict(X_scaled_parallel_df)[0]
        X_input['pred_parallel'] = p_par # Добавляем предсказанный параллелизм

        # 4. Определение группы (листа дерева)
        features_for_grouper = features_for_parallel_present # Исправлено: используем только фичи, доступные в X_input
        features_for_grouper_present = [col for col in features_for_grouper if col in X_input.columns]

        # Применяем скейлер к входным фичам для группировщика
        X_scaled_grouper_df = pd.DataFrame(self.scaler.transform(X_input[features_for_grouper_present]),
                                            columns=features_for_grouper_present, index=X_input.index)


        # Получаем ID листа
        lid_pred = self.grouper_model.apply(X_scaled_grouper_df)[0]

        # 5. Получение профиля для предсказанной группы
        profile = self.leaf_profiles.get(lid_pred)
        if profile is None:
            # Если предсказанный лист отсутствует, берем наиболее частый профиль
            print(f"Предупреждение: Предсказанный лист {lid_pred} отсутствует в leaf_profiles. Использую дефолтный.")
            # Пытаемся найти наиболее частый профиль или первый попавшийся
            if self.leaf_profiles:
                default_profile_key = next(iter(self.leaf_profiles))
                profile = self.leaf_profiles[default_profile_key]
            else:
                # Крайний случай: если leaf_profiles совсем пуст
                profile = {
                    'target_jit': 0, 'target_enable_hashjoin': 0, 'target_enable_mergejoin': 0, 'target_enable_nestloop': 1,
                    'target_enable_indexscan': 1, 'target_enable_seqscan': 0, 'target_enable_bitmapscan': 0,
                    'target_max_parallel_workers_per_gather': 0, 'work_mem_mb': 1024
                }

        # 6. Формирование результата
        res = profile.copy()
        res['planner_error_estimate'] = float(np.expm1(p_err)) # Обратное преобразование логарифма
        res['target_parallel_workers'] = int(p_par)
        res['target_work_mem'] = f"{res.get('work_mem_mb', 1024)}MB" # Используем get для безопасности
        res['group_id'] = int(lid_pred) # ID предсказанной группы

        # Добавляем предсказанные значения, если они не перекрываются
        if 'pred_error' not in res:
            res['pred_error'] = float(np.expm1(p_err))
        if 'pred_parallel' not in res:
            res['pred_parallel'] = int(p_par)

        return res

