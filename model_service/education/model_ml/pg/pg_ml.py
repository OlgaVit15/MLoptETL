import pandas as pd
import numpy as np
from sklearn.tree import export_text
from sqlalchemy import create_engine

from model_ml.pg.pg_ensemble_2 import PostgresCascadeEnsemble
from model_ml.pg.pg_prep import PostgresRepresentativityBooster

pg = create_engine('postgresql://suser:example@localhost:5432/model_tuning_pg')


def load_dataset(engine):
    query = "SELECT * FROM ml_training_dataset"
    df = pd.read_sql(query, engine)

    numeric_cols = [
        'feature_total_cost'
        , 'feature_plan_rows'
        , 'feature_plan_width'
        , 'feature_num_joins'
        , 'feature_num_scans'
        , 'feature_num_aggs'
        , 'feature_num_sorts'
        , 'feature_num_filters'
        , 'feature_num_index_scans'
        , 'feature_num_mem_nodes'
        , 'feature_total_scan_size_bytes'
        , 'feature_max_node_cost'
        , 'target_work_mem_mb'
        , 'target_max_parallel_workers_per_gather'
        , 'metric_planner_error_ratio'
        , 'max_mem_limit_dop0'
        , 'metric_duration_ms'
        , 'metric_actual_rows'
        , 'metric_temp_written_blocks'
        , 'metric_shared_hit_blocks'
        , 'metric_shared_read_blocks'
        , 'metric_peak_memory_mb'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    cat_cols = [
        'target_jit',
        'target_enable_indexscan',
        'target_enable_seqscan',
        'target_enable_bitmapscan',
        'target_enable_hashjoin',
        'target_enable_mergejoin',
        'target_enable_nestloop'

    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    return df


def debug_postgres_model_logic(trained_model, train_df):
    """
    аудит логики каскада.
    """
    print("\n" + "=" * 60)
    print("АУДИТ ОБУЧЕНИЯ")
    print("=" * 60)

    # 1. Проверка количества групп
    X_strat = trained_model._prepare_features(train_df)
    X_strat['pred_error'] = trained_model.model_error.predict(X_strat[trained_model.feature_cols])
    X_strat['pred_parallel'] = trained_model.model_parallel.predict(
        X_strat[trained_model.feature_cols + ['pred_error']])

    leaf_indices = trained_model.grouper.apply(X_strat[trained_model.feature_cols + ['pred_error', 'pred_parallel']])
    unique_groups = np.unique(leaf_indices)

    print(f"Активных групп (листьев): {len(unique_groups)} из {trained_model.max_groups} возможных")
    print(f"Распределение по группам: {np.bincount(leaf_indices)[unique_groups]}")

    # 2. Анализ Модели Ошибки
    error_importances = pd.Series(trained_model.model_error.feature_importances_,
                                  index=trained_model.feature_cols)
    print("\n[STEP 1] Топ фич для Planner Error:")
    print(error_importances.sort_values(ascending=False).head(3))

    # 3. Почему Importance 0?
    parallel_features = trained_model.feature_cols + ['pred_error']
    parallel_importances = pd.Series(trained_model.model_parallel.feature_importances_,
                                     index=parallel_features)

    print("\n[STEP 2] Влияние предсказанной ошибки на Parallelism:")
    p_err_imp = parallel_importances.get('pred_error', 0)
    print(f"Importance of 'pred_error': {p_err_imp:.6f}")
    if p_err_imp < 0.0001:
        print("!!! ПРЕДУПРЕЖДЕНИЕ: Модель параллелизма игнорирует ошибку планировщика.")
        print("Это значит, что таргеты в датасете не коррелируют с ошибкой.")

    # 4. Визуализация правил (только если групп > 1)
    if len(unique_groups) > 1:
        print("\n[STEP 3] Ключевые правила разделения на группы:")
        rules = export_text(trained_model.grouper,
                            feature_names=trained_model.feature_cols + ['pred_error', 'pred_parallel'],
                            max_depth=2)
        print(rules)
    else:
        print("\n[STEP 3] ВНИМАНИЕ: Все запросы попали в одну группу (Class 0).")
        print("Рекомендация: Уменьшите min_samples_leaf или увеличьте вариативность данных.")


def evaluate_postgres_model_smart(model_instance, test_df):
    results = []

    # Группируем флаги для оценки стратегий
    join_flags = ['target_enable_hashjoin', 'target_enable_mergejoin', 'target_enable_nestloop']
    index_flags = ['target_enable_indexscan', 'target_enable_seqscan', 'target_enable_bitmapscan']

    for _, row in test_df.iterrows():
        pred = model_instance.predict(row.to_dict())

        # Проверка совпадения СТРАТЕГИИ целиком
        joins_match = all(str(pred[f]).upper() == str(row[f]).upper() for f in join_flags)
        index_match = all(str(pred[f]).upper() == str(row[f]).upper() for f in index_flags)

        # Полное совпадение всех флагов (Full Combo)
        all_flags = join_flags + index_flags + ['target_jit']
        full_match = all(str(pred[f]).upper() == str(row[f]).upper() for f in all_flags)

        results.append({
            'error_mae': abs(np.log1p(row['metric_planner_error_ratio']) - np.log1p(pred['planner_error_estimate'])),
            'parallel_acc': int(pred['target_parallel_workers']) == int(row['target_max_parallel_workers_per_gather']),
            'joins_strategy_acc': joins_match,
            'index_strategy_acc': index_match,
            'full_strategy_acc': full_match,
            'spill_risk': int(pred['target_work_mem'].replace('MB', '')) <= row['target_work_mem_mb']
        })

    res_df = pd.DataFrame(results)

    print("n" + "=" * 45)
    print("SMART ACCURACY REPORT (STRATEGY BASED)")
    print("=" * 45)
    print(f"1. Planner Error (MAE Log):       {res_df['error_mae'].mean():.4f}")
    print(f"2. Parallel Workers Accuracy:      {res_df['parallel_acc'].mean() * 100:.2f}%")
    print(f"3. Join Strategy Match (3 flags):  {res_df['joins_strategy_acc'].mean() * 100:.2f}%")
    print(f"4. Index Strategy Match (3 flags): {res_df['index_strategy_acc'].mean() * 100:.2f}%")
    print(f"5. FULL COMBO MATCH:               {res_df['full_strategy_acc'].mean() * 100:.2f}%")
    print("-" * 45)
    print(f"Memory Underestimation Risk:      {res_df['spill_risk'].mean() * 100:.2f}%")

    return res_df


if __name__ == '__main__':
    # 1. Загрузка исходных "чистых" данных (912 строк)
    df_raw = load_dataset(pg)

    # 2. ПОДГОТОВКА ВЫБОРКИ
    booster = PostgresRepresentativityBooster()
    train_boosted, test_clean = booster.prepare_for_training(df_raw)

    # 5. ОБУЧЕНИЕ
    model = PostgresCascadeEnsemble(max_groups=100)
    model.train(train_boosted)

    # 6. ДЕБАГ (на трейне)
    debug_postgres_model_logic(model, train_boosted)

    # 6. ФИНАЛЬНЫЙ ТЕСТ
    # Тестируем на raw_test, который не проходил через Booster
    evaluate_postgres_model_smart(model, test_clean)
