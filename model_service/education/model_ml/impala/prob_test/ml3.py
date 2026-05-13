import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.tree import export_text, plot_tree

from model_ml.impala.ensemble2 import ImpalaCascadeEnsemble
from model_ml.impala.prep import ImpalaDatasetPreparer

pg = create_engine('postgresql://suser:example@localhost:5432/model_tuning')
dsPrep = ImpalaDatasetPreparer()


def debug_model_logic(trained_model, model_type='cascade'):
    """
    Анализирует обученные внутренние деревья для разных типов моделей.
    """
    if model_type == 'cascade':
        # Для каскада визуализируем вторую модель (стратегию DOP)
        inner_tree = trained_model.model_strategy
        cols = trained_model.feature_cols + ['pred_base_pmu']  # Вторая модель использует предсказание первой как фичу
        title_suffix = "(Strategy Model - DOP Selection)"
    elif model_type == 'ensemble':
        # Для ансамбля визуализируем первое дерево из леса
        inner_tree = trained_model.model.estimators_[0]
        cols = trained_model.feature_cols
        title_suffix = "(First Tree from Random Forest)"
    else:
        print("Unknown ml_models type for debugging.")
        return

    print("\n" + "=" * 50)
    print(f"АНАЛИЗ ЛОГИКИ МОДЕЛИ {model_type.upper()} {title_suffix}")
    print("=" * 50)

    # 1. Текстовые правила
    try:
        rules = export_text(inner_tree, feature_names=cols, max_depth=5)  # Ограничим глубину для читаемости
        print("\n--- ПРАВИЛА ПРИНЯТИЯ РЕШЕНИЙ (Max Depth 5) ---")
        print(rules)
    except Exception as e:
        print(f"Could not export text rules: {e}")

    # 2. Важность признаков
    try:
        importances = pd.Series(inner_tree.feature_importances_, index=cols)
        print("\n--- ВАЖНОСТЬ ПРИЗНАКОВ (0.0 - 1.0) ---")
        print(importances.sort_values(ascending=False))
    except Exception as e:
        print(f"Could not get feature importances: {e}")

    # 3. Визуализация дерева
    try:
        plt.figure(figsize=(24, 12))  # Увеличиваем размер для лучшей читаемости
        plot_tree(inner_tree,
                  feature_names=cols,
                  filled=True,
                  rounded=True,
                  fontsize=7,  # Уменьшаем размер шрифта для сложных деревьев
                  precision=2,
                  max_depth=3)  # Ограничиваем глубину визуализации
        plt.title(f"Визуальное дерево групп запросов {title_suffix}")
        plt.show()
    except Exception as e:
        print(f"Could not plot tree: {e}")


def load_dataset(engine):
    query = "SELECT * FROM ml_training_dataset"
    df = pd.read_sql(query, engine)

    numeric_cols = [
        'metric_pmu', 'target_mem_limit_dop0',  # <-- Важно для каскада
        'feature_plan_num_joins', 'feature_plan_num_scan_nodes',
        'feature_plan_num_agg_nodes', 'feature_plan_total_scan_size_bytes',
        'feature_plan_max_cardinality', 'feature_plan_max_row_size',
        'target_mt_dop', 'target_num_scanner_threads', 'is_success'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    cat_cols = ['target_default_join_distribution_mode', 'target_disable_codegen']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    df = df[df['metric_pmu'] > 0]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def evaluate_impala_model(model_instance, df_raw, model_type):
    train_df, test_df = dsPrep.prepare_full_pipeline(df_raw)
    print(f"Training on {len(train_df)} queries, Testing on {len(test_df)} queries...")

    # 2. Обучаем модель
    model_instance.train(train_df)

    results = []

    # 3. Прогоняем тест
    for _, row in test_df.iterrows():
        plan_dict = row.to_dict()
        prediction = model_instance.predict(plan_dict)

        pred_mem = int(prediction['target_mem_limit'].replace('mb', ''))
        actual_mem = row['metric_pmu']

        error_mb = pred_mem - actual_mem
        is_oom = pred_mem < actual_mem

        # Санитизация для сравнения
        actual_mt_dop = int(row['target_mt_dop']) if pd.notna(row['target_mt_dop']) else 0
        actual_mem_limit_dop0 = int(row['target_mem_limit_dop0']) if pd.notna(row['target_mem_limit_dop0']) else 0
        actual_threads = int(row['target_num_scanner_threads']) if pd.notna(row['target_num_scanner_threads']) else 0
        actual_join_mode = str(row['target_default_join_distribution_mode']).upper() if pd.notna(
            row['target_default_join_distribution_mode']) else ''
        actual_codegen = str(row['target_disable_codegen']).lower() if pd.notna(row['target_disable_codegen']) else ''
        prediction_mem0 = int(prediction['target_mem_limit_dop0'])
        results.append({
            'actual_pmu': actual_mem,
            'pred_limit': pred_mem,
            'error_mb': error_mb,
            'is_oom_event': is_oom,
            'waste_mb': error_mb if error_mb > 0 else 0,
            # Точность по DOP (приведение к int)
            'mt_dop_correct': int(prediction['target_mt_dop']) == actual_mt_dop,
            'mem_limit_correct': actual_mem_limit_dop0 <= prediction_mem0 <= actual_mem_limit_dop0 + 256,
            'threads_correct': int(prediction['target_num_scanner_threads']) == actual_threads,
            'join_mode_correct': prediction['target_default_join_distribution_mode'].upper() == actual_join_mode,
            'codegen_correct': prediction['target_disable_codegen'].lower() == actual_codegen,

            'group_id': prediction['group_id'] if 'group_id' in prediction else -1
        })

    res_df = pd.DataFrame(results)
    value_scores = {}

    print("\n" + "=" * 40)
    print("IMPALA MODEL ACCURACY REPORT")
    print("=" * 40)

    oom_rate = res_df['is_oom_event'].mean() * 100
    value_scores['oom_rate'] = oom_rate
    print(f"OOM Risk (Underestimation): {oom_rate:.2f}%")
    if oom_rate > 5:
        print("  -> WARNING: Too many potential OOMs. Increase quantile or buffer.")
    else:
        print("  -> SUCCESS: Model is safe.")

    avg_waste = res_df['waste_mb'].mean()
    value_scores['avg_waste'] = avg_waste
    print(f"Average Memory Waste: {avg_waste:.2f} MB per query")

    print(f"MT_DOP Prediction Accuracy: {res_df['mt_dop_correct'].mean() * 100:.2f}%")
    print(f"MEM_LIMIT Prediction Accuracy: {res_df['mem_limit_correct'].mean() * 100:.2f}%")
    print(f"Scanner Threads Prediction Accuracy: {res_df['threads_correct'].mean() * 100:.2f}%")
    print(f"Join Mode Prediction Accuracy: {res_df['join_mode_correct'].mean() * 100:.2f}%")
    print(f"Codegen Prediction Accuracy: {res_df['codegen_correct'].mean() * 100:.2f}%")

    if 'group_id' in res_df.columns:
        print(f"Active Groups: {res_df['group_id'].nunique()}")

    huge_errors = res_df[res_df['error_mb'].abs() > 1024]
    value_scores['huge_errors'] = len(huge_errors)
    print(f"Critical Mispredictions (>1GB): {len(huge_errors)} cases")

    return res_df, value_scores


if __name__ == '__main__':
    df_full_dataset = load_dataset(pg)
    print(df_full_dataset['target_mt_dop'].value_counts())

    # print("\n--- ОЦЕНКА КАСКАДНОЙ МОДЕЛИ ---")
    # cascade_model = ImpalaCascadeModel(max_groups=50)
    # eval_cascade_results = evaluate_impala_model(cascade_model, df_full_dataset, model_type='cascade')

    # print("\n--- ОЦЕНКА АНСАМБЛЕВОЙ МОДЕЛИ (RANDOM FOREST) ---")
    # ensemble_model = ImpalaProEnsembleModel(n_trees=90)  # Уменьшаем n_trees для скорости
    # eval_ensemble_results = evaluate_impala_model(ensemble_model, df_full_dataset, model_type='ensemble')

    print("\n--- ОЦЕНКА МОДЕЛИ ---")
    censemble_model = ImpalaCascadeEnsemble(n_estimators=150)  # Уменьшаем n_trees для скорости
    eval_censemble_results, score = evaluate_impala_model(censemble_model, df_full_dataset, model_type='ensemble')
    # if score['oom_rate'] < 1.5 and score['avg_waste'] < 1024 and score['huge_errors'] == 0:
    #     censemble_model.save("impala_cascade_ensemble_model.joblib")


    # print("\n--- ОЦЕНКА МОДЕЛИ ---")
    # ensemble_model = ImpalaUltraPrecisionModel()  # Уменьшаем n_trees для скорости
    # eval_ensemble_results = evaluate_impala_model(ensemble_model, df_full_dataset, model_type='ensemble')

