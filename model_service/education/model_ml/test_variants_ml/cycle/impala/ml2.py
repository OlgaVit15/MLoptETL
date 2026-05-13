from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.tree import export_text, plot_tree

from model_ml.ds_prepare import ImpalaDatasetPreparer
from model_ml.cascade import ImpalaCascadeModel

pg = create_engine('postgresql://suser:example@localhost:5432/model_tuning')

dsPrep = ImpalaDatasetPreparer()

def debug_model_logic(trained_model):
    """
    Анализирует обученное дерево внутри ImpalaMLModel
    """
    # Достаем саму модель sklearn из твоего класса
    inner_tree = trained_model.model
    cols = trained_model.feature_cols

    print("\n" + "=" * 50)
    print("АНАЛИЗ ЛОГИКИ МОДЕЛИ (ГРУППИРОВКА)")
    print("=" * 50)

    # 1. Текстовые правила (покажет пороги if-else)
    rules = export_text(inner_tree, feature_names=cols)
    print("\n--- ПРАВИЛА ПРИНЯТИЯ РЕШЕНИЙ ---")
    print(rules)

    # 2. Важность признаков (какие колонки реально влияют на память)
    importances = pd.Series(inner_tree.feature_importances_, index=cols)
    print("\n--- ВАЖНОСТЬ ПРИЗНАКОВ (0.0 - 1.0) ---")
    print(importances.sort_values(ascending=False))

    # 3. Визуализация дерева (всплывающее окно)
    plt.figure(figsize=(20, 10))
    plot_tree(inner_tree,
              feature_names=cols,
              filled=True,
              rounded=True,
              fontsize=9,
              precision=2)
    plt.title("Визуальное дерево групп запросов")
    plt.show()


def load_dataset(engine):
    # 1. Загружаем данные
    query = "SELECT * FROM ml_training_dataset"
    df = pd.read_sql(query, engine)

    # 2. Принудительное приведение типов (Критически важно!)
    # Список колонок, которые должны быть числами
    numeric_cols = [
        'metric_pmu', 'feature_plan_num_joins', 'feature_plan_num_scan_nodes',
        'feature_plan_num_agg_nodes', 'feature_plan_total_scan_size_bytes',
        'feature_plan_max_cardinality', 'feature_plan_max_row_size',
        'target_mt_dop', 'target_num_scanner_threads', 'is_success'
    ]

    for col in numeric_cols:
        if col in df.columns:
            # errors='coerce' превратит мусор в NaN, а fillna(0) сделает их числами
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Нормализация категориальных признаков (строки в нижний регистр)
    cat_cols = ['target_default_join_distribution_mode', 'target_disable_codegen']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

            # 4. Санитизация (Sanity Check)
    # Удаляем записи, где PMU нулевой или отрицательный (это шум в данных)
    df = df[df['metric_pmu'] > 0]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def evaluate_impala_model(model, df_raw):
    # 1. Разделяем данные на Train и Test (80/20)
    # Важно: перемешиваем (shuffle), чтобы не было временных перекосов
    # df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
    # df_raw['pmu_bin'] = pd.qcut(df_raw['metric_pmu'], q=5, labels=False, duplicates='drop')
    # train_df, test_df = train_test_split(df_raw, test_size=0.25, random_state=42, stratify=df_raw['pmu_bin'])
    # train_df = train_df.drop(columns=['pmu_bin'])
    # test_df = test_df.drop(columns=['pmu_bin'])
    train_df, test_df = dsPrep.prepare_full_pipeline(df_raw)

    print(f"Training on {len(train_df)} queries, Testing on {len(test_df)} queries...")

    # 2. Обучаем модель
    model.train(train_df)

    # debug_model_logic(ml_models)

    results = []

    # 3. Прогоняем тест
    for _, row in test_df.iterrows():
        # Формируем входной словарь (имитируем приход нового запроса)
        plan_dict = row.to_dict()
        prediction = model.predict(plan_dict)

        # Извлекаем числа из строк типа "512mb"
        pred_mem = int(prediction['target_mem_limit'].replace('mb', ''))
        actual_mem = row['metric_pmu']

        # Аналитика ошибки
        error_mb = pred_mem - actual_mem
        is_oom = pred_mem < actual_mem  # Предсказали меньше, чем реально съел

        results.append({
            'actual_pmu': actual_mem,
            'pred_limit': pred_mem,
            'error_mb': error_mb,
            'is_oom_event': is_oom,
            'waste_mb': error_mb if error_mb > 0 else 0,
            'mt_dop_correct': prediction['target_mt_dop'] == row['target_mt_dop'],
            'group_id': prediction['group_id']
        })

    res_df = pd.DataFrame(results)

    # --- ИТОГОВАЯ АНАЛИТИКА ---

    print("\n" + "=" * 30)
    print("IMPALA MODEL ACCURACY REPORT")
    print("=" * 30)

    # Метрика 1: Безопасность (OOM Rate)
    oom_rate = res_df['is_oom_event'].mean() * 100
    print(f"OOM Risk (Underestimation): {oom_rate:.2f}%")
    if oom_rate > 5:
        print("  -> WARNING: Too many potential OOMs. Increase quantile or buffer.")
    else:
        print("  -> SUCCESS: Model is safe.")

    # Метрика 2: Эффективность (Resource Waste)
    avg_waste = res_df['waste_mb'].mean()
    print(f"Average Memory Waste: {avg_waste:.2f} MB per query")

    # Метрика 3: Точность параметров (DOP)
    dop_acc = res_df['mt_dop_correct'].mean() * 100
    print(f"MT_DOP Prediction Accuracy: {dop_acc:.2f}%")

    # Метрика 4: Распределение по группам (стабильность)
    print(f"Active Groups: {res_df['group_id'].nunique()}")

    # Посмотрим на "Тяжелые" ошибки (где промахнулись более чем на 1 ГБ)
    huge_errors = res_df[res_df['error_mb'].abs() > 1024]
    print(f"Critical Mispredictions (>1GB): {len(huge_errors)} cases")

    return res_df


# Пример запуска:
if __name__ == '__main__':
    evaluator = evaluate_impala_model(ImpalaCascadeModel(max_groups=20), load_dataset(pg))
