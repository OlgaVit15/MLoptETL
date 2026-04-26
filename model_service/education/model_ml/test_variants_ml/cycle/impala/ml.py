import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
# Предполагаем, что класс ImpalaMLModel находится в файле impala_ml.py
from model_ml.impala_ml import ImpalaMLModel

engine = create_engine('postgresql://suser:example@localhost:5432/model_tuning')


def parse_mem(val):
    if not val or str(val).lower() in ['0', 'none', 'nan']: return 0.0
    val = str(val).lower()
    match = re.search(r'(\d+)', val)
    if not match: return 0.0
    num = float(match.group(1))
    return num * 1024 if 'gb' in val or 'g' in val else num


def load_dataset():
    # Загружаем данные
    df = pd.read_sql("SELECT * FROM ml_training_dataset", engine)

    # Предварительная очистка: для оценки точности нам нужны записи с реальными метриками
    df['metric_pmu'] = pd.to_numeric(df['metric_pmu'], errors='coerce').fillna(0)

    # Инициализируем модель только для парсинга (вспомогательно)
    model_tool = ImpalaMLModel()
    df['target_mem_mb'] = df['target_mem_limit'].apply(parse_mem)
    df['target_mt_dop'] = pd.to_numeric(df['target_mt_dop'], errors='coerce').fillna(1)

    print(f"✅ Loaded {len(df)} rows.")
    return df


def visualize_results(df_with_clusters, model):
    """Визуализация качества кластеризации"""
    # 1. Готовим данные для PCA (берем только признаки, на которых учились)
    X_features = df_with_clusters[model.feature_cols].fillna(0)
    # X_scaled = model.scaler.transform(X_features)
    X_scaled = X_features.to_numpy()

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    plt.figure(figsize=(16, 6))

    # Левый график: Разброс кластеров
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1],
        hue=df_with_clusters['cluster'],
        style=df_with_clusters['is_success'],
        palette='tab20', alpha=0.7
    )
    plt.title("Cluster Map (PCA 2D)")

    # Правый график: Процент успеха по кластерам
    plt.subplot(1, 2, 2)
    # Собираем данные из профилей (фильтруем None)
    stats = []
    for cid, profile in model.cluster_profiles.items():
        if profile:
            stats.append({'cluster': cid, 'success_rate': profile['success_rate']})

    if stats:
        stats_df = pd.DataFrame(stats)
        sns.barplot(data=stats_df, x='cluster', y='success_rate', palette='coolwarm')
        plt.axhline(y=0.85, color='red', linestyle='--', label='Target 85%')
        plt.title("Success Rate by Cluster")

    plt.tight_layout()
    plt.show()


def plot_advanced_analysis(res_df):
    """Анализ точности предсказания памяти"""
    plt.figure(figsize=(16, 6))

    # ГРАФИК 1: Scatter Predicted vs Actual
    plt.subplot(1, 2, 1)
    # Цветовая схема: SAFE - зеленый, WARNING - желтый, CRITICAL - красный
    status_colors = {'SAFE': '#2ecc71', 'WARNING': '#f1c40f', 'CRITICAL': '#e74c3c'}

    sns.scatterplot(
        data=res_df, x='actual_pmu', y='pred_mem',
        hue='status', palette=status_colors, alpha=0.6
    )

    # Линии ориентиры
    max_val = max(res_df['actual_pmu'].max(), res_df['pred_mem'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Ideal (1:1)')
    plt.plot([0, max_val], [0, max_val * 1.3], 'g--', alpha=0.4, label='Safety Margin (+30%)')

    plt.xscale('log')
    plt.yscale('log')
    plt.title("Memory Prediction Accuracy (Log-Log Scale)")
    plt.xlabel("Actual Peak Memory (MB)")
    plt.ylabel("Predicted Memory Limit (MB)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)

    # ГРАФИК 2: Распределение Ratio
    plt.subplot(1, 2, 2)
    # Отрезаем экстремальные выбросы для визуализации
    plot_df = res_df[res_df['error_ratio'] < 10]
    sns.histplot(plot_df['error_ratio'], bins=40, kde=True, color='teal')
    plt.axvline(1.0, color='red', linestyle='--', label='OOM Border')
    plt.axvline(1.3, color='green', linestyle=':', label='Target Zone')
    plt.title("Error Ratio Distribution (Pred / Actual)")
    plt.xlabel("Ratio (1.0 = Perfect, >1.0 = Waste, <1.0 = OOM)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ТЕКСТОВАЯ ОЦЕНКА
    print("\n" + "=" * 40)
    print("📈 EXPERT ACCURACY REPORT")
    print("=" * 40)
    total = len(res_df)
    metrics = {
        "🔴 OOM Risk (Ratio < 1.0)": len(res_df[res_df['error_ratio'] < 1.0]),
        "🟢 Ideal (1.0 - 1.5x)": len(res_df[(res_df['error_ratio'] >= 1.0) & (res_df['error_ratio'] <= 1.5)]),
        "🟡 Over-allocated (1.5 - 3x)": len(res_df[(res_df['error_ratio'] > 1.5) & (res_df['error_ratio'] <= 3.0)]),
        "⚪ Waste (> 3x)": len(res_df[res_df['error_ratio'] > 3.0])
    }
    for k, v in metrics.items():
        print(f"{k.ljust(30)}: {v} queries ({v / total * 100:.1f}%)")
    print("=" * 40)


def full_ml_cycle(df_raw, n_clusters=45):
    # 1. Сплит данных
    train_df, test_df = train_test_split(df_raw, test_size=0.2, random_state=42, shuffle=True)

    # 2. Обучение
    model = ImpalaMLModel(n_clusters=n_clusters)
    # train_with_clusters теперь содержит все логарифмические колонки
    train_with_clusters = model.train(train_df)

    # 3. Визуализация кластеров
    visualize_results(train_with_clusters, model)

    # 4. Тестирование
    results = []
    print("🚀 Starting prediction on test_variants_ml set...")

    for _, row in test_df.iterrows():
        # Модель принимает словарь фичей плана
        pred = model.predict(row.to_dict())

        pred_mem = parse_mem(pred['target_mem_limit'])
        actual_pmu = parse_mem(row['target_mem_limit'])

        # Оцениваем только если есть данные о реальном запуске
        if actual_pmu > 0:
            error_ratio = pred_mem / actual_pmu
            results.append({
                'query_id': row['query_id'],
                'actual_pmu': actual_pmu,
                'pred_mem': pred_mem,
                'error_ratio': error_ratio,
                'status': pred.get('cluster_status', 'SAFE'),
                'heaviness': pred.get('heaviness_ratio', 1.0)
            })

    res_df = pd.DataFrame(results)

    # 5. Аналитика точности
    plot_advanced_analysis(res_df)

    return model, res_df


def run_elbow_method(df_raw, max_k=50):
    """
    Визуализация метода локтя для определения оптимального K.
    """
    print(f"🔎 Запуск метода локтя (K от 1 до {max_k})...")

    # 1. Готовим данные так же, как в модели
    model_tool = ImpalaMLModel()
    df_enriched = _prepare_features(df_raw)
    X_features = df_enriched[model_tool.feature_cols].fillna(0)
    X_scaled = model_tool.scaler.fit_transform(X_features)

    # 2. Применяем веса признаков (ВАЖНО: они должны совпадать с train)
    # Если в train вы умножаете log_total_scan_size на 3.0, здесь тоже нужно!
    idx_size = model_tool.feature_cols.index('log_total_scan_size')
    X_scaled[:, idx_size] *= 3.0

    distortions = []
    K = range(1, max_k + 1)

    for k in K:
        # n_init=10 достаточно для быстрой оценки
        kmeanModel = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeanModel.fit(X_scaled)
        # inertia_ - это сумма квадратов расстояний до центроидов (WCSS)
        distortions.append(kmeanModel.inertia_)
        if k % 5 == 0:
            print(f"Обработано k={k}")

    # 3. Строим график
    plt.figure(figsize=(12, 6))
    plt.plot(K, distortions, 'bx-', markersize=8, linewidth=2, color='#2c3e50')
    plt.xlabel('Количество кластеров (k)', fontsize=12)
    plt.ylabel('Инерция (Inertia / WCSS)', fontsize=12)
    plt.title('Метод локтя для определения оптимального K', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Пытаемся автоматически подсветить зону "локтя" (примерно)
    plt.axvspan(20, 45, color='green', alpha=0.1, label='Зона оптимального выбора')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    data = load_dataset()
    # run_elbow_method(data, 50)
    # Запускаем цикл с 40-50 кластерами для максимальной нарезки
    model, final_results = full_ml_cycle(data, n_clusters=10)
