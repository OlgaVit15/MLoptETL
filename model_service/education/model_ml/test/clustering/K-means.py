import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _create_strategy_label(df: pd.DataFrame) -> pd.Series:
    """Создает метку стратегии"""
    return (
            "DOP" + df['target_mt_dop'].astype(int).astype(str) + "_" +
            "TH" + df['target_num_scanner_threads'].astype(int).astype(str) + "_" +
            "JM" + df['target_default_join_distribution_mode'].astype(str).str.upper() + "_" +
            "CG" + df['target_disable_codegen'].astype(str).str.upper()
    )


def load_and_clean_data() -> pd.DataFrame:
    logging.info(f"Загрузка данных...")
    path = "D:/ml_training_dataset.csv"
    df = pd.read_csv(path, sep=";")

    # Типизация
    numeric_cols = [
        'metric_pmu', 'target_mem_limit_dop0',
        'feature_plan_num_joins', 'feature_plan_num_scan_nodes',
        'feature_plan_num_agg_nodes', 'feature_plan_total_scan_size_bytes',
        'feature_plan_max_cardinality', 'feature_plan_max_row_size',
        'target_mt_dop', 'target_num_scanner_threads'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Очистка строк и создание таргетов
    cat_cols = ['target_default_join_distribution_mode', 'target_disable_codegen']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    df['strategy_label'] = _create_strategy_label(df)
    df['base_pmu'] = df['target_mem_limit_dop0']  # Цель для регрессора
    df.drop(columns='target_mem_limit_dop0')

    # Фильтрация выбросов (ваши принципы)
    df = df[df['metric_pmu'] > 0]
    q_high = df['base_pmu'].quantile(0.99)
    df = df[df['base_pmu'] < q_high]

    # Перемешивание
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# 1. ПОДГОТОВКА ПРИЗНАКОВ (на основе вашей логики классификатора)
def extract_features_for_viz(df):
    results = []
    for _, row in df.iterrows():
        # Используем вашу логику признаков классификатора
        log_base_pmu = np.log1p(row['base_pmu'])
        scans = row['feature_plan_num_scan_nodes']
        joins = row['feature_plan_num_joins']
        aggs = row['feature_plan_num_agg_nodes']
        vol_theory = row['feature_plan_max_cardinality'] * row['feature_plan_max_row_size']
        log_scan_size = np.log1p(row['feature_plan_total_scan_size_bytes'])
        nodes = row['feature_plan_num_scan_nodes']
        pmu_per_node = row['base_pmu'] / nodes if nodes > 0 else 0
        row_size = row['feature_plan_max_row_size']

        results.append([
            log_base_pmu, scans, joins, aggs, vol_theory,
            log_scan_size, nodes, pmu_per_node, row_size
        ])

    cols = ['log_pmu', 'scans', 'joins', 'aggs', 'vol', 'log_scan', 'nodes', 'pmu_node', 'row']
    res_df = pd.DataFrame(results, columns=cols)
    return res_df.replace([np.inf, -np.inf], 0).fillna(0)


# --- ЗАГРУЗКА ---
# Здесь предполагается, что df уже загружен вашей функцией load_and_clean_data()
df = load_and_clean_data()

# Подготовка матрицы признаков
X_raw = extract_features_for_viz(df)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 2. ОБУЧЕНИЕ K-MEANS
# Возьмем количество кластеров равное количеству уникальных стратегий
n_strats = df['strategy_label'].nunique()
kmeans = KMeans(n_clusters=n_strats, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# 3. PCA ДЛЯ ВИЗУАЛИЗАЦИИ
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PCA1 (Масштаб запроса)', 'PCA2 (Сложность структуры)'],
    index=X_raw.columns
)

plt.figure(figsize=(12, 6))
sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0, fmt=".2f", cbar=True)
plt.title("Влияние физических признаков плана на компоненты PCA1 и PCA2", fontsize=14, fontweight='bold')
plt.ylabel("Исходные признаки плана")
plt.tight_layout()
plt.show()

# 4. ПОСТРОЕНИЕ ГРАФИКОВ
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
sns.set_style("white")

# График А: K-Means
sns.scatterplot(
    data=df, x='pca1', y='pca2', hue='kmeans_cluster',
    palette='viridis', ax=axes[0], s=50, alpha=0.6, edgecolor='none'
)
axes[0].set_title(f"Кластеризация K-Means (k={n_strats})", fontsize=14,
                  fontweight='bold')

# График Б: Реальные стратегии
sns.scatterplot(
    data=df, x='pca1', y='pca2', hue='strategy_label',
    palette='tab10', ax=axes[1], s=50, alpha=0.6, edgecolor='none'
)
axes[1].set_title("Реальные стратегии", fontsize=14, fontweight='bold')

plt.suptitle("Сравнение K-Means и фактических стратегий выполнения", fontsize=16, y=1.02)
plt.show()

# 5. АНАЛИЗ ПРИМЕСЕЙ (IMPURITY)
print("АНАЛИЗ ОШИБКИ КЛАСТЕРИЗАЦИИ:")
for cluster in sorted(df['kmeans_cluster'].unique()):
    subset = df[df['kmeans_cluster'] == cluster]
    top_strat = subset['strategy_label'].value_counts(normalize=True).max()
    print(f"Кластер {cluster}: точность попадания в одну стратегию = {top_strat:.2%}")
    if top_strat < 0.7:
        print(f"  -> ВНИМАНИЕ: Кластер сильно смешан!")
