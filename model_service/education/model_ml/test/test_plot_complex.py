import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix
from scipy.stats import sem, t
import warnings

# Настройка четкости графиков
plt.rcParams['figure.dpi'] = 100
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")


def clean_to_float_robust(data):
    if hasattr(data, 'values'): data = data.values
    arr = np.asanyarray(data).ravel()
    clean_data = []
    for val in arr:
        try:
            if isinstance(val, str):
                val = val.lower().replace('mb', '').replace('gb', '000')
                if '000' in val: val = float(val) / 1000 * 1024
            f_val = float(val)
            clean_data.append(f_val if np.isfinite(f_val) else np.nan)
        except:
            clean_data.append(np.nan)
    return np.array(clean_data)


def calculate_confidence_interval_mean(data, confidence=0.95):
    n = len(data)
    if n < 2: return np.nan, np.nan, np.nan
    m = np.mean(data)
    h = sem(data) * t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def plot_full_cascade_evaluation(
        y_true_base_mem, y_pred_cb_base_mem, y_pred_rf_base_mem, y_pred_xgb_base_mem, y_pred_lgb_base_mem,
        y_true_dop, y_pred_cb_dop, y_pred_rf_dop, y_pred_xgb_dop, y_pred_lgb_dop,
        y_true_join_mode, y_pred_cb_join_mode, y_pred_rf_join_mode, y_pred_xgb_join_mode, y_pred_lgb_join_mode,
        y_true_codegen, y_pred_cb_codegen, y_pred_rf_codegen, y_pred_xgb_codegen, y_pred_lgb_codegen,
        y_true_final_mem, y_pred_cb_final_mem, y_pred_rf_final_mem, y_pred_xgb_final_mem,y_pred_lgb_final_mem
):
    # --- 0. Синхронизация и очистка ---
    df_raw = pd.DataFrame({
        'True_BaseMem': clean_to_float_robust(y_true_base_mem),
        'Pred_CB_BaseMem': clean_to_float_robust(y_pred_cb_base_mem),
        'Pred_RF_BaseMem': clean_to_float_robust(y_pred_rf_base_mem),
        'Pred_XGB_BaseMem': clean_to_float_robust(y_pred_xgb_base_mem),
        'Pred_LGBM_BaseMem': clean_to_float_robust(y_pred_lgb_base_mem),
        'True_DOP': y_true_dop, 'Pred_CB_DOP': y_pred_cb_dop, 'Pred_RF_DOP': y_pred_rf_dop,
        'Pred_XGB_DOP': y_pred_xgb_dop,'Pred_LGBM_DOP': y_pred_lgb_dop,
        'True_JoinMode': y_true_join_mode, 'Pred_CB_JoinMode': y_pred_cb_join_mode,
        'Pred_RF_JoinMode': y_pred_rf_join_mode, 'Pred_XGB_JoinMode': y_pred_xgb_join_mode,'Pred_LGBM_JoinMode': y_pred_lgb_join_mode,
        'True_Codegen': y_true_codegen, 'Pred_CB_Codegen': y_pred_cb_codegen, 'Pred_RF_Codegen': y_pred_rf_codegen,
        'Pred_XGB_Codegen': y_pred_xgb_codegen,'Pred_LGBM_Codegen': y_pred_lgb_codegen,
        'True_FinalMem': clean_to_float_robust(y_true_final_mem),
        'Pred_CB_FinalMem': clean_to_float_robust(y_pred_cb_final_mem),
        'Pred_RF_FinalMem': clean_to_float_robust(y_pred_rf_final_mem),
        'Pred_XGB_FinalMem': clean_to_float_robust(y_pred_xgb_final_mem),
        'Pred_LGBM_FinalMem': clean_to_float_robust(y_pred_lgb_final_mem)
    })
    df_raw = df_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=['True_BaseMem', 'True_FinalMem'])
    df_raw = df_raw[(df_raw['True_BaseMem'] > 0) & (df_raw['True_FinalMem'] > 0)]

    # --- 1. График: Регрессор базовой памяти ---
    df_base_sorted = df_raw.sort_values(by='True_BaseMem').reset_index(drop=True)
    window = max(5, int(len(df_base_sorted) * 0.04))
    for m in ['CB', 'RF', 'XGB', 'LGBM']:
        df_base_sorted[f'{m}_Trend'] = df_base_sorted[f'Pred_{m}_BaseMem'].rolling(window=window, center=True,
                                                                                   min_periods=1).mean()

    plt.figure(figsize=(16, 8))
    sns.set_style("whitegrid")
    plt.plot(df_base_sorted['True_BaseMem'], color='black', label='Истинная базовая память', linewidth=4, zorder=5)
    plt.plot(df_base_sorted['CB_Trend'], color='#1f77b4', linewidth=2.5, label='CatBoost Trend')
    plt.plot(df_base_sorted['RF_Trend'], color='#d62728', linewidth=2.5, linestyle='--', label='Random Forest Trend')
    plt.plot(df_base_sorted['XGB_Trend'], color='#2ca02c', linewidth=2.5, linestyle=':', label='XGBoost Trend')
    plt.plot(df_base_sorted['LGBM_Trend'], color='#ffff00', linewidth=2.5, linestyle=':', label='LGBM Trend')
    plt.title('Оценка базового регрессора (DOP=0)', fontsize=16, fontweight='bold')
    plt.ylabel('Memory (MB)'), plt.legend(loc='upper left')
    plt.tight_layout(), plt.show()

    # --- 2. Оценка Классификаторов ---
    print("\n" + "=" * 80 + "\nОЦЕНКА КЛАССИФИКАТОРОВ СТРАТЕГИЙ\n" + "=" * 80)
    classification_targets = {
        'MT_DOP': ('True_DOP', 'Pred_CB_DOP', 'Pred_RF_DOP', 'Pred_XGB_DOP', 'Pred_LGBM_DOP'),
        'Join_Mode': ('True_JoinMode', 'Pred_CB_JoinMode', 'Pred_RF_JoinMode', 'Pred_XGB_JoinMode', 'Pred_LGBM_JoinMode'),
        'Codegen': ('True_Codegen', 'Pred_CB_Codegen', 'Pred_RF_Codegen', 'Pred_XGB_Codegen', 'Pred_LGBM_Codegen')
    }

    for target_name, cols in classification_targets.items():
        true_c, p_cb, p_rf, p_xgb, p_lgbm = cols

        # Вывод отчетов в консоль
        for name, col in [("CatBoost", p_cb), ("Random Forest", p_rf), ("XGBoost", p_xgb), ("LGBM", p_xgb)]:
            print(f"\n--- {name}: {target_name} ---")
            print(classification_report(df_raw[true_c], df_raw[col], zero_division=0))

        # Визуализация матриц 1x3
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        cms = [
            (confusion_matrix(df_raw[true_c], df_raw[p_cb]), 'CatBoost', 'Blues'),
            (confusion_matrix(df_raw[true_c], df_raw[p_rf]), 'Random Forest', 'Reds'),
            (confusion_matrix(df_raw[true_c], df_raw[p_xgb]), 'XGBoost', 'Greens'),
            (confusion_matrix(df_raw[true_c], df_raw[p_lgbm]), 'LGBM', 'Greys')
        ]
        for ax, (cm, m_name, cmap) in zip(axes, cms):
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False)
            ax.set_title(f'{m_name}: {target_name}', fontweight='bold')
            ax.set_xlabel('Predicted'), ax.set_ylabel('True')
        plt.tight_layout(), plt.show()

    # --- 3. График: Финальный каскад памяти ---
    df_final_sorted = df_raw.sort_values(by='True_FinalMem').reset_index(drop=True)
    for m in ['CB', 'RF', 'XGB', 'LGBM']:
        df_final_sorted[f'{m}_Trend'] = df_final_sorted[f'Pred_{m}_FinalMem'].rolling(window=window, center=True,
                                                                                      min_periods=1).mean()

    plt.figure(figsize=(16, 8))
    plt.plot(df_final_sorted['True_FinalMem'], color='black', label='Истинный Memory Limit', linewidth=4, zorder=5)
    plt.plot(df_final_sorted['CB_Trend'], color='#1f77b4', linewidth=2.5, label='CatBoost (Final)')
    plt.plot(df_final_sorted['RF_Trend'], color='#d62728', linewidth=2.5, linestyle='--', label='Random Forest (Final)')
    plt.plot(df_final_sorted['XGB_Trend'], color='#2ca02c', linewidth=2.5, linestyle=':', label='XGBoost (Final)')
    plt.plot(df_final_sorted['LGBM_Trend'], color='#ffff00', linewidth=2.5, linestyle=':', label='LGM (Final)')

    # Штриховка дефицита для RF
    plt.fill_between(df_final_sorted.index, df_final_sorted['True_FinalMem'], df_final_sorted['RF_Trend'],
                     where=(df_final_sorted['RF_Trend'] < df_final_sorted['True_FinalMem']),
                     color='darkred', alpha=0.2, label='Дефицит памяти (RF)')

    plt.title('Оценка итогового каскада: финальный Memory Limit', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left'), plt.tight_layout(), plt.show()

    # Гистограмма остатков
    plt.figure(figsize=(12, 6))
    for m, c, name in [('CB', '#1f77b4', 'CatBoost'), ('RF', '#d62728', 'Random Forest'),
                       ('XGB', '#2ca02c', 'XGBoost'), ('LGBM', '#ffff00', 'LGBM')]:
        sns.kdeplot(df_raw[f'Pred_{m}_FinalMem'] - df_raw['True_FinalMem'], color=c, label=name, fill=True, alpha=0.2)
    plt.axvline(0, color='black', linestyle='--'), plt.title(
        'Распределение остатков (Ошибка предсказания)'), plt.legend()
    plt.tight_layout(), plt.show()

    # --- 4. Финальные метрики в консоль ---
    print("\n" + "=" * 80 + "\nИТОГОВЫЕ МЕТРИКИ РЕГРЕССИИ\n" + "=" * 80)
    for stage, df_sort, t_col, p_pref in [("Базовый", df_base_sorted, 'True_BaseMem', 'Pred'),
                                          ("Финальный", df_final_sorted, 'True_FinalMem', 'Pred')]:
        print(f"\n--- {stage} регрессор ---")
        for m in ['CB', 'RF', 'XGB', 'LGBM']:
            y_t, y_p = df_sort[t_col], df_sort[f'{p_pref}_{m}_{t_col.split("_")[1]}']
            print(f"{m} -> R2: {r2_score(y_t, y_p):.4f}, MAE: {mean_absolute_error(y_t, y_p):.2f} MB")

    print("\n--- 95% Доверительные интервалы (MAE Финальный каскад) ---")
    for m in ['CB', 'RF', 'XGB', 'LGBM']:
        errs = np.abs(df_raw['True_FinalMem'] - df_raw[f'Pred_{m}_FinalMem'])
        m_val, low, upp = calculate_confidence_interval_mean(errs)
        print(f"{m} MAE: {m_val:.2f} MB (CI: [{low:.2f}, {upp:.2f}])")
    print("=" * 80)


df_rf = pd.read_csv('results_rf.csv', sep=';')
df_cb = pd.read_csv('results_cb.csv', sep=';')
df_xgb = pd.read_csv('results_xgb.csv', sep=';')
df_lgb = pd.read_csv('results_lgb.csv', sep=';')
df_t = pd.read_csv('results_t.csv', sep=';')
y_true_base_mem = df_t['base_pmu']
y_pred_cb_base_mem = df_cb['base_pmu']
y_pred_rf_base_mem = df_rf['base_pmu']
y_pred_xgb_base_mem = df_xgb['base_pmu']
y_pred_lgb_base_mem = df_lgb['base_pmu']
y_true_dop = df_t['mt_dop']
y_pred_cb_dop = df_cb['mt_dop']
y_pred_rf_dop = df_rf['mt_dop']
y_pred_xgb_dop = df_xgb['mt_dop']
y_pred_lgb_dop = df_lgb['mt_dop']
y_true_join_mode = df_t['join_mode']
y_pred_cb_join_mode = df_cb['join_mode']
y_pred_rf_join_mode = df_rf['join_mode']
y_pred_xgb_join_mode = df_xgb['join_mode']
y_pred_lgb_join_mode = df_lgb['join_mode']
y_true_codegen = df_t['codegen']
y_pred_cb_codegen = df_cb['codegen']
y_pred_rf_codegen = df_rf['codegen']
y_pred_xgb_codegen = df_xgb['codegen']
y_pred_lgb_codegen = df_lgb['codegen']
y_true_final_mem = df_t['mem_limit']
y_pred_cb_final_mem = df_cb['pred_mem_limit']
y_pred_rf_final_mem = df_rf['pred_mem_limit']
y_pred_xgb_final_mem = df_xgb['pred_mem_limit']
y_pred_lgb_final_mem = df_lgb['pred_mem_limit']
plot_full_cascade_evaluation(
    y_true_base_mem, y_pred_cb_base_mem, y_pred_rf_base_mem, y_pred_xgb_base_mem, y_pred_lgb_base_mem,
    y_true_dop, y_pred_cb_dop, y_pred_rf_dop, y_pred_xgb_dop, y_pred_lgb_dop,
    y_true_join_mode, y_pred_cb_join_mode, y_pred_rf_join_mode, y_pred_xgb_join_mode, y_pred_lgb_join_mode,
    y_true_codegen, y_pred_cb_codegen, y_pred_rf_codegen, y_pred_xgb_codegen, y_pred_lgb_codegen,
    y_true_final_mem, y_pred_cb_final_mem, y_pred_rf_final_mem, y_pred_xgb_final_mem, y_pred_lgb_final_mem
)
