import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score


def clean_to_float(data):
    # 1. Если это DataFrame или Series, берем значения
    if hasattr(data, 'values'):
        data = data.values

    # 2. Преобразуем в numpy array и выпрямляем в 1D (flatten)
    arr = np.asanyarray(data).ravel()

    clean_data = []
    for val in arr:
        try:
            # Если внутри еще один массив/список берем первый элемент
            while isinstance(val, (np.ndarray, list)):
                val = val[0]

            # Если это строка (например, из RF: '256mb')
            if isinstance(val, str):
                val = val.lower().replace('mb', '').replace('gb', '000')

            clean_data.append(float(val))
        except (ValueError, TypeError, IndexError):
            # Если совсем не число, ставим заглушку (потом заменим на 1.0)
            clean_data.append(1.0)

    return np.array(clean_data)


def plot_model_comparison(y_test, y_pred_cb, y_pred_rf):
    """
    Сравнивает распределение реальных данных и предсказаний двух моделей.
    """
    # Подготовка данных
    y_true = clean_to_float(y_test)
    y_pred_catboost = clean_to_float(y_pred_cb)
    y_pred_rf = clean_to_float(y_pred_rf)

    # log_scale требует строго положительных чисел
    y_true = np.maximum(y_true, 1.0)
    y_pred_catboost = np.maximum(y_pred_catboost, 1.0)
    y_pred_rf = np.maximum(y_pred_rf, 1.0)

    # --- ВИЗУАЛИЗАЦИЯ ---
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # KDE Plot с логарифмической шкалой
    try:
        sns.kdeplot(y_true, label='Реальные данные (Target)',
                    color='black', linewidth=3, fill=True, alpha=0.1, log_scale=False)

        sns.kdeplot(y_pred_catboost, label='Каскад CatBoost (Предсказание)',
                    color='blue', linewidth=2, linestyle='--', log_scale=False)

        sns.kdeplot(y_pred_rf, label='Ансамбль Random Forest (Предсказание)',
                    color='red', linewidth=2, linestyle=':', log_scale=False)
    except Exception as e:
        print(f"Предупреждение: KDE не удалось построить ({e}). Строим гистограммы.")
        plt.hist(y_true, bins=50, alpha=0.3, label='Target', color='black', density=True)
        plt.hist(y_pred_catboost, bins=50, alpha=0.3, label='CatBoost', color='blue', density=True)

    plt.title('Сравнение распределения Mem_Limit: CatBoost vs Random Forest', fontsize=15)
    plt.xlabel('Memory Limit (MB), Log Scale', fontsize=12)
    plt.ylabel('Плотность распределения', fontsize=12)
    plt.legend(fontsize=11)

    ax = plt.gca()
    plt.text(0.05, 0.95, 'CatBoost точнее аппроксимирует\n"тяжелый хвост" распределения',
             transform=ax.transAxes, fontsize=12, color='blue',
             verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

    print(f"\n" + "=" * 30)
    print(f"ИТОГОВЫЕ МЕТРИКИ (Тестовая выборка)")
    print(f"=" * 30)
    print(f"MAE CatBoost:      {mean_absolute_error(y_true, y_pred_catboost):.2f} MB")
    print(f"MAE Random Forest:  {mean_absolute_error(y_true, y_pred_rf):.2f} MB")
    print(f"R2 CatBoost:       {r2_score(y_true, y_pred_catboost):.4f}")
    print(f"R2 Random Forest:  {r2_score(y_true, y_pred_rf):.4f}")
    print(f"=" * 30)


y_truth = pd.read_csv('results_t.csv', sep=';')
print(y_truth.iloc[:, 0].max())
y_cb = pd.read_csv('results_cb.csv', sep=';')
print(y_cb.iloc[:, 0].max())
y_rf = pd.read_csv('results_rf.csv', sep=';')
print(y_rf.iloc[:, 0].max())

plot_model_comparison(y_truth, y_cb, y_rf)


def plot_smooth_trends(y_test, y_pred_cb, y_pred_rf):
    # 1. Подготовка и очистка
    t = clean_to_float(y_test)
    cb = clean_to_float(y_pred_cb)
    rf = clean_to_float(y_pred_rf)

    # 2. Собираем в DF и сортируем по истине
    df = pd.DataFrame({
        'truth': t,
        'cb': cb,
        'rf': rf
    }).sort_values(by='truth').reset_index(drop=True)

    # 3. Сглаживание
    window = max(1, int(len(df) * 0.03))
    df['cb_smooth'] = df['cb'].rolling(window=window, center=True).mean()
    df['rf_smooth'] = df['rf'].rolling(window=window, center=True).mean()

    # 4. Визуализация
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Черная линия - Истина
    plt.plot(df['truth'], color='black', linewidth=3, label='Истинная потребность (Target)', zorder=5)

    # Линия CatBoost (Плавный тренд)
    plt.plot(df['cb_smooth'], color='#1f77b4', linewidth=2.5, label='Тренд CatBoost (Наше решение)', zorder=4)

    # Линия Random Forest (Плавный тренд)
    plt.plot(df['rf_smooth'], color='#d62728', linewidth=2.5, label='Тренд Random Forest (Усреднение)', zorder=3)

    # Настройка осей
    # plt.yscale('log')  # Используем лог-шкалу, чтобы видеть и "хвост" и начало
    plt.title('Сравнение моделей: Способность следовать за вектором нагрузки', fontsize=16)
    plt.xlabel('Запросы (отсортированы по возрастанию сложности)', fontsize=12)
    plt.ylabel('Memory Limit (MB), Log Scale', fontsize=12)

    plt.axvspan(0, len(df) * 0.2, color='gray', alpha=0.1)  # Зона малых запросов
    plt.text(len(df) * 0.02, df['truth'].max() * 1.5, "Зона малых запросов:\nЗавышение (Запас прочности)", fontsize=10,
             color='gray')

    plt.legend(fontsize=12, loc='upper left')

    # Сетка
    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.tight_layout()
    plt.show()

    # Метрики
    print(f"R2 CatBoost: {r2_score(df['truth'], df['cb']):.4f}")
    print(f"R2 RF:       {r2_score(df['truth'], df['rf']):.4f}")


plot_smooth_trends(y_truth, y_cb, y_rf)


def clean_to_float_robust(data):
    """Превращает входные данные в чистый одномерный массив float, обрабатывая строки и ошибки."""
    if hasattr(data, 'values'):
        data = data.values
    arr = np.asanyarray(data).ravel()

    clean_data = []
    for val in arr:
        try:
            if isinstance(val, str):
                val = val.lower().replace('mb', '').replace('gb', '000')
            f_val = float(val)
            if not np.isfinite(f_val):
                f_val = np.nan
            clean_data.append(f_val)
        except:
            clean_data.append(np.nan)
    return np.array(clean_data)


def plot_absolute_and_residual_analysis(y_truth, y_cb, y_rf):
    """
    Строит два графика:
    1. Линейный график истины и предсказаний (для наглядности отклонений).
    2. График остатков (ошибок) на линейной шкале.
    """
    # 1. Очистка и подготовка данных
    y_true_clean = clean_to_float_robust(y_truth)
    y_cb_clean = clean_to_float_robust(y_cb)
    y_rf_clean = clean_to_float_robust(y_rf)

    df = pd.DataFrame({
        'True': y_true_clean,
        'CatBoost': y_cb_clean,
        'RandomForest': y_rf_clean
    })

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df > 0].dropna()

    # Сортируем по истинным значениям
    df = df.sort_values(by='True').reset_index(drop=True)

    # 2. Вычисление трендов для сглаживания (для основного графика)
    window_size = max(1, int(len(df) * 0.03))
    df['CB_Trend'] = df['CatBoost'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['RF_Trend'] = df['RandomForest'].rolling(window=window_size, center=True, min_periods=1).mean()

    # 3. Визуализация: График 1 - Линейные тренды
    fig, ax1 = plt.subplots(figsize=(16, 9))  # Основной график

    ax1.plot(df['True'], color='black', label='Истинная потребность (Target)', linewidth=4, zorder=5)
    ax1.plot(df['CB_Trend'], color='#1f77b4', linestyle='-', linewidth=2.5,
             label='CatBoost (Средний тренд)', zorder=4)
    ax1.plot(df['RF_Trend'], color='#d62728', linestyle='--', linewidth=2.5,
             label='Random Forest (Средний тренд)', zorder=3)

    ax1.fill_between(df.index, df['True'], df['RF_Trend'],
                     where=(df['RF_Trend'] < df['True']),
                     color='darkred', alpha=0.3, label='RF: Зона риска OOM')

    ax1.set_yscale('linear')  # Используем ЛИНЕЙНУЮ шкалу здесь
    ax1.set_title('Сравнение моделей: Линейный тренд фактического потребления', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Запросы, отсортированные по сложности', fontsize=14)
    ax1.set_ylabel('Memory Limit (MB), Линейная шкала', fontsize=14)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, which="both", ls="--", alpha=0.2)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 40)
    print("ИТОГОВЫЕ МЕТРИКИ (после фильтрации)")
    print("=" * 40)
    print(f"MAE CatBoost:      {mean_absolute_error(df['True'], df['CatBoost']):.2f} MB")
    print(f"MAE Random Forest:  {mean_absolute_error(df['True'], df['RandomForest']):.2f} MB")
    print(f"R2 CatBoost:       {r2_score(df['True'], df['CatBoost']):.4f}")
    print(f"R2 Random Forest:  {r2_score(df['True'], df['RandomForest']):.4f}")
    print("=" * 40)


plot_absolute_and_residual_analysis(y_truth, y_cb, y_rf)
