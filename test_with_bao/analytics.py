import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def run_analytical_report():
    # 1. Загрузка CSV в df
    try:
        df_base = pd.read_csv('benchmark_results_full_all_1.csv', sep=';')
        df_bao = pd.read_csv('benchmark_results_full_all_2.csv', sep=';')
        df_ml = pd.read_csv('benchmark_results_full_all_3.csv', sep=';')
    except Exception as e:
        print(f"Ошибка загрузки файлов: {e}")
        return

    def preprocess(df, name):
        # 2. Отбор RUN_TYPE = 'measured' and STATUS = 'success'
        df = df[(df['run_type'] == 'measured') & (df['status'] == 'success')].copy()

        if df.empty:
            return pd.DataFrame(columns=['query_id', 'duration_ms', 'latency', 'temp_written_blocks', 'system'])

        # 3. В каждом для каждого QUERY_ID отбираем лучший run по duration_ms
        df = df.sort_values('duration_ms').groupby('query_id').first().reset_index()

        # 4. Считаем latency = total_duration_ms - duration_ms
        # Пытаемся найти колонку с полным временем (она может называться total_latency_ms или total_duration_ms)
        total_col = 'total_duration_ms' if 'total_duration_ms' in df.columns else 'total_latency_ms'
        df['latency'] = df[total_col] - df['duration_ms']

        df['system'] = name
        return df[['query_id', 'duration_ms', 'latency', 'temp_written_blocks', 'system']]

    # Обрабатываем данные
    base = preprocess(df_base, 'Baseline')
    bao = preprocess(df_bao, 'Bao')
    ml = preprocess(df_ml, 'MyModule')

    # --- 5. Тест LATENCY Bao vs MyModule ---
    # Пересечение Baseline, Bao, MyModule по query_id
    common_all = set(base['query_id']) & set(bao['query_id']) & set(ml['query_id'])

    if common_all:
        lat_comp = pd.merge(base[base['query_id'].isin(common_all)][['query_id', 'latency']],
                            bao[bao['query_id'].isin(common_all)][['query_id', 'latency']],
                            on='query_id', suffixes=('_base', '_bao'))
        lat_comp = pd.merge(lat_comp,
                            ml[ml['query_id'].isin(common_all)][['query_id', 'latency']],
                            on='query_id')
        lat_comp.rename(columns={'latency': 'latency_ml'}, inplace=True)

        # Сравнение отклонения от baseline
        lat_comp['dev_bao'] = lat_comp['latency_bao'] - lat_comp['latency_base']
        lat_comp['dev_ml'] = lat_comp['latency_ml'] - lat_comp['latency_base']
    else:
        lat_comp = pd.DataFrame()

    # --- 7. Тест EFFICIENCY Bao vs Mymodule ---
    # Пересечение Baseline, Bao, MyModule по query_id
    if common_all:
        eff_comp = pd.merge(base[base['query_id'].isin(common_all)][['query_id', 'duration_ms']],
                            bao[bao['query_id'].isin(common_all)][['query_id', 'duration_ms']],
                            on='query_id', suffixes=('_base', '_bao'))
        eff_comp = pd.merge(eff_comp,
                            ml[ml['query_id'].isin(common_all)][['query_id', 'duration_ms']],
                            on='query_id')
        eff_comp.rename(columns={'duration_ms': 'duration_ms_ml'}, inplace=True)

        # Сравнение отклонения duration_ms от baseline
        print(len(eff_comp))
        eff_comp['dev_eff_bao'] = eff_comp['duration_ms_bao'] - eff_comp['duration_ms_base']
        eff_comp['dev_eff_ml'] = eff_comp['duration_ms_ml'] - eff_comp['duration_ms_base']
        eff_comp.to_csv("test3.csv", sep=';')
    else:
        eff_comp = pd.DataFrame()

    # --- 8. Тест SAFETY Bao vs Mymodule ---
    # Пересечение Bao, MyModule по query_id
    common_bao_ml = set(bao['query_id']) & set(ml['query_id'])
    spills_bao = bao[bao['query_id'].isin(common_bao_ml) & (bao['temp_written_blocks'] > 0)].shape[0]
    spills_ml = ml[ml['query_id'].isin(common_bao_ml) & (ml['temp_written_blocks'] > 0)].shape[0]

    # --- 9. Тест EFFICIENCY Baseline vs Mymodule ---
    # Пересечение Baseline, MyModule по query_id
    common_base_ml = set(base['query_id']) & set(ml['query_id'])
    eff_base_ml = pd.merge(base[base['query_id'].isin(common_base_ml)][['query_id', 'duration_ms']],
                           ml[ml['query_id'].isin(common_base_ml)][['query_id', 'duration_ms']],
                           on='query_id', suffixes=('_base', '_ml'))

    # --- 10. Тест ОТКАЗОУСТОЙЧИВОСТЬ ---
    # Сколько успешных Baseline отсутствуют в других
    base_ids = set(base['query_id'])
    absent_in_bao = base_ids - set(bao['query_id'])
    absent_in_ml = base_ids - set(ml['query_id'])

    # --- ВЫВОД ---
    print("\n" + "=" * 60)
    print("АНАЛИТИЧЕСКИЙ ОТЧЕТ")
    print("=" * 60)

    if not lat_comp.empty:
        print(f"\n[5] LATENCY TEST (N={len(common_all)}):")
        print(f"  Среднее откл. Latency Bao от Baseline: {lat_comp['dev_bao'].mean():.2f} ms")
        print(f"  Среднее откл. Latency MyModule от Baseline: {lat_comp['dev_ml'].mean():.2f} ms")

        print(f"\n[7] EFFICIENCY Bao vs MyModule (N={len(common_all)}):")
        print(f"  Среднее откл. Duration Bao от Baseline: {eff_comp['dev_eff_bao'].mean():.2f} ms")
        print(f"  Среднее откл. Duration MyModule от Baseline: {eff_comp['dev_eff_ml'].mean():.2f} ms")

    print(f"\n[8] SAFETY (Spills) (N={len(common_bao_ml)}):")
    print(f"  Запросов со спиллами у Bao: {spills_bao}")
    print(f"  Запросов со спиллами у MyModule: {spills_ml}")

    if not eff_base_ml.empty:
        print(f"\n[9] EFFICIENCY Baseline vs MyModule (N={len(common_base_ml)}):")
        print(f"  Средний Duration Baseline: {eff_base_ml['duration_ms_base'].mean():.2f} ms")
        print(f"  Средний Duration MyModule: {eff_base_ml['duration_ms_ml'].mean():.2f} ms")
        improv = (eff_base_ml['duration_ms_base'].mean() - eff_base_ml['duration_ms_ml'].mean()) / eff_base_ml[
            'duration_ms_base'].mean() * 100
        print(f"  Улучшение MyModule: {improv:.2f}%")

    print(f"\n[10] RELIABILITY (Baseline total success = {len(base_ids)}):")
    print(f"  Отсутствует в Bao: {len(absent_in_bao)} запросов")
    print(f"  Отсутствует в MyModule: {len(absent_in_ml)} запросов")

    # Визуализация
    if not eff_comp.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=['Bao Deviation', 'MyModule Deviation'],
                    y=[eff_comp['dev_eff_bao'].mean(), eff_comp['dev_eff_ml'].mean()], palette='coolwarm')
        plt.title('Efficiency: Average Duration Deviation from Baseline (Lower is better)')
        plt.ylabel('ms')
        plt.savefig('efficiency_report.png')
        plt.show()


if __name__ == "__main__":
    run_analytical_report()
