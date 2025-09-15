#!/usr/bin/env python3
"""
Quick benchmark script to compare all libraries on 5M dataset
"""
import time
import pandas as pd
import polars as pl
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_pandas(file_path):
    """Benchmark pandas operations"""
    print("PANDAS BENCHMARK")
    start_mem = get_memory_usage()

    # Load
    start_time = time.time()
    df = pd.read_csv(file_path, header=None, names=['year_month', 'category1', 'category2', 'category3', 'code', 'flag', 'value1', 'value2'])
    load_time = time.time() - start_time

    # Clean
    start_time = time.time()
    df = df.fillna(0)
    clean_time = time.time() - start_time

    # Aggregate
    start_time = time.time()
    agg_result = df.groupby(['year_month', 'category1', 'category2']).agg({
        'value2': ['mean', 'median', 'max']
    })
    agg_time = time.time() - start_time

    # Sort
    start_time = time.time()
    df_sorted = df.sort_values('value2', ascending=False)
    sort_time = time.time() - start_time

    # Filter
    start_time = time.time()
    mean_val = df['value2'].mean()
    df_filtered = df[df['value2'] > mean_val]
    filter_time = time.time() - start_time

    end_mem = get_memory_usage()
    total_time = load_time + clean_time + agg_time + sort_time + filter_time

    print(f"  Load:     {load_time:.2f}s")
    print(f"  Clean:    {clean_time:.2f}s")
    print(f"  Agg:      {agg_time:.2f}s")
    print(f"  Sort:     {sort_time:.2f}s")
    print(f"  Filter:   {filter_time:.2f}s")
    print(f"  Total:    {total_time:.2f}s")
    print(f"  Memory:   {end_mem - start_mem:.1f} MB")
    print(f"  Rows:     {len(df):,}")
    print(f"  Filtered: {len(df_filtered):,}")
    print()

    return {
        'library': 'Pandas',
        'load_time': load_time,
        'clean_time': clean_time,
        'agg_time': agg_time,
        'sort_time': sort_time,
        'filter_time': filter_time,
        'total_time': total_time,
        'memory_mb': end_mem - start_mem,
        'rows': len(df),
        'filtered_rows': len(df_filtered)
    }

def benchmark_polars(file_path):
    """Benchmark polars operations"""
    print("POLARS BENCHMARK")
    start_mem = get_memory_usage()

    # Load
    start_time = time.time()
    df = pl.read_csv(file_path, has_header=False, new_columns=['year_month', 'category1', 'category2', 'category3', 'code', 'flag', 'value1', 'value2'])
    load_time = time.time() - start_time

    # Clean
    start_time = time.time()
    df = df.fill_null(0)
    clean_time = time.time() - start_time

    # Aggregate
    start_time = time.time()
    agg_result = df.group_by(['year_month', 'category1', 'category2']).agg([
        pl.col('value2').mean().alias('mean_value2'),
        pl.col('value2').median().alias('median_value2'),
        pl.col('value2').max().alias('max_value2')
    ])
    agg_time = time.time() - start_time

    # Sort
    start_time = time.time()
    df_sorted = df.sort('value2', descending=True)
    sort_time = time.time() - start_time

    # Filter
    start_time = time.time()
    mean_val = df['value2'].mean()
    df_filtered = df.filter(pl.col('value2') > mean_val)
    filter_time = time.time() - start_time

    end_mem = get_memory_usage()
    total_time = load_time + clean_time + agg_time + sort_time + filter_time

    print(f"  Load:     {load_time:.2f}s")
    print(f"  Clean:    {clean_time:.2f}s")
    print(f"  Agg:      {agg_time:.2f}s")
    print(f"  Sort:     {sort_time:.2f}s")
    print(f"  Filter:   {filter_time:.2f}s")
    print(f"  Total:    {total_time:.2f}s")
    print(f"  Memory:   {end_mem - start_mem:.1f} MB")
    print(f"  Rows:     {len(df):,}")
    print(f"  Filtered: {len(df_filtered):,}")
    print()

    return {
        'library': 'Polars',
        'load_time': load_time,
        'clean_time': clean_time,
        'agg_time': agg_time,
        'sort_time': sort_time,
        'filter_time': filter_time,
        'total_time': total_time,
        'memory_mb': end_mem - start_mem,
        'rows': len(df),
        'filtered_rows': len(df_filtered)
    }

def main():
    file_path = "data/benchmark_5m.csv"

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return

    print("=" * 60)
    print("QUICK 5M DATASET BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"File: {file_path}")
    print(f"Size: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")
    print()

    results = []

    try:
        results.append(benchmark_pandas(file_path))
    except Exception as e:
        print(f"Pandas failed: {e}\n")

    try:
        results.append(benchmark_polars(file_path))
    except Exception as e:
        print(f"Polars failed: {e}\n")

    # Summary
    if results:
        print("=" * 60)
        print("SUMMARY COMPARISON")
        print("=" * 60)

        fastest_total = min(results, key=lambda x: x['total_time'])
        fastest_load = min(results, key=lambda x: x['load_time'])
        lowest_memory = min(results, key=lambda x: x['memory_mb'])

        for r in results:
            print(f"{r['library']:>8}: {r['total_time']:6.2f}s total | {r['load_time']:5.2f}s load | {r['memory_mb']:6.1f} MB")

        print()
        print(f"Fastest Total:  {fastest_total['library']} ({fastest_total['total_time']:.2f}s)")
        print(f"Fastest Load:   {fastest_load['library']} ({fastest_load['load_time']:.2f}s)")
        print(f"Lowest Memory:  {lowest_memory['library']} ({lowest_memory['memory_mb']:.1f} MB)")

if __name__ == "__main__":
    main()