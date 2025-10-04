"""
Benchmark 1M dataset with warmup runs for all libraries.
Compares cold start (no warmup) vs warmed up performance.
"""

import time
import json
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.compute as pc
import dask.dataframe as dd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, 'benchmark_1m.csv')
WARMUP_RUNS = 3  # Number of warmup iterations


def benchmark_pandas_with_warmup():
    """Benchmark Pandas with warmup runs."""
    print("\n=== Pandas Benchmark (with warmup) ===")

    # Warmup phase
    print(f"Running {WARMUP_RUNS} warmup iterations...")
    for i in range(WARMUP_RUNS):
        print(f"  Warmup {i+1}/{WARMUP_RUNS}...")
        df = pd.read_csv(DATA_FILE)
        df = df.fillna(0)
        grouped = df.groupby(['year_month', 'category1', 'category2'])['value2'].agg(['mean', 'median', 'max'])
        sorted_df = df.sort_values('value2', ascending=False)
        mean_val = df['value2'].mean()
        filtered = df[df['value2'] > mean_val]
        corr = df.select_dtypes(include=['number']).corr()

    # Actual measurement
    print("Running measured iteration...")
    metrics = {}

    start = time.time()
    df = pd.read_csv(DATA_FILE)
    metrics['loading_time'] = time.time() - start

    start = time.time()
    df = df.fillna(0)
    metrics['cleaning_time'] = time.time() - start

    start = time.time()
    grouped = df.groupby(['year_month', 'category1', 'category2'])['value2'].agg(['mean', 'median', 'max'])
    metrics['aggregation_time'] = time.time() - start

    start = time.time()
    sorted_df = df.sort_values('value2', ascending=False)
    metrics['sorting_time'] = time.time() - start

    start = time.time()
    mean_val = df['value2'].mean()
    filtered = df[df['value2'] > mean_val]
    metrics['filtering_time'] = time.time() - start

    start = time.time()
    corr = df.select_dtypes(include=['number']).corr()
    metrics['correlation_time'] = time.time() - start

    metrics['total_time'] = sum([v for k, v in metrics.items() if k != 'total_time'])
    metrics['row_count'] = len(df)
    metrics['filtered_count'] = len(filtered)

    print(f"Total time (warmed): {metrics['total_time']:.3f}s")
    return metrics


def benchmark_polars_with_warmup():
    """Benchmark Polars with warmup runs."""
    print("\n=== Polars Benchmark (with warmup) ===")

    # Warmup phase
    print(f"Running {WARMUP_RUNS} warmup iterations...")
    for i in range(WARMUP_RUNS):
        print(f"  Warmup {i+1}/{WARMUP_RUNS}...")
        df = pl.read_csv(DATA_FILE)
        df = df.fill_null(0)
        grouped = df.group_by(['year_month', 'category1', 'category2']).agg([
            pl.col('value2').mean().alias('mean'),
            pl.col('value2').median().alias('median'),
            pl.col('value2').max().alias('max')
        ])
        sorted_df = df.sort('value2', descending=True)
        mean_val = df['value2'].mean()
        filtered = df.filter(pl.col('value2') > mean_val)
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]
        corr = df.select(numeric_cols).corr()

    # Actual measurement
    print("Running measured iteration...")
    metrics = {}

    start = time.time()
    df = pl.read_csv(DATA_FILE)
    metrics['loading_time'] = time.time() - start

    start = time.time()
    df = df.fill_null(0)
    metrics['cleaning_time'] = time.time() - start

    start = time.time()
    grouped = df.group_by(['year_month', 'category1', 'category2']).agg([
        pl.col('value2').mean().alias('mean'),
        pl.col('value2').median().alias('median'),
        pl.col('value2').max().alias('max')
    ])
    metrics['aggregation_time'] = time.time() - start

    start = time.time()
    sorted_df = df.sort('value2', descending=True)
    metrics['sorting_time'] = time.time() - start

    start = time.time()
    mean_val = df['value2'].mean()
    filtered = df.filter(pl.col('value2') > mean_val)
    metrics['filtering_time'] = time.time() - start

    start = time.time()
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]
    corr = df.select(numeric_cols).corr()
    metrics['correlation_time'] = time.time() - start

    metrics['total_time'] = sum([v for k, v in metrics.items() if k != 'total_time'])
    metrics['row_count'] = len(df)
    metrics['filtered_count'] = len(filtered)

    print(f"Total time (warmed): {metrics['total_time']:.3f}s")
    return metrics


def benchmark_pyarrow_with_warmup():
    """Benchmark PyArrow with warmup runs."""
    print("\n=== PyArrow Benchmark (with warmup) ===")

    # Warmup phase
    print(f"Running {WARMUP_RUNS} warmup iterations...")
    for i in range(WARMUP_RUNS):
        print(f"  Warmup {i+1}/{WARMUP_RUNS}...")
        table = csv.read_csv(DATA_FILE)
        df = table.to_pandas()
        df = df.fillna(0)
        table = pa.Table.from_pandas(df)
        grouped = df.groupby(['year_month', 'category1', 'category2'])['value2'].agg(['mean', 'median', 'max'])
        indices = pc.sort_indices(table, sort_keys=[('value2', 'descending')])
        sorted_table = pc.take(table, indices)
        mean_val = pc.mean(table['value2']).as_py()
        mask = pc.greater(table['value2'], mean_val)
        filtered = table.filter(mask)

    # Actual measurement
    print("Running measured iteration...")
    metrics = {}

    start = time.time()
    table = csv.read_csv(DATA_FILE)
    metrics['loading_time'] = time.time() - start

    start = time.time()
    df = table.to_pandas()
    df = df.fillna(0)
    table = pa.Table.from_pandas(df)
    metrics['cleaning_time'] = time.time() - start

    start = time.time()
    grouped = df.groupby(['year_month', 'category1', 'category2'])['value2'].agg(['mean', 'median', 'max'])
    metrics['aggregation_time'] = time.time() - start

    start = time.time()
    indices = pc.sort_indices(table, sort_keys=[('value2', 'descending')])
    sorted_table = pc.take(table, indices)
    metrics['sorting_time'] = time.time() - start

    start = time.time()
    mean_val = pc.mean(table['value2']).as_py()
    mask = pc.greater(table['value2'], mean_val)
    filtered = table.filter(mask)
    metrics['filtering_time'] = time.time() - start

    start = time.time()
    metrics['correlation_time'] = time.time() - start

    metrics['total_time'] = sum([v for k, v in metrics.items() if k != 'total_time'])
    metrics['row_count'] = len(table)
    metrics['filtered_count'] = len(filtered)

    print(f"Total time (warmed): {metrics['total_time']:.3f}s")
    return metrics


def benchmark_dask_with_warmup():
    """Benchmark Dask with warmup runs."""
    print("\n=== Dask Benchmark (with warmup) ===")

    # Warmup phase
    print(f"Running {WARMUP_RUNS} warmup iterations...")
    for i in range(WARMUP_RUNS):
        print(f"  Warmup {i+1}/{WARMUP_RUNS}...")
        df = dd.read_csv(DATA_FILE)
        df = df.fillna(0)
        grouped = df.groupby(['year_month', 'category1', 'category2'])['value2'].agg(['mean', 'median', 'max']).compute()
        sorted_df = df.sort_values('value2', ascending=False).compute()
        mean_val = df['value2'].mean().compute()
        filtered = df[df['value2'] > mean_val].compute()

    # Actual measurement
    print("Running measured iteration...")
    metrics = {}

    start = time.time()
    df = dd.read_csv(DATA_FILE)
    metrics['loading_time'] = time.time() - start

    start = time.time()
    df = df.fillna(0)
    metrics['cleaning_time'] = time.time() - start

    start = time.time()
    grouped = df.groupby(['year_month', 'category1', 'category2'])['value2'].agg(['mean', 'median', 'max']).compute()
    metrics['aggregation_time'] = time.time() - start

    start = time.time()
    sorted_df = df.sort_values('value2', ascending=False).compute()
    metrics['sorting_time'] = time.time() - start

    start = time.time()
    mean_val = df['value2'].mean().compute()
    filtered = df[df['value2'] > mean_val].compute()
    metrics['filtering_time'] = time.time() - start

    start = time.time()
    metrics['correlation_time'] = time.time() - start

    metrics['total_time'] = sum([v for k, v in metrics.items() if k != 'total_time'])
    metrics['row_count'] = len(df)
    metrics['filtered_count'] = len(filtered)

    print(f"Total time (warmed): {metrics['total_time']:.3f}s")
    return metrics


def benchmark_pyspark_with_warmup():
    """Benchmark PySpark with warmup runs."""
    print("\n=== PySpark Benchmark (with warmup) ===")

    spark = SparkSession.builder \
        .appName("Benchmark_1M_Warmup") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Warmup phase
    print(f"Running {WARMUP_RUNS} warmup iterations...")
    for i in range(WARMUP_RUNS):
        print(f"  Warmup {i+1}/{WARMUP_RUNS}...")
        df = spark.read.csv(DATA_FILE, header=True, inferSchema=True)
        df = df.fillna(0)
        grouped = df.groupBy('year_month', 'category1', 'category2').agg(
            F.mean('value2').alias('mean'),
            F.expr('percentile_approx(value2, 0.5)').alias('median'),
            F.max('value2').alias('max')
        ).collect()
        sorted_df = df.orderBy(F.col('value2').desc()).collect()
        mean_val = df.agg(F.mean('value2')).collect()[0][0]
        filtered = df.filter(F.col('value2') > mean_val).collect()

    # Actual measurement
    print("Running measured iteration...")
    metrics = {}

    start = time.time()
    df = spark.read.csv(DATA_FILE, header=True, inferSchema=True)
    df.cache()
    df.count()
    metrics['loading_time'] = time.time() - start

    start = time.time()
    df = df.fillna(0)
    metrics['cleaning_time'] = time.time() - start

    start = time.time()
    grouped = df.groupBy('year_month', 'category1', 'category2').agg(
        F.mean('value2').alias('mean'),
        F.expr('percentile_approx(value2, 0.5)').alias('median'),
        F.max('value2').alias('max')
    ).collect()
    metrics['aggregation_time'] = time.time() - start

    start = time.time()
    sorted_df = df.orderBy(F.col('value2').desc()).collect()
    metrics['sorting_time'] = time.time() - start

    start = time.time()
    mean_val = df.agg(F.mean('value2')).collect()[0][0]
    filtered = df.filter(F.col('value2') > mean_val)
    filtered_count = filtered.count()
    metrics['filtering_time'] = time.time() - start

    start = time.time()
    metrics['correlation_time'] = time.time() - start

    metrics['total_time'] = sum([v for k, v in metrics.items() if k != 'total_time'])
    metrics['row_count'] = df.count()
    metrics['filtered_count'] = filtered_count

    spark.stop()

    print(f"Total time (warmed): {metrics['total_time']:.3f}s")
    return metrics


def main():
    """Run all benchmarks with warmup."""
    print("=" * 60)
    print("1M Dataset Benchmark WITH WARMUP")
    print("=" * 60)
    print(f"Data file: {DATA_FILE}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print()

    results = {}

    try:
        results['pandas'] = benchmark_pandas_with_warmup()
    except Exception as e:
        print(f"Pandas failed: {e}")
        results['pandas'] = {'error': str(e)}

    try:
        results['polars'] = benchmark_polars_with_warmup()
    except Exception as e:
        print(f"Polars failed: {e}")
        results['polars'] = {'error': str(e)}

    try:
        results['pyarrow'] = benchmark_pyarrow_with_warmup()
    except Exception as e:
        print(f"PyArrow failed: {e}")
        results['pyarrow'] = {'error': str(e)}

    try:
        results['dask'] = benchmark_dask_with_warmup()
    except Exception as e:
        print(f"Dask failed: {e}")
        results['dask'] = {'error': str(e)}

    try:
        results['pyspark'] = benchmark_pyspark_with_warmup()
    except Exception as e:
        print(f"PySpark failed: {e}")
        results['pyspark'] = {'error': str(e)}

    # Save results
    output_file = os.path.join(RESULTS_DIR, 'performance_metrics_1m_warmup.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("WARMUP BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_file}")

    # Print summary
    print("\nPerformance Summary (with warmup):")
    print("-" * 60)
    for lib, metrics in results.items():
        if 'error' not in metrics:
            print(f"{lib:12s}: {metrics['total_time']:8.3f}s")


if __name__ == "__main__":
    main()
