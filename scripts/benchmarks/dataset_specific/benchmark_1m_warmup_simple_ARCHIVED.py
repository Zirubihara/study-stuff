"""
Benchmark 1M dataset with warmup runs (Pandas, Polars, PyArrow, Dask only).
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
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, 'benchmark_1m.csv')
WARMUP_RUNS = 2  # Number of warmup iterations

# Column names for the CSV file (no header in file)
COLUMN_NAMES = [
    "year_month",
    "category1",
    "category2",
    "category3",
    "code",
    "flag",
    "value1",
    "value2",
]


def benchmark_pandas_with_warmup():
    """Benchmark Pandas with warmup runs."""
    print("\n=== Pandas Benchmark (with warmup) ===")

    # Warmup phase
    print(f"Running {WARMUP_RUNS} warmup iterations...")
    for i in range(WARMUP_RUNS):
        print(f"  Warmup {i+1}/{WARMUP_RUNS}...")
        df = pd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=None)
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
    df = pd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=None)
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

    metrics['total_time'] = sum([v for k, v in metrics.items() if 'time' in k and k != 'total_time'])
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
        df = pl.read_csv(DATA_FILE, has_header=False, new_columns=COLUMN_NAMES)
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
    df = pl.read_csv(DATA_FILE, has_header=False, new_columns=COLUMN_NAMES)
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

    metrics['total_time'] = sum([v for k, v in metrics.items() if 'time' in k and k != 'total_time'])
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
        table = csv.read_csv(DATA_FILE, read_options=csv.ReadOptions(column_names=COLUMN_NAMES))
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
    table = csv.read_csv(DATA_FILE, read_options=csv.ReadOptions(column_names=COLUMN_NAMES))
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

    metrics['total_time'] = sum([v for k, v in metrics.items() if 'time' in k and k != 'total_time'])
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
        df = dd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=None)
        df = df.fillna(0)
        grouped = df.groupby(['year_month', 'category1', 'category2'])['value2'].agg(['mean', 'median', 'max']).compute()
        sorted_df = df.sort_values('value2', ascending=False).compute()
        mean_val = df['value2'].mean().compute()
        filtered = df[df['value2'] > mean_val].compute()

    # Actual measurement
    print("Running measured iteration...")
    metrics = {}

    start = time.time()
    df = dd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=None)
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

    metrics['total_time'] = sum([v for k, v in metrics.items() if 'time' in k and k != 'total_time'])
    metrics['row_count'] = len(df)
    metrics['filtered_count'] = len(filtered)

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
