#!/usr/bin/env python3
"""
Specialized benchmarks to showcase each technology's strengths.
Tests each library on data optimized for its capabilities.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def create_pandas_specialized_script():
    """Create a script showcasing Pandas strengths."""
    script_content = '''
import pandas as pd
import json
import time
from pathlib import Path

def pandas_specialized_benchmark():
    """Showcase Pandas strengths with complex data operations."""
    
    print("Running Pandas on specialized dataset (complex strings, dates, missing data)...")
    start_time = time.time()
    
    # Load data
    load_start = time.time()
    df = pd.read_csv("../data/specialized/pandas_showcase.csv")
    load_time = time.time() - load_start
    
    # String operations (Pandas strength)
    string_start = time.time()
    df['first_name'] = df['full_name'].str.split().str[0]
    df['domain'] = df['email'].str.extract(r'@([^.]+)')
    df['phone_clean'] = df['phone'].str.replace(r'[^\d]', '', regex=True)
    string_time = time.time() - string_start
    
    # DateTime operations (Pandas strength)
    datetime_start = time.time()
    df['hire_date'] = pd.to_datetime(df['hire_date'])
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['age'] = (pd.Timestamp.now() - df['birth_date']).dt.days // 365
    df['tenure_years'] = (pd.Timestamp.now() - df['hire_date']).dt.days // 365
    df['hire_year'] = df['hire_date'].dt.year
    datetime_time = time.time() - datetime_start
    
    # Missing data handling (Pandas strength)
    missing_start = time.time()
    df['salary_filled'] = df['salary'].fillna(
        df.groupby('department')['salary'].transform('median')
    )
    df['bonus_filled'] = df['bonus'].fillna(0)
    missing_data_pct = (
        df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    )
    missing_time = time.time() - missing_start
    
    # Complex aggregations
    agg_start = time.time()
    summary = df.groupby(['department', 'city']).agg({
        'salary_filled': ['mean', 'median', 'std'],
        'age': ['mean', 'min', 'max'],
        'tenure_years': 'mean',
        'rating': 'mean'
    }).round(2)
    agg_time = time.time() - agg_start
    
    # Pivot operations (Pandas strength)
    pivot_start = time.time()
    pivot_table = df.pivot_table(
        values='salary_filled', 
        index='department', 
        columns='city', 
        aggfunc='mean'
    ).fillna(0)
    pivot_time = time.time() - pivot_start
    
    total_time = time.time() - start_time
    
    metrics = {
        'technology': 'pandas_specialized',
        'dataset_size_mb': (
            Path("../data/specialized/pandas_showcase.csv").stat().st_size
            / (1024**2)
        ),
        'row_count': len(df),
        'load_time_seconds': load_time,
        'string_operations_time_seconds': string_time,
        'datetime_operations_time_seconds': datetime_time,
        'missing_data_handling_time_seconds': missing_time,
        'aggregation_time_seconds': agg_time,
        'pivot_operations_time_seconds': pivot_time,
        'total_time_seconds': total_time,
        'missing_data_percentage': missing_data_pct,
        'unique_departments': df['department'].nunique(),
        'average_salary': df['salary_filled'].mean()
    }
    
    # Save metrics
    with open("../results/pandas_specialized_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Pandas specialized benchmark completed in {total_time:.2f} seconds")
    print(f"String operations: {string_time:.2f}s")
    print(f"DateTime operations: {datetime_time:.2f}s")
    print(f"Missing data handling: {missing_time:.2f}s")
    return metrics

if __name__ == "__main__":
    pandas_specialized_benchmark()
'''

    with open("temp_pandas_specialized.py", "w") as f:
        f.write(script_content)
    return "temp_pandas_specialized.py"


def create_pyarrow_specialized_script():
    """Create a script showcasing PyArrow strengths."""
    script_content = '''
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.compute as pc
import json
import time
from pathlib import Path

def pyarrow_specialized_benchmark():
    """Showcase PyArrow strengths with pure numerical columnar operations."""
    
    print("Running PyArrow on specialized dataset (pure numerical, columnar analytics)...")
    start_time = time.time()
    
    # Load data (PyArrow strength - fast columnar loading)
    load_start = time.time()
    table = csv.read_csv("../data/specialized/pyarrow_showcase.csv")
    load_time = time.time() - load_start
    
    # Columnar filtering (PyArrow strength)
    filter_start = time.time()
    high_value = pc.greater(table['total_amount'], 100)
    premium_users = pc.equal(table['is_premium'], 1)
    filtered_table = table.filter(pc.and_(high_value, premium_users))
    filter_time = time.time() - filter_start
    
    # Vectorized computations (PyArrow strength)
    compute_start = time.time()
    # Revenue calculations
    net_revenue = pc.subtract(table['total_amount'], table['discount'])
    total_revenue_with_tax = pc.add(net_revenue, table['tax'])
    
    # Statistical computations
    avg_amount = pc.mean(table['total_amount'])
    total_transactions = pc.count(table['transaction_id'])
    max_discount = pc.max(table['discount'])
    compute_time = time.time() - compute_start
    
    # Group by operations (efficient on columnar data)
    groupby_start = time.time()
    # Group by category and region
    grouped = table.group_by(['category_id', 'region_id']).aggregate([
        ('total_amount', 'sum'),
        ('quantity', 'mean'),
        ('customer_score', 'mean'),
        ('transaction_id', 'count')
    ])
    groupby_time = time.time() - groupby_start
    
    # Sorting (PyArrow strength)
    sort_start = time.time()
    sorted_table = table.sort_by([('total_amount', 'descending')])
    sort_time = time.time() - sort_start
    
    # Advanced analytics
    analytics_start = time.time()
    # Calculate percentiles
    percentile_95 = pc.quantile(table['total_amount'], q=0.95)
    
    # Boolean operations
    high_value_customers = pc.greater(table['customer_score'], 80)
    high_value_count = pc.sum(pc.cast(high_value_customers, pa.int64()))
    analytics_time = time.time() - analytics_start
    
    total_time = time.time() - start_time
    
    metrics = {
        'technology': 'pyarrow_specialized',
        'dataset_size_mb': (
            Path("../data/specialized/pyarrow_showcase.csv").stat().st_size
            / (1024**2)
        ),
        'row_count': len(table),
        'load_time_seconds': load_time,
        'filtering_time_seconds': filter_time,
        'vectorized_compute_time_seconds': compute_time,
        'groupby_operations_time_seconds': groupby_time,
        'sorting_time_seconds': sort_time,
        'analytics_operations_time_seconds': analytics_time,
        'total_time_seconds': total_time,
        'average_transaction_amount': float(avg_amount.as_py()),
        'total_transactions': int(total_transactions.as_py()),
        'high_value_customers': int(high_value_count.as_py())
    }
    
    # Save metrics
    with open("../results/pyarrow_specialized_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"PyArrow specialized benchmark completed in {total_time:.2f} seconds")
    print(f"Columnar loading: {load_time:.2f}s")
    print(f"Vectorized operations: {compute_time:.2f}s")
    print(f"Filtering: {filter_time:.2f}s")
    return metrics

if __name__ == "__main__":
    pyarrow_specialized_benchmark()
'''

    with open("temp_pyarrow_specialized.py", "w") as f:
        f.write(script_content)
    return "temp_pyarrow_specialized.py"


def create_polars_specialized_script():
    """Create a script showcasing Polars strengths."""
    script_content = '''
import polars as pl
import json
import time
from pathlib import Path

def polars_specialized_benchmark():
    """Showcase Polars strengths with complex queries and time series operations."""
    
    print("Running Polars on specialized dataset (time series, complex queries)...")
    start_time = time.time()
    
    # Lazy loading (Polars strength)
    load_start = time.time()
    df = pl.read_csv("../data/specialized/polars_showcase.csv")
    load_time = time.time() - load_start
    
    # Complex query optimization (Polars strength)
    query_start = time.time()
    complex_query = (
        df.lazy()
        .with_columns([
            pl.col("timestamp").str.to_datetime().alias("datetime"),
            (pl.col("temperature") * 9 / 5 + 32).alias("temp_fahrenheit"),
            pl.when(pl.col("battery_level") < 20).then("low")
            .when(pl.col("battery_level") < 50).then("medium")
            .otherwise("high").alias("battery_status")
        ])
        .filter(
            (pl.col("quality_score") > 0.8) &
            (pl.col("error_code") == 0) &
            (pl.col("temperature") > 0)
        )
        .group_by(["device_id", "sensor_type"])
        .agg([
            pl.col("measurement_value").mean().alias("avg_measurement"),
            pl.col("temperature").min().alias("min_temp"),
            pl.col("temperature").max().alias("max_temp"),
            pl.col("humidity").mean().alias("avg_humidity"),
            pl.col("battery_level").last().alias("current_battery"),
            pl.col("datetime").count().alias("reading_count")
        ])
        .sort("avg_measurement", descending=True)
        .collect()
    )
    query_time = time.time() - query_start
    
    # Time series operations (Polars strength)
    timeseries_start = time.time()
    df_with_time = df.with_columns([
        pl.col("timestamp").str.to_datetime().alias("datetime")
    ])
    
    # Window functions and rolling statistics
    windowed = (
        df_with_time
        .sort("datetime")
        .with_columns([
            pl.col("measurement_value")
            .rolling_mean(window_size=10)
            .alias("rolling_avg_10"),
            pl.col("temperature")
            .rolling_std(window_size=5)
            .alias("temp_volatility"),
            pl.col("measurement_value").shift(1).alias("prev_measurement"),
            (
                pl.col("measurement_value")
                - pl.col("measurement_value").shift(1)
            ).alias("measurement_diff")
        ])
    )
    timeseries_time = time.time() - timeseries_start
    
    # Memory-efficient aggregations (Polars strength)
    agg_start = time.time()
    device_summary = (
        df.group_by("device_id")
        .agg([
            pl.col("measurement_value")
            .quantile(0.95)
            .alias("p95_measurement"),
            pl.col("battery_level").first().alias("initial_battery"),
            pl.col("battery_level").last().alias("final_battery"),
            pl.col("temperature").std().alias("temp_stability"),
            pl.col("quality_score").mean().alias("avg_quality"),
            pl.when(pl.col("error_code") > 0)
            .then(1)
            .otherwise(0)
            .sum()
            .alias("error_count")
        ])
    )
    agg_time = time.time() - agg_start
    
    # String operations and categorical data
    categorical_start = time.time()
    sensor_analysis = (
        df.with_columns([
            pl.col("sensor_type").cast(pl.Categorical),
            pl.when(pl.col("signal_strength") > -70).then("good")
            .when(pl.col("signal_strength") > -85).then("fair")
            .otherwise("poor").alias("signal_quality")
        ])
        .group_by(["sensor_type", "signal_quality"])
        .agg([
            pl.col("device_id").n_unique().alias("device_count"),
            pl.col("measurement_value").mean().alias("avg_reading")
        ])
    )
    categorical_time = time.time() - categorical_start
    
    total_time = time.time() - start_time
    
    metrics = {
        'technology': 'polars_specialized',
        'dataset_size_mb': (
            Path("../data/specialized/polars_showcase.csv").stat().st_size
            / (1024**2)
        ),
        'row_count': len(df),
        'load_time_seconds': load_time,
        'complex_query_time_seconds': query_time,
        'timeseries_operations_time_seconds': timeseries_time,
        'aggregation_time_seconds': agg_time,
        'categorical_operations_time_seconds': categorical_time,
        'total_time_seconds': total_time,
        'unique_devices': df['device_id'].n_unique(),
        'unique_sensors': df['sensor_type'].n_unique(),
        'avg_quality_score': df['quality_score'].mean()
    }
    
    # Save metrics
    with open("../results/polars_specialized_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Polars specialized benchmark completed in {total_time:.2f} seconds")
    print(f"Complex query optimization: {query_time:.2f}s")
    print(f"Time series operations: {timeseries_time:.2f}s")
    print(f"Memory-efficient aggregations: {agg_time:.2f}s")
    return metrics

if __name__ == "__main__":
    polars_specialized_benchmark()
'''

    with open("temp_polars_specialized.py", "w") as f:
        f.write(script_content)
    return "temp_polars_specialized.py"


def run_specialized_benchmark(script_path, technology_name):
    """Run a specialized benchmark script."""
    print(f"\n{'='*60}")
    print(f"Running {technology_name.upper()} Specialized Benchmark")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(
                f"SUCCESS: {technology_name} completed in "
                f"{execution_time:.2f} seconds"
            )
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(
                f"FAILED: {technology_name} failed with return code "
                f"{result.returncode}"
            )
            print("Error:", result.stderr)
            if result.stdout:
                print("Output:", result.stdout)

        # Clean up script
        if Path(script_path).exists():
            Path(script_path).unlink()

        return result.returncode == 0, execution_time

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {technology_name} exceeded 5 minute limit")
        return False, time.time() - start_time
    except Exception as e:
        print(f"ERROR: {technology_name} crashed: {e}")
        return False, time.time() - start_time


def main():
    """Run all specialized benchmarks."""
    print("SPECIALIZED BENCHMARKS - Each Technology On Its Optimal Data")
    print("=" * 80)

    # Create specialized scripts
    benchmarks = [
        ("Pandas", create_pandas_specialized_script),
        ("PyArrow", create_pyarrow_specialized_script),
        ("Polars", create_polars_specialized_script),
    ]

    results = {}
    total_start = time.time()

    for tech_name, script_creator in benchmarks:
        script_path = script_creator()
        success, exec_time = run_specialized_benchmark(script_path, tech_name)
        results[tech_name.lower()] = {"success": success, "execution_time": exec_time}

    total_time = time.time() - total_start

    # Print summary
    print("\\n" + "=" * 80)
    print("SPECIALIZED BENCHMARK RESULTS")
    print("=" * 80)

    for tech_name, result in results.items():
        status = "SUCCESS" if result["success"] else "FAILED"
        print(
            f"{tech_name.upper():15} | {status:10} | "
            f"{result['execution_time']:8.2f}s"
        )

    print(f"\\nTotal benchmark time: {total_time:.2f} seconds")
    print("\\nGenerated metrics files:")
    print("- ../results/pandas_specialized_metrics.json")
    print("- ../results/pyarrow_specialized_metrics.json")
    print("- ../results/polars_specialized_metrics.json")

    print("\\nThese benchmarks showcase each technology's optimal use cases!")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
