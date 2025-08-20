#!/usr/bin/env python3
"""
Simple specialized benchmarks to showcase each technology's strengths clearly.
"""

import json
import time
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv


def pandas_strength_benchmark():
    """Showcase Pandas strengths: string ops, datetime, missing data, rich API."""
    print("=" * 60)
    print("PANDAS STRENGTH TEST: Complex Data Wrangling")
    print("=" * 60)

    start_time = time.time()

    # Load complex employee data
    load_start = time.time()
    df = pd.read_csv("../data/specialized/pandas_showcase.csv")
    load_time = time.time() - load_start
    print(f"Loaded {len(df):,} rows in {load_time:.2f}s")

    # STRING OPERATIONS (Pandas specialty)
    string_start = time.time()
    df["first_name"] = df["full_name"].str.split().str[0]
    df["last_name"] = df["full_name"].str.split().str[-1]
    df["email_domain"] = df["email"].str.extract(r"@([^.]+)")
    df["phone_digits"] = df["phone"].str.replace(r"[^\d]", "", regex=True)
    df["phone_clean"] = df["phone_digits"].str.replace(
        r"(\d{3})(\d{3})(\d{4})", r"(\1) \2-\3", regex=True
    )
    string_time = time.time() - string_start
    print(f"SUCCESS: String processing: {string_time:.2f}s")

    # DATETIME OPERATIONS (Pandas specialty)
    datetime_start = time.time()
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["birth_date"] = pd.to_datetime(df["birth_date"])
    df["age"] = (pd.Timestamp.now() - df["birth_date"]).dt.days // 365
    df["tenure_years"] = (pd.Timestamp.now() - df["hire_date"]).dt.days // 365
    df["hire_month"] = df["hire_date"].dt.month_name()
    df["birth_decade"] = (df["birth_date"].dt.year // 10) * 10
    datetime_time = time.time() - datetime_start
    print(f"SUCCESS: DateTime processing: {datetime_time:.2f}s")

    # MISSING DATA HANDLING (Pandas specialty)
    missing_start = time.time()
    print(f"  Original missing values: {df.isnull().sum().sum():,}")
    df["salary_filled"] = df["salary"].fillna(
        df.groupby("department")["salary"].transform("median")
    )
    df["bonus_filled"] = df["bonus"].fillna(
        df.groupby(["department", "city"])["bonus"].transform("mean")
    )
    df["bonus_filled"] = df["bonus_filled"].fillna(0)
    print(f"  After filling: {df.isnull().sum().sum():,}")
    missing_time = time.time() - missing_start
    print(f"SUCCESS: Missing data handling: {missing_time:.2f}s")

    # COMPLEX PIVOT OPERATIONS (Pandas specialty)
    pivot_start = time.time()
    salary_pivot = pd.pivot_table(
        df,
        values="salary_filled",
        index="department",
        columns="city",
        aggfunc=["mean", "count"],
        fill_value=0,
    )
    pivot_time = time.time() - pivot_start
    print(f"SUCCESS: Pivot table creation: {pivot_time:.2f}s")

    total_time = time.time() - start_time
    print(f"PANDAS TOTAL: {total_time:.2f} seconds")

    return {
        "technology": "pandas_specialized",
        "dataset": "complex_employee_data",
        "rows": len(df),
        "load_time": load_time,
        "string_ops_time": string_time,
        "datetime_ops_time": datetime_time,
        "missing_data_time": missing_time,
        "pivot_ops_time": pivot_time,
        "total_time": total_time,
        "strength": "Complex data wrangling, strings, dates, missing data",
    }


def pyarrow_strength_benchmark():
    """Showcase PyArrow strengths: pure numerical, columnar speed, vectorized ops."""
    print("\n" + "=" * 60)
    print("PYARROW STRENGTH TEST: Massive Numerical Analytics")
    print("=" * 60)

    start_time = time.time()

    # Load pure numerical data
    load_start = time.time()
    table = csv.read_csv("../data/specialized/pyarrow_showcase.csv")
    load_time = time.time() - load_start
    print(f"SUCCESS: Loaded {len(table):,} rows in {load_time:.2f}s")

    # VECTORIZED FILTERING (PyArrow specialty)
    filter_start = time.time()
    high_value = pc.greater(table["total_amount"], 100)
    premium = pc.equal(table["is_premium"], 1)
    high_score = pc.greater(table["customer_score"], 80)
    complex_filter = pc.and_(pc.and_(high_value, premium), high_score)
    filtered = table.filter(complex_filter)
    filter_time = time.time() - filter_start
    print(
        f"SUCCESS: Complex filtering: {filter_time:.2f}s ({len(filtered):,} rows selected)"
    )

    # VECTORIZED COMPUTATIONS (PyArrow specialty)
    compute_start = time.time()
    # Revenue calculations
    net_amount = pc.subtract(table["total_amount"], table["discount"])
    final_amount = pc.add(net_amount, table["tax"])
    profit_margin = pc.divide(table["discount"], table["total_amount"])

    # Statistical computations
    total_revenue = pc.sum(final_amount)
    avg_transaction = pc.mean(table["total_amount"])
    transaction_std = pc.stddev(table["total_amount"])
    compute_time = time.time() - compute_start
    print(f"SUCCESS: Vectorized computations: {compute_time:.2f}s")

    # LIGHTNING-FAST AGGREGATIONS (PyArrow specialty)
    agg_start = time.time()
    grouped = table.group_by(["category_id", "region_id"]).aggregate(
        [
            ("total_amount", "sum"),
            ("total_amount", "mean"),
            ("quantity", "sum"),
            ("customer_score", "mean"),
            ("transaction_id", "count"),
        ]
    )
    agg_time = time.time() - agg_start
    print(f"SUCCESS: Group-by aggregations: {agg_time:.2f}s ({len(grouped):,} groups)")

    # COLUMNAR SORTING (PyArrow specialty)
    sort_start = time.time()
    sorted_table = table.sort_by(
        [("total_amount", "descending"), ("customer_score", "descending")]
    )
    sort_time = time.time() - sort_start
    print(f"SUCCESS: Multi-column sorting: {sort_time:.2f}s")

    total_time = time.time() - start_time
    print(f"PYARROW TOTAL: {total_time:.2f} seconds")

    return {
        "technology": "pyarrow_specialized",
        "dataset": "pure_numerical_transactions",
        "rows": len(table),
        "load_time": load_time,
        "filtering_time": filter_time,
        "compute_time": compute_time,
        "aggregation_time": agg_time,
        "sorting_time": sort_time,
        "total_time": total_time,
        "strength": "Columnar speed, vectorized operations, massive numerical data",
    }


def polars_strength_benchmark():
    """Showcase Polars strengths: query optimization, time series, memory efficiency."""
    print("\n" + "=" * 60)
    print("POLARS STRENGTH TEST: Complex Query Optimization")
    print("=" * 60)

    start_time = time.time()

    # Load time series IoT data
    load_start = time.time()
    df = pl.read_csv("../data/specialized/polars_showcase.csv")
    load_time = time.time() - load_start
    print(f"Loaded {len(df):,} rows in {load_time:.2f}s")

    # COMPLEX QUERY OPTIMIZATION (Polars specialty)
    query_start = time.time()
    optimized_result = (
        df.lazy()
        .with_columns(
            [
                pl.col("timestamp").str.to_datetime().alias("dt"),
                (pl.col("temperature") * 9 / 5 + 32).alias("temp_f"),
                pl.when(pl.col("battery_level") < 20)
                .then(pl.lit("critical"))
                .when(pl.col("battery_level") < 50)
                .then(pl.lit("low"))
                .otherwise(pl.lit("normal"))
                .alias("battery_status"),
            ]
        )
        .filter((pl.col("quality_score") > 0.8) & (pl.col("error_code") == 0))
        .group_by(["device_id", "sensor_type", "battery_status"])
        .agg(
            [
                pl.col("measurement_value").mean().alias("avg_reading"),
                pl.col("temperature").std().alias("temp_variance"),
                pl.col("humidity").median().alias("median_humidity"),
                pl.count().alias("reading_count"),
            ]
        )
        .filter(pl.col("reading_count") > 10)
        .sort("avg_reading", descending=True)
        .collect()
    )
    query_time = time.time() - query_start
    print(
        f"SUCCESS: Complex optimized query: {query_time:.2f}s ({len(optimized_result):,} results)"
    )

    # TIME SERIES OPERATIONS (Polars specialty)
    timeseries_start = time.time()
    ts_df = df.with_columns([pl.col("timestamp").str.to_datetime().alias("dt")]).sort(
        "dt"
    )

    windowed = ts_df.with_columns(
        [
            pl.col("measurement_value")
            .rolling_mean(window_size=100)
            .alias("rolling_avg"),
            pl.col("temperature").rolling_std(window_size=50).alias("temp_volatility"),
            pl.col("measurement_value").shift(1).alias("prev_value"),
            (pl.col("measurement_value") - pl.col("measurement_value").shift(1)).alias(
                "change"
            ),
        ]
    )
    timeseries_time = time.time() - timeseries_start
    print(f"SUCCESS: Time series + windowing: {timeseries_time:.2f}s")

    # MEMORY-EFFICIENT AGGREGATIONS (Polars specialty)
    agg_start = time.time()
    device_stats = df.group_by("device_id").agg(
        [
            pl.col("measurement_value").quantile(0.95).alias("p95"),
            pl.col("battery_level").first().alias("initial_battery"),
            pl.col("battery_level").last().alias("final_battery"),
            pl.col("quality_score").mean().alias("avg_quality"),
            pl.col("error_code").sum().alias("total_errors"),
            pl.col("temperature").min().alias("min_temp"),
            pl.col("temperature").max().alias("max_temp"),
        ]
    )
    agg_time = time.time() - agg_start
    print(
        f"SUCCESS: Device aggregations: {agg_time:.2f}s ({len(device_stats):,} devices)"
    )

    # CATEGORICAL OPERATIONS (Polars specialty)
    categorical_start = time.time()
    sensor_summary = (
        df.with_columns(
            [
                pl.col("sensor_type").cast(pl.Categorical),
            ]
        )
        .group_by("sensor_type")
        .agg(
            [
                pl.col("device_id").n_unique().alias("unique_devices"),
                pl.col("measurement_value").mean().alias("avg_measurement"),
                pl.col("quality_score").mean().alias("avg_quality"),
            ]
        )
        .sort("avg_measurement", descending=True)
    )
    categorical_time = time.time() - categorical_start
    print(f"SUCCESS: Categorical analysis: {categorical_time:.2f}s")

    total_time = time.time() - start_time
    print(f"POLARS TOTAL: {total_time:.2f} seconds")

    return {
        "technology": "polars_specialized",
        "dataset": "iot_timeseries_data",
        "rows": len(df),
        "load_time": load_time,
        "query_optimization_time": query_time,
        "timeseries_ops_time": timeseries_time,
        "aggregation_time": agg_time,
        "categorical_ops_time": categorical_time,
        "total_time": total_time,
        "strength": "Query optimization, time series, memory efficiency, lazy evaluation",
    }


def main():
    """Run all specialized benchmarks and compare results."""
    print("SPECIALIZED STRENGTH SHOWCASE")
    print("Testing each technology on data optimized for its strengths")
    print("=" * 80)

    # Ensure results directory exists
    Path("../results").mkdir(exist_ok=True)

    # Run all benchmarks
    results = []

    try:
        pandas_result = pandas_strength_benchmark()
        results.append(pandas_result)

        pyarrow_result = pyarrow_strength_benchmark()
        results.append(pyarrow_result)

        polars_result = polars_strength_benchmark()
        results.append(polars_result)

    except Exception as e:
        print(f"Error during benchmarks: {e}")
        return

    # Save all results
    with open("../results/specialized_strength_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    # Print comparison
    print("\n" + "=" * 80)
    print("SPECIALIZED STRENGTH COMPARISON RESULTS")
    print("=" * 80)

    for result in results:
        tech = result["technology"].replace("_specialized", "").upper()
        time_sec = result["total_time"]
        rows = result["rows"]
        throughput = rows / time_sec / 1000  # K rows/sec

        print(
            f"\n{tech:15} | {time_sec:8.2f}s | {rows:>9,} rows | {throughput:>8.1f}K rows/sec"
        )
        print(f"                | Strength: {result['strength']}")
        print(f"                | Dataset:  {result['dataset']}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("- Each technology excels on data optimized for its strengths")
    print("- Choose the right tool for your specific data characteristics")
    print("- Performance varies dramatically based on data type and operations")
    print("=" * 80)


if __name__ == "__main__":
    main()
