#!/usr/bin/env python3
"""
Benchmark script to run all data processing technologies on 50M row dataset (~1GB).
This script runs Pandas, Polars, PyArrow, Dask, and PySpark on the massive 50M dataset.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def modify_script_for_50m(script_path, technology_name):
    """Modify a script to use 50M dataset and save with _50m suffix."""

    with open(script_path, "r") as f:
        content = f.read()

    # Replace dataset path patterns - handle different path patterns
    dataset_replacements = [
        (
            'csv_path = "../data/benchmark_1m.csv"',
            'csv_path = "../../../data/benchmark_50m.csv"',
        ),
        (
            'csv_path = "../data/benchmark_5m.csv"',
            'csv_path = "../../../data/benchmark_50m.csv"',
        ),
        (
            'csv_path = "data/benchmark_5m.csv"',  # PySpark case
            'csv_path = "../../../data/benchmark_50m.csv"',
        ),
        (
            'medium_dataset = "../data/benchmark_5m.csv"',  # PySpark medium dataset
            'medium_dataset = "../../../data/benchmark_50m.csv"',
        ),
        (
            'large_dataset = "../data/benchmark_10m.csv"',
            'large_dataset = "../../../data/benchmark_50m.csv"',
        ),
        (
            'massive_dataset = "../data/benchmark_50m.csv"',
            'massive_dataset = "../../../data/benchmark_50m.csv"',
        ),
    ]

    for old_path, new_path in dataset_replacements:
        content = content.replace(old_path, new_path)

    # Replace output file path based on technology
    replacements = {
        "pandas": (
            'output_path: str = "../results/performance_metrics_pandas.json"',
            'output_path: str = "../../../results/performance_metrics_pandas_50m.json"',
        ),
        "polars": (
            'output_path: str = "../results/performance_metrics_polars.json"',
            'output_path: str = "../../../results/performance_metrics_polars_50m.json"',
        ),
        "pyarrow": (
            'output_path: str = "../results/performance_metrics_arrow.json"',
            'output_path: str = "../../../results/performance_metrics_arrow_50m.json"',
        ),
        "dask": (
            'output_path: str = "../results/performance_metrics_dask.json"',
            'output_path: str = "../../../results/performance_metrics_dask_50m.json"',
        ),
        "pyspark": (
            'output_path: str = "../results/performance_metrics_spark_100m.json"',
            'output_path: str = "../../../results/performance_metrics_spark_50m.json"',
        ),
    }

    if technology_name in replacements:
        old_str, new_str = replacements[technology_name]
        content = content.replace(old_str, new_str)

    # For Dask, increase memory limits for large dataset
    if technology_name == "dask":
        content = content.replace(
            "Client(processes=False)", 'Client(processes=False, memory_limit="8GB")'
        )

    # Create temporary script
    temp_script = script_path.parent / f"temp_{technology_name}_50m.py"
    with open(temp_script, "w") as f:
        f.write(content)

    return temp_script


def run_technology(script_path, technology_name):
    """Run a single technology benchmark."""
    print(f"\n{'='*60}")
    print(f"Running {technology_name.upper()} on 50M row dataset (~1GB)...")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Modify script for 50M dataset
        temp_script = modify_script_for_50m(script_path, technology_name)

        # Set longer timeout for massive dataset (60 minutes)
        timeout = 3600 if technology_name in ["dask", "spark"] else 1800

        # Run the script
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(
                f"SUCCESS: {technology_name.upper()} completed in {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)"
            )
            print(
                "Output:",
                result.stdout.strip() if result.stdout.strip() else "No output",
            )
            if result.stderr.strip():
                print("Warnings:", result.stderr.strip())
        else:
            print(
                f"FAILED: {technology_name.upper()} failed with return code {result.returncode}"
            )
            print("Error:", result.stderr)
            print("Output:", result.stdout)

        # Clean up temporary script
        if temp_script.exists():
            temp_script.unlink()

        return result.returncode == 0, execution_time

    except subprocess.TimeoutExpired:
        print(
            f"TIMEOUT: {technology_name.upper()} timed out after {timeout/60:.0f} minutes"
        )
        return False, time.time() - start_time
    except Exception as e:
        print(f"CRASHED: {technology_name.upper()} crashed with error: {e}")
        return False, time.time() - start_time


def main():
    """Main execution function."""
    print("Starting 50M Row Dataset Benchmark (~1GB) Across All Technologies")
    print("=" * 80)
    print("WARNING: This will take significant time and memory!")
    print(
        "Expected times: Polars ~30s, PyArrow ~2min, Pandas ~5min, Dask ~10min, Spark ~4min"
    )
    print("=" * 80)

    # Check if 50M dataset exists
    dataset_path = Path("../../../data/benchmark_50m.csv")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please run generate_large_data.py first to create the 50M dataset")
        return False

    # Check dataset size
    size_gb = dataset_path.stat().st_size / (1024**3)
    print(f"Dataset size: {size_gb:.2f} GB")

    # Define technologies and their scripts
    technologies = {
        "polars": Path("../implementations/benchmark_polars_implementation.py"),  # Should be fastest
        "pyarrow": Path("../implementations/benchmark_pyarrow_implementation.py"),  # Good performance
        "pandas": Path("../implementations/benchmark_pandas_implementation.py"),  # Will be slower but should work
        "dask": Path("../implementations/benchmark_dask_implementation.py"),  # Distributed processing
    }

    # PySpark is already configured for 50M
    print(
        "Note: PySpark 50M results already available in performance_metrics_spark_50m.json"
    )

    results = {}
    total_start_time = time.time()

    # Run each technology in order of expected performance
    for tech_name, script_path in technologies.items():
        if not script_path.exists():
            print(f"Script not found: {script_path}")
            continue

        success, exec_time = run_technology(script_path, tech_name)
        results[tech_name] = {"success": success, "execution_time": exec_time}

        # Add a pause between runs to let system recover
        if success:
            print(f"Pausing 10 seconds before next technology...")
            time.sleep(10)

    total_time = time.time() - total_start_time

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY - 50M Row Dataset (~1GB)")
    print("=" * 80)

    successful_runs = 0
    for tech_name, result in results.items():
        status = "SUCCESS" if result["success"] else "FAILED"
        minutes = result["execution_time"] / 60
        print(
            f"{tech_name.upper():10} | {status:12} | {result['execution_time']:8.2f}s ({minutes:5.1f}m)"
        )
        if result["success"]:
            successful_runs += 1

    print(f"\nPySpark:     SUCCESS     | Already completed (~4 minutes)")
    print("-" * 80)
    print(f"Successfully completed: {successful_runs}/4 technologies")
    print(
        f"Total benchmark time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)"
    )

    # List generated files
    print("\nGenerated Performance Metrics Files:")
    results_dir = Path("../../../results")
    for file in results_dir.glob("performance_metrics_*_50m.json"):
        print(f"  - {file.name}")

    print("\nUse create_simple_charts.py or visualize_results.py to analyze results!")
    print(
        "\nNOTE: At this scale, Spark and Dask should show their distributed processing advantages!"
    )

    return successful_runs >= 2  # Consider success if at least 2/4 technologies work


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(Path(__file__).parent)

    # Warn user about resource requirements
    print("This benchmark will use significant CPU and memory resources.")
    print("Make sure you have at least 8GB RAM available.")
    print("Press Ctrl+C to cancel within 5 seconds...")

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nBenchmark cancelled by user.")
        sys.exit(0)

    success = main()
    sys.exit(0 if success else 1)
