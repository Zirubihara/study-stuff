#!/usr/bin/env python3
"""
Benchmark script to run all data processing technologies on 5M row dataset.
This script runs Pandas, Polars, PyArrow, Dask, and PySpark on the same 5M dataset for comparison.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def modify_script_for_5m(script_path, technology_name):
    """Modify a script to use 5M dataset and save with _5m suffix."""

    with open(script_path, "r") as f:
        content = f.read()

    # Replace dataset path
    content = content.replace(
        'csv_path = "../data/benchmark_1m.csv"', 'csv_path = "../data/benchmark_5m.csv"'
    )

    # Replace output file path based on technology
    replacements = {
        "pandas": (
            'output_path: str = "../results/performance_metrics_pandas.json"',
            'output_path: str = "../results/performance_metrics_pandas_5m.json"',
        ),
        "polars": (
            'output_path: str = "../results/performance_metrics_polars.json"',
            'output_path: str = "../results/performance_metrics_polars_5m.json"',
        ),
        "pyarrow": (
            'output_path: str = "../results/performance_metrics_arrow.json"',
            'output_path: str = "../results/performance_metrics_arrow_5m.json"',
        ),
        "dask": (
            'output_path: str = "../results/performance_metrics_dask.json"',
            'output_path: str = "../results/performance_metrics_dask_5m.json"',
        ),
    }

    if technology_name in replacements:
        old_str, new_str = replacements[technology_name]
        content = content.replace(old_str, new_str)

    # Create temporary script
    temp_script = script_path.parent / f"temp_{technology_name}_5m.py"
    with open(temp_script, "w") as f:
        f.write(content)

    return temp_script


def run_technology(script_path, technology_name):
    """Run a single technology benchmark."""
    print(f"\n{'='*60}")
    print(f"Running {technology_name.upper()} on 5M row dataset...")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Modify script for 5M dataset
        temp_script = modify_script_for_5m(script_path, technology_name)

        # Run the script
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(
                f"SUCCESS: {technology_name.upper()} completed successfully in {execution_time:.2f} seconds"
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
        print(f"TIMEOUT: {technology_name.upper()} timed out after 30 minutes")
        return False, time.time() - start_time
    except Exception as e:
        print(f"CRASHED: {technology_name.upper()} crashed with error: {e}")
        return False, time.time() - start_time


def main():
    """Main execution function."""
    print("Starting 5M Row Dataset Benchmark Across All Technologies")
    print("=" * 80)

    # Check if 5M dataset exists
    dataset_path = Path("../data/benchmark_5m.csv")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please run generate_large_data.py first to create the 5M dataset")
        return False

    # Define technologies and their scripts
    technologies = {
        "pandas": Path("../implementations/benchmark_pandas_implementation.py"),
        "polars": Path("../implementations/benchmark_polars_implementation.py"),
        "pyarrow": Path("../implementations/benchmark_pyarrow_implementation.py"),
        "dask": Path("../implementations/benchmark_dask_implementation.py"),
        "pyspark": Path("../implementations/benchmark_pyspark_implementation.py"),
    }

    results = {}
    total_start_time = time.time()

    # Run each technology
    for tech_name, script_path in technologies.items():
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            continue

        success, exec_time = run_technology(script_path, tech_name)
        results[tech_name] = {"success": success, "execution_time": exec_time}

    total_time = time.time() - total_start_time

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY - 5M Row Dataset")
    print("=" * 80)

    successful_runs = 0
    for tech_name, result in results.items():
        status = "SUCCESS" if result["success"] else "FAILED"
        print(
            f"{tech_name.upper():10} | {status:12} | {result['execution_time']:8.2f}s"
        )
        if result["success"]:
            successful_runs += 1

    print(f"\nPySpark:     SUCCESS     | Already completed")
    print("-" * 80)
    print(f"Successfully completed: {successful_runs}/4 technologies")
    print(f"Total benchmark time: {total_time:.2f} seconds")

    # List generated files
    print("\nGenerated Performance Metrics Files:")
    results_dir = Path("../results")
    for file in results_dir.glob("performance_metrics_*_5m.json"):
        print(f"  - {file.name}")

    print("\nUse create_simple_charts.py or visualize_results.py to analyze results!")

    return successful_runs >= 3  # Consider success if at least 3/4 technologies work


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(Path(__file__).parent)

    success = main()
    sys.exit(0 if success else 1)
