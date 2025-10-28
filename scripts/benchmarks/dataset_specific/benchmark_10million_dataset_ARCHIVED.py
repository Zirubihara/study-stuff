#!/usr/bin/env python3
"""
Benchmark script to run all data processing technologies on 10M row dataset.
This script runs Pandas, Polars, PyArrow, Dask, and PySpark on the same
10M dataset for comparison.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def modify_script_for_10m(script_path, technology_name):
    """Modify a script to use 10M dataset and save with _10m suffix."""

    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace dataset path patterns - handle different path patterns
    # found in implementation files
    dataset_replacements = [
        (
            'csv_path = "../data/benchmark_1m.csv"',
            'csv_path = "../../../data/benchmark_10m.csv"',
        ),
        ("csv_path = small_dataset", "csv_path = medium_dataset"),
        ("csv_path = medium_dataset", 'csv_path = "../../../data/benchmark_10m.csv"'),
        (
            'csv_path = "data/benchmark_5m.csv"',  # PySpark case
            'csv_path = "../../../data/benchmark_10m.csv"',
        ),
        (
            'medium_dataset = "../data/benchmark_5m.csv"',  # PySpark medium dataset
            'medium_dataset = "../../../data/benchmark_10m.csv"',
        ),
        (
            'large_dataset = "../data/benchmark_10m.csv"',  # Fix path consistency
            'large_dataset = "../../../data/benchmark_10m.csv"',
        ),
        (
            '# 5M rows',  # Update comments
            '# 10M rows',
        ),
        (
            '# Choose dataset to use (using 5M for comparison with other libraries)',
            '# Choose dataset to use (using 10M for comparison with other libraries)',
        ),
    ]

    for old_path, new_path in dataset_replacements:
        content = content.replace(old_path, new_path)

    # Replace output file path based on technology
    output_replacements = {
        "pandas": [
            (
                'output_path: str = "../results/performance_metrics_pandas.json"',
                'output_path: str = "../../../results/performance_metrics_pandas_10m.json"',
            ),  # noqa: E501
        ],
        "polars": [
            (
                'output_path: str = "../results/performance_metrics_polars.json"',
                'output_path: str = "../../../results/performance_metrics_polars_10m.json"',
            ),  # noqa: E501
        ],
        "pyarrow": [
            (
                'output_path: str = "../results/performance_metrics_arrow.json"',
                'output_path: str = "../../../results/performance_metrics_arrow_10m.json"',
            ),  # noqa: E501
        ],
        "dask": [
            (
                'output_path: str = "../results/performance_metrics_dask.json"',
                'output_path: str = "../../../results/performance_metrics_dask_10m.json"',
            ),  # noqa: E501
        ],
        "pyspark": [
            (
                'output_path: str = "../results/performance_metrics_spark_100m.json"',  # noqa: E501
                'output_path: str = "../../../results/performance_metrics_spark_10m.json"',
            ),  # noqa: E501
        ],
    }

    if technology_name in output_replacements:
        for old_str, new_str in output_replacements[technology_name]:
            content = content.replace(old_str, new_str)

    # Create temporary script
    temp_script = script_path.parent / f"temp_{technology_name}_10m.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(content)

    return temp_script


def run_technology(script_path, technology_name):
    """Run a single technology benchmark."""
    print(f"\n{'='*60}")
    print(f"Running {technology_name.upper()} on 10M row dataset...")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Modify script for 10M dataset
        temp_script = modify_script_for_10m(script_path, technology_name)

        # Run the script
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1800,  # 30 minute timeout
            check=False,  # We handle return codes manually
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(
                f"SUCCESS: {technology_name.upper()} completed successfully "
                f"in {execution_time:.2f} seconds"
            )
            output_text = (
                result.stdout.strip() if result.stdout.strip() else "No output"
            )
            print("Output:", output_text)
            if result.stderr and result.stderr.strip():
                print("Warnings:", result.stderr.strip())
        else:
            print(
                f"FAILED: {technology_name.upper()} failed with return code "
                f"{result.returncode}"
            )
            print("Error:", result.stderr if result.stderr else "No error info")
            print("Output:", result.stdout if result.stdout else "No output")

        # Clean up temporary script
        if temp_script.exists():
            temp_script.unlink()

        return result.returncode == 0, execution_time

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {technology_name.upper()} timed out after 30 minutes")
        return False, time.time() - start_time
    except (OSError, ValueError) as e:
        print(f"CRASHED: {technology_name.upper()} crashed with error: {e}")
        return False, time.time() - start_time


def main():
    """Main execution function."""
    print("Starting 10M Row Dataset Benchmark Across All Technologies")
    print("=" * 80)

    # Check if 10M dataset exists
    dataset_path = Path("../../../data/benchmark_10m.csv")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please run generate_large_data.py first to create the 10M dataset")
        return False

    # Define technologies and their scripts
    impl_base = "../implementations/"
    technologies = {
        "pandas": Path(f"{impl_base}benchmark_pandas_implementation.py"),
        "polars": Path(f"{impl_base}benchmark_polars_implementation.py"),
        "pyarrow": Path(f"{impl_base}benchmark_pyarrow_implementation.py"),
        "dask": Path(f"{impl_base}benchmark_dask_implementation.py"),
        "pyspark": Path(f"{impl_base}benchmark_pyspark_implementation.py"),
    }

    results = {}
    total_start_time = time.time()

    # Run each technology
    for tech_name, script_path in technologies.items():
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            continue

        tech_success, exec_time = run_technology(script_path, tech_name)
        results[tech_name] = {"success": tech_success, "execution_time": exec_time}

    total_time = time.time() - total_start_time

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY - 10M Row Dataset")
    print("=" * 80)

    successful_runs = 0
    for tech_name, result in results.items():
        status = "SUCCESS" if result["success"] else "FAILED"
        print(
            f"{tech_name.upper():10} | {status:12} | "
            f"{result['execution_time']:8.2f}s"
        )
        if result["success"]:
            successful_runs += 1

    print("-" * 80)
    print(f"Successfully completed: {successful_runs}/5 technologies")
    print(f"Total benchmark time: {total_time:.2f} seconds")

    # List generated files
    print("\nGenerated Performance Metrics Files:")
    results_dir = Path("../../../results")
    for file in results_dir.glob("performance_metrics_*_10m.json"):
        print(f"  - {file.name}")

    print(
        "\nUse create_simple_charts.py or visualize_results.py to " "analyze results!"
    )  # noqa: E501

    return successful_runs >= 3  # Consider success if at least 3/5 work


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(Path(__file__).parent)

    success = main()
    sys.exit(0 if success else 1)
