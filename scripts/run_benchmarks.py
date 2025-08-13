#!/usr/bin/env python3
"""
Run benchmarks for all libraries on 1M and 10M datasets
"""

import os
import subprocess
import sys
import time


def run_benchmark(script, dataset_size, dataset_path):
    """Run a benchmark script with specified dataset"""
    print(f"\n{'='*60}")
    print(f"Running {script} with {dataset_size} dataset")
    print(f"Dataset: {dataset_path}")
    print("=" * 60)

    # Modify the script to use the specific dataset
    with open(script, "r") as f:
        content = f.read()

    # Replace dataset path in main function
    if "1m" in dataset_size.lower():
        new_content = content.replace(
            "csv_path = large_dataset", f'csv_path = "{dataset_path}"'
        )
        new_content = new_content.replace(
            "csv_path = massive_dataset", f'csv_path = "{dataset_path}"'
        )
        new_content = new_content.replace(
            "csv_path = medium_dataset", f'csv_path = "{dataset_path}"'
        )
    else:
        new_content = content.replace(
            "csv_path = large_dataset", f'csv_path = "{dataset_path}"'
        )
        new_content = new_content.replace(
            "csv_path = massive_dataset", f'csv_path = "{dataset_path}"'
        )
        new_content = new_content.replace(
            "csv_path = medium_dataset", f'csv_path = "{dataset_path}"'
        )

    # Write temp script
    temp_script = f"temp_{script}"
    with open(temp_script, "w") as f:
        f.write(new_content)

    try:
        # Run the script
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, temp_script], capture_output=True, text=True, timeout=600
        )
        end_time = time.time()

        if result.returncode == 0:
            print(f"✅ SUCCESS: {script} completed in {end_time - start_time:.2f}s")
            print("STDOUT:", result.stdout[-500:] if result.stdout else "No output")
        else:
            print(f"❌ ERROR: {script} failed")
            print(
                "STDERR:", result.stderr[-500:] if result.stderr else "No error output"
            )

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {script} timed out after 10 minutes")
    except Exception as e:
        print(f"💥 EXCEPTION: {str(e)}")
    finally:
        # Clean up temp script
        if os.path.exists(temp_script):
            os.remove(temp_script)


def main():
    """Run all benchmarks"""

    # Dataset configurations
    datasets = [
        ("1M", "../data/benchmark_1m.csv"),
        ("10M", "../data/benchmark_10m.csv"),
    ]

    # Scripts to run
    scripts = [
        "pandas-usage.py",
        "polars-usage.py",
        "pyarrow-usage.py",
        "dask-usage.py",
        "pyspark-usage.py",
    ]

    print("Starting comprehensive benchmarks...")

    for dataset_size, dataset_path in datasets:
        for script in scripts:
            if os.path.exists(script):
                run_benchmark(script, dataset_size, dataset_path)
            else:
                print(f"⚠️  WARNING: {script} not found, skipping...")

    print(f"\n{'='*60}")
    print("All benchmarks completed!")
    print("Check the ../results/ directory for performance metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()
