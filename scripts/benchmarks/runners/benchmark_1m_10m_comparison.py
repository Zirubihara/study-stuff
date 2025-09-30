#!/usr/bin/env python3
"""
Run benchmarks for all 5 libraries on 1M and 10M datasets only
"""

import os
import sys

sys.path.append(".")

# Import all processors from all.py
from all import (
    DaskDataProcessor,
    PandasDataProcessor,
    PolarsDataProcessor,
    PyArrowDataProcessor,
    SparkDataProcessor,
    cleanup_memory,
    process_implementation,
)


def run_1m_10m_benchmark():
    """Run benchmarks on 1M and 10M datasets for all libraries"""

    # Dataset configurations for 1M and 10M only
    dataset_configs = [
        {"rows": 1000000, "name": "1M", "file": "../data/benchmark_1m.csv"},
        {"rows": 10000000, "name": "10M", "file": "../data/benchmark_10m.csv"},
    ]

    # Library configurations
    libraries = ["pandas", "polars", "pyarrow", "dask", "spark"]

    # Number of runs for statistical reliability
    num_runs = 3  # Reduced for faster execution
    warmup_runs = 1

    all_results = {}

    print(f"{'=' * 80}")
    print("STARTING 1M and 10M BENCHMARK COMPARISON")
    print(f"{'=' * 80}")
    print(f"Libraries: {', '.join(libraries)}")
    print(f"Datasets: 1M and 10M rows")
    print(f"Runs per test: {num_runs} (plus {warmup_runs} warmup)")
    print(f"{'=' * 80}")

    for config in dataset_configs:
        print(f"\nProcessing {config['name']} dataset ({config['rows']:,} rows)...")
        print(f"File: {config['file']}")

        # Check if dataset exists
        if not os.path.exists(config["file"]):
            print(f"WARNING: Dataset {config['file']} not found, skipping...")
            continue

        dataset_results = {}

        for lib_name in libraries:
            print(f"\n--- Testing {lib_name.upper()} on {config['name']} dataset ---")

            try:
                # Create processor instance
                if lib_name == "pandas":
                    processor = PandasDataProcessor(config["file"])
                elif lib_name == "polars":
                    processor = PolarsDataProcessor(config["file"])
                elif lib_name == "pyarrow":
                    processor = PyArrowDataProcessor(config["file"])
                elif lib_name == "dask":
                    processor = DaskDataProcessor(config["file"])
                elif lib_name == "spark":
                    processor = SparkDataProcessor(config["file"])

                # Run benchmark
                result = process_implementation(
                    processor, f"{lib_name}_{config['name']}", num_runs, warmup_runs
                )
                if result:
                    dataset_results[lib_name] = result
                    print(f"SUCCESS: {lib_name} completed on {config['name']} dataset")

                # Cleanup
                del processor
                cleanup_memory()

            except Exception as e:
                print(f"ERROR: {lib_name} failed on {config['name']} dataset: {e}")
                continue

        all_results[config["name"]] = dataset_results
        print(f"\nCompleted {config['name']} dataset benchmark")

    # Print final summary
    print(f"\n{'=' * 80}")
    print("1M vs 10M BENCHMARK COMPLETED!")
    print(f"{'=' * 80}")

    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name} dataset results:")
        for lib_name, metrics in results.items():
            total_time = metrics.get("total_operation_time_seconds_mean", 0)
            memory_gb = metrics.get("memory_size_gb_mean", 0)
            print(f"  {lib_name:>8}: {total_time:6.2f}s, {memory_gb:5.2f}GB")

    print(f"\nPerformance files saved to:")
    print("  ../results/performance_metrics_<library>_<size>.json")

    return all_results


if __name__ == "__main__":
    # Set Java environment for Spark
    os.environ["JAVA_HOME"] = "/c/Program Files/Java/jdk-17"
    os.environ["PATH"] = "/c/Program Files/Java/jdk-17/bin:" + os.environ.get(
        "PATH", ""
    )

    results = run_1m_10m_benchmark()
