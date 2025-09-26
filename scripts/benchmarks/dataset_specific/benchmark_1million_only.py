#!/usr/bin/env python3
"""
Run benchmarks for all 5 libraries on 1M dataset only
"""

import os
import sys

sys.path.append(".")

# Create results directory if it doesn't exist
os.makedirs("../../results", exist_ok=True)

# Import all processors from unified benchmark
sys.path.append("../unified")
from benchmark_all_libraries import (
    DaskDataProcessor,
    PandasDataProcessor,
    PolarsDataProcessor,
    PyArrowDataProcessor,
    SparkDataProcessor,
    cleanup_memory,
    process_implementation,
)


def process_implementation_with_custom_path(
    processor, name: str, num_runs: int = 5, warmup_runs: int = 1
):
    """Process implementation with custom results path."""
    # Use the original process_implementation function
    result = process_implementation(processor, name, num_runs, warmup_runs)

    # Move the generated file to the results directory
    if result:
        old_path = f"performance_metrics_{name}.json"
        new_path = f"../../results/performance_metrics_{name}.json"

        if os.path.exists(old_path):
            # Move file to results directory
            import shutil

            shutil.move(old_path, new_path)
            print(f"Moved results to: {new_path}")

    return result


def run_1m_benchmark():
    """Run benchmarks on 1M dataset for all libraries"""

    # Only 1M dataset
    dataset_config = {"rows": 1000000, "name": "1M", "file": "../../data/benchmark_1m.csv"}

    # Library configurations
    libraries = ["pandas", "polars", "pyarrow", "dask", "spark"]

    # Reduced runs for faster testing
    num_runs = 1
    warmup_runs = 0

    print(f"{'=' * 60}")
    print("TESTING 1M DATASET BENCHMARK")
    print(f"{'=' * 60}")
    print(f"Libraries: {', '.join(libraries)}")
    print(f"Dataset: 1M rows")
    print(f"File: {dataset_config['file']}")
    print(f"Runs: {num_runs}")
    print(f"{'=' * 60}")

    # Check if dataset exists
    if not os.path.exists(dataset_config["file"]):
        print(f"ERROR: Dataset {dataset_config['file']} not found!")
        return

    results = {}
    for lib_name in libraries:
        print(f"\n--- Running {lib_name.upper()} ---")

        try:
            # Create processor instance
            if lib_name == "pandas":
                processor = PandasDataProcessor(dataset_config["file"])
            elif lib_name == "polars":
                processor = PolarsDataProcessor(dataset_config["file"])
            elif lib_name == "pyarrow":
                processor = PyArrowDataProcessor(dataset_config["file"])
            elif lib_name == "dask":
                processor = DaskDataProcessor(dataset_config["file"])
            elif lib_name == "spark":
                processor = SparkDataProcessor(dataset_config["file"])

            # Run benchmark
            result = process_implementation_with_custom_path(
                processor, f"{lib_name}_1M_test", num_runs, warmup_runs
            )
            if result:
                results[lib_name] = result
                total_time = result.get("total_operation_time_seconds_mean", 0)
                memory_gb = result.get("memory_size_gb_mean", 0)
                print(f"SUCCESS: {lib_name} - {total_time:.2f}s, {memory_gb:.3f}GB")

            # Cleanup
            del processor
            cleanup_memory()

        except Exception as e:
            print(f"ERROR: {lib_name} failed: {e}")
            continue

    # Print summary
    print(f"\n{'=' * 60}")
    print("1M DATASET RESULTS SUMMARY")
    print(f"{'=' * 60}")

    print(f"{'Library':<10} {'Time (s)':<10} {'Memory (GB)':<12}")
    print("-" * 35)

    for lib_name, metrics in results.items():
        total_time = metrics.get("total_operation_time_seconds_mean", 0)
        memory_gb = metrics.get("memory_size_gb_mean", 0)
        print(f"{lib_name:<10} {total_time:<10.2f} {memory_gb:<12.3f}")

    print(f"\nFiles saved to: ../../results/performance_metrics_<library>_1M_test.json")
    return results


if __name__ == "__main__":
    # Set Java environment for Spark
    os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-17"
    os.environ["PATH"] = "C:\\Program Files\\Java\\jdk-17\\bin;" + os.environ.get(
        "PATH", ""
    )

    results = run_1m_benchmark()
