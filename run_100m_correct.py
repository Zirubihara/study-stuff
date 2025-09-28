#!/usr/bin/env python3
"""
Run 100M benchmark using the correct process_implementation API
"""

import sys
import time
import os
from pathlib import Path

# Add unified benchmarks to path
sys.path.append("scripts/benchmarks/unified")
from benchmark_all_libraries import (
    PandasDataProcessor,
    PolarsDataProcessor,
    PyArrowDataProcessor,
    DaskDataProcessor,
    SparkDataProcessor,
    process_implementation,
    cleanup_memory
)

def run_technology_benchmark(processor_class, name, csv_path):
    """Run benchmark for a single technology using the correct API."""
    print(f"\n{'='*60}")
    print(f"Running {name.upper()} on 100M dataset...")
    print(f"{'='*60}")

    try:
        # Create processor
        processor = processor_class(csv_path)

        # Run benchmark using the correct API (1 run, no warmup for speed)
        result = process_implementation(processor, f"{name}_100m_new", 1, 0)

        if result:
            total_time = result.get("total_operation_time_seconds_mean", 0)
            memory_gb = result.get("memory_size_gb_mean", 0)
            print(f"SUCCESS: {name} - {total_time:.2f}s, {memory_gb:.3f}GB")

            # Move result file to have consistent naming
            old_file = f"performance_metrics_{name}_100m_new.json"
            new_file = f"results/performance_metrics_{name}_100m_new.json"

            if os.path.exists(old_file):
                import shutil
                shutil.move(old_file, new_file)
                print(f"Results saved to: {new_file}")

        # Cleanup
        del processor
        cleanup_memory()

        return result

    except Exception as e:
        print(f"ERROR: {name} failed: {e}")
        return None

def main():
    """Run 100M benchmarks for all technologies."""

    # Check dataset exists
    csv_path = "data/benchmark_100m.csv"
    if not Path(csv_path).exists():
        print(f"ERROR: Dataset not found: {csv_path}")
        return

    dataset_size = Path(csv_path).stat().st_size / (1024**3)
    print(f"Starting 100M Benchmark (Dataset: {dataset_size:.2f}GB)")
    print(f"WARNING: This will use significant memory and take 15-30 minutes")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Technologies to test (starting with fastest from previous results)
    technologies = [
        (PolarsDataProcessor, "polars"),
        (PyArrowDataProcessor, "pyarrow"),
        (PandasDataProcessor, "pandas"),
        (DaskDataProcessor, "dask"),
        # Skip PySpark due to Java version issues
    ]

    results = {}
    start_total = time.time()

    for processor_class, name in technologies:
        print(f"\n⏰ Starting {name} benchmark...")

        result = run_technology_benchmark(processor_class, name, csv_path)
        if result:
            results[name] = result

        # Give system a moment to recover
        print("Pausing 10 seconds before next technology...")
        time.sleep(10)

    total_time = time.time() - start_total

    # Summary
    print(f"\n{'='*60}")
    print("100M BENCHMARK SUMMARY")
    print(f"{'='*60}")

    if results:
        print(f"{'Technology':<12} {'Time (s)':<10} {'Memory (GB)':<12}")
        print("-" * 40)

        # Sort by time
        for name, data in results.items():
            total_time_s = data.get("total_operation_time_seconds_mean", 0)
            memory_gb = data.get("memory_size_gb_mean", 0)
            print(f"{name:<12} {total_time_s:<10.2f} {memory_gb:<12.3f}")
    else:
        print("No technologies completed successfully")

    print(f"\nCompleted: {len(results)}/4 technologies")
    print(f"Total benchmark time: {total_time/60:.1f} minutes")

    # Show winner
    if results:
        fastest = min(results.items(), key=lambda x: x[1].get("total_operation_time_seconds_mean", float('inf')))
        print(f"🏆 Fastest: {fastest[0]} ({fastest[1].get('total_operation_time_seconds_mean', 0):.2f}s)")

if __name__ == "__main__":
    main()