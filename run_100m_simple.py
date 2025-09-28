#!/usr/bin/env python3
"""
Simple 100M benchmark runner - direct approach without complex orchestration
"""

import sys
import time
import json
import psutil
import gc
from pathlib import Path

# Add unified benchmarks to path
sys.path.append("scripts/benchmarks/unified")
from benchmark_all_libraries import (
    PandasDataProcessor,
    PolarsDataProcessor,
    PyArrowDataProcessor,
    DaskDataProcessor,
    SparkDataProcessor
)

def run_single_benchmark(processor_class, name, csv_path):
    """Run benchmark for a single technology."""
    print(f"\n{'='*60}")
    print(f"Running {name.upper()} on 100M dataset...")
    print(f"{'='*60}")

    try:
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**3)

        start_time = time.time()

        # Create processor and run benchmark
        processor = processor_class(csv_path)
        result = processor.process_data()

        end_time = time.time()
        total_time = end_time - start_time

        # Memory after
        mem_after = process.memory_info().rss / (1024**3)
        mem_used = mem_after - mem_before

        # Save results
        results = {
            "technology": name,
            "total_time_seconds": total_time,
            "memory_used_gb": mem_used,
            "timestamp": time.time(),
            "dataset_rows": 100000000,
            "status": "success"
        }

        output_file = f"results/simple_100m_{name}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"SUCCESS: {name} - {total_time:.2f}s, {mem_used:.2f}GB")
        print(f"Results saved to: {output_file}")

        # Cleanup
        del processor, result
        gc.collect()

        return results

    except Exception as e:
        print(f"ERROR: {name} failed: {e}")

        # Save error results
        error_results = {
            "technology": name,
            "status": "failed",
            "error": str(e),
            "timestamp": time.time()
        }

        output_file = f"results/simple_100m_{name}_error.json"
        with open(output_file, 'w') as f:
            json.dump(error_results, f, indent=2)

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
    print(f"WARNING: This will use significant memory and take 10-30 minutes")

    # Technologies to test
    technologies = [
        (PolarsDataProcessor, "polars"),
        (PyArrowDataProcessor, "pyarrow"),
        (PandasDataProcessor, "pandas"),
        (DaskDataProcessor, "dask"),
        (SparkDataProcessor, "pyspark")
    ]

    results = {}

    for processor_class, name in technologies:
        result = run_single_benchmark(processor_class, name, csv_path)
        if result:
            results[name] = result

        # Give system a moment to recover
        time.sleep(5)

    # Summary
    print(f"\n{'='*60}")
    print("100M BENCHMARK SUMMARY")
    print(f"{'='*60}")

    if results:
        print(f"{'Technology':<12} {'Time (s)':<10} {'Memory (GB)':<12}")
        print("-" * 40)

        # Sort by time
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_time_seconds'])

        for name, data in sorted_results:
            time_s = data['total_time_seconds']
            mem_gb = data['memory_used_gb']
            print(f"{name:<12} {time_s:<10.2f} {mem_gb:<12.2f}")
    else:
        print("No technologies completed successfully")

    print(f"\nCompleted: {len(results)}/5 technologies")

if __name__ == "__main__":
    main()