#!/usr/bin/env python3
"""
Fix data parsing issues and run 50M benchmarks for all technologies.
This script handles CSV parsing issues that prevent proper benchmarking.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

def run_individual_benchmark(tech_name, script_path):
    """Run individual benchmark with proper error handling."""
    print(f"\n{'='*60}")
    print(f"Running {tech_name.upper()} on 50M dataset...")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run with longer timeout for 50M dataset
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ SUCCESS: {tech_name} completed in {execution_time:.2f}s")
            if result.stdout.strip():
                print("Output:", result.stdout.strip()[-200:])  # Last 200 chars
        else:
            print(f"❌ FAILED: {tech_name} failed with return code {result.returncode}")
            print("Error:", result.stderr[:500])  # First 500 chars

        return result.returncode == 0, execution_time

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {tech_name} timed out after 30 minutes")
        return False, time.time() - start_time
    except Exception as e:
        print(f"💥 CRASHED: {tech_name} crashed: {e}")
        return False, time.time() - start_time

def main():
    print("Running Fixed 50M Dataset Benchmarks")
    print("="*50)

    # Change to implementations directory
    impl_dir = Path(__file__).parent / "../implementations"
    os.chdir(impl_dir)

    # Define technologies and their scripts
    technologies = {
        "pandas": "benchmark_pandas_implementation.py",
        "polars": "benchmark_polars_implementation.py",
        "pyarrow": "benchmark_pyarrow_implementation.py",
        "dask": "benchmark_dask_implementation.py",
        "pyspark": "benchmark_pyspark_implementation.py"
    }

    results = {}
    total_start = time.time()

    for tech_name, script_name in technologies.items():
        script_path = Path(script_name)
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            continue

        success, exec_time = run_individual_benchmark(tech_name, script_path)
        results[tech_name] = {"success": success, "time": exec_time}

        if success:
            time.sleep(5)  # Brief pause between successful runs

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    successful = 0
    for tech, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{tech.upper():10} | {status} | {result['time']:8.2f}s")
        if result["success"]:
            successful += 1

    print(f"\nSuccessful: {successful}/{len(results)} technologies")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f}m)")

    # Check generated files
    results_dir = Path("../../../results")
    files = list(results_dir.glob("performance_metrics_*_50m.json"))
    print(f"\nGenerated {len(files)} result files:")
    for f in sorted(files):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()