#!/usr/bin/env python3
"""Test Pandas on 100M dataset"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "unified"))
from benchmark_all_libraries import PandasDataProcessor, process_implementation, cleanup_memory

dataset_file = "../../../data/benchmark_100m.csv"

print("Testing Pandas on 100M dataset...")
print(f"File: {dataset_file}")

if not os.path.exists(dataset_file):
    print(f"ERROR: Dataset {dataset_file} not found!")
    sys.exit(1)

try:
    processor = PandasDataProcessor(dataset_file)
    result = process_implementation(processor, "pandas_100M", num_runs=1, warmup_runs=0)

    if result:
        import shutil
        old_path = "performance_metrics_pandas_100M.json"
        new_path = "../../../results/performance_metrics_pandas_100M.json"
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            print(f"SUCCESS! Results saved to: {new_path}")
            print(f"Total time: {result.get('total_operation_time_seconds', 0):.2f}s")
            print(f"Memory: {result.get('memory_size_gb', 0):.2f}GB")
    else:
        print("FAILED: No result returned")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    cleanup_memory()
