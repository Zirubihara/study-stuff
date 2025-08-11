#!/usr/bin/env python3
"""
Simple progress checker for the comprehensive benchmark.
"""

import os
import time
from pathlib import Path

def check_benchmark_progress():
    """Check the progress of the comprehensive benchmark."""
    
    print("BENCHMARK PROGRESS CHECKER")
    print("=" * 50)
    
    # Expected result file patterns for comprehensive benchmark
    expected_patterns = [
        "5M", "10M", "50M"  # Dataset sizes
    ]
    
    libraries = ["pandas", "polars", "arrow", "dask", "spark"]
    
    results_dir = Path("../results")
    
    print("CURRENT RESULTS STATUS:")
    print("-" * 30)
    
    total_expected = len(expected_patterns) * len(libraries)
    found_count = 0
    
    for lib in libraries:
        print(f"\n{lib.upper()}:")
        for pattern in expected_patterns:
            # Look for result files with dataset size patterns
            matching_files = list(results_dir.glob(f"*{lib}*{pattern}*"))
            if matching_files:
                latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
                mod_time = time.strftime("%H:%M:%S", time.localtime(latest_file.stat().st_mtime))
                print(f"  [✓] {pattern}: {latest_file.name} (modified {mod_time})")
                found_count += 1
            else:
                print(f"  [ ] {pattern}: Not completed yet")
    
    print(f"\nOVERALL PROGRESS:")
    print(f"   {found_count}/{total_expected} tests completed ({found_count/total_expected*100:.1f}%)")
    
    # Check if benchmark process is still running
    print(f"\nPROCESS STATUS:")
    try:
        # Simple check - look at recent file modifications
        recent_files = []
        for file in results_dir.glob("*.json"):
            mod_time = file.stat().st_mtime
            if time.time() - mod_time < 300:  # Modified in last 5 minutes
                recent_files.append(file.name)
        
        if recent_files:
            print(f"   [ACTIVE] {len(recent_files)} files modified recently")
            print(f"   Latest: {recent_files[-1] if recent_files else 'None'}")
        else:
            print(f"   [IDLE] No recent file activity (may be processing)")
            
    except Exception as e:
        print(f"   [ERROR] Could not check file activity: {e}")
    
    print(f"\nTOTAL RESULT FILES: {len(list(results_dir.glob('*.json')))}")

if __name__ == "__main__":
    check_benchmark_progress()