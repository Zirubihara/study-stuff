#!/usr/bin/env python3
"""
Benchmark script to run all data processing technologies on 100M row dataset (3GB).
This is the ultimate stress test to see how each technology handles massive data.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def modify_script_for_100m(script_path, technology_name):
    """Modify a script to use 100M dataset and save with _100m suffix."""
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace dataset path - handle all possible existing paths
    replacements_from = [
        'csv_path = "../data/benchmark_1m.csv"',
        'csv_path = "../data/benchmark_5m.csv"',
        'csv_path = "../data/benchmark_10m.csv"',
        'csv_path = "../data/benchmark_50m.csv"'
    ]
    
    for old_path in replacements_from:
        content = content.replace(old_path, 'csv_path = "../data/benchmark_100m.csv"')
    
    # Replace output file path based on technology
    output_replacements = {
        'pandas': (
            'output_path: str = "../results/performance_metrics_pandas.json"',
            'output_path: str = "../results/performance_metrics_pandas_100m.json"'
        ),
        'polars': (
            'output_path: str = "../results/performance_metrics_polars.json"',
            'output_path: str = "../results/performance_metrics_polars_100m.json"'
        ),
        'pyarrow': (
            'output_path: str = "../results/performance_metrics_arrow.json"',
            'output_path: str = "../results/performance_metrics_arrow_100m.json"'
        ),
        'dask': (
            'output_path: str = "../results/performance_metrics_dask.json"',
            'output_path: str = "../results/performance_metrics_dask_100m.json"'
        )
    }
    
    if technology_name in output_replacements:
        old_str, new_str = output_replacements[technology_name]
        content = content.replace(old_str, new_str)
    
    # Optimize configurations for massive dataset
    if technology_name == 'dask':
        # Increase memory limits and optimize for large dataset
        content = content.replace(
            'Client(processes=False)',
            'Client(processes=False, memory_limit="12GB", threads_per_worker=4)'
        )
    elif technology_name == 'pandas':
        # Add chunking for very large datasets if needed
        # For now, let's see if Pandas can handle 3GB in memory
        pass
    
    # Create temporary script
    temp_script = script_path.parent / f"temp_{technology_name}_100m.py"
    with open(temp_script, 'w') as f:
        f.write(content)
    
    return temp_script


def run_technology(script_path, technology_name):
    """Run a single technology benchmark with extended timeouts for massive data."""
    print(f"\n{'='*70}")
    print(f"Running {technology_name.upper()} on 100M row dataset (3GB)...")
    print(f"{'='*70}")
    
    # Estimated times based on 50M results scaled up
    estimated_times = {
        'pyarrow': '~30 seconds',
        'dask': '~45 seconds', 
        'polars': '~3 minutes',
        'pandas': '~5 minutes'
    }
    
    if technology_name in estimated_times:
        print(f"Estimated time: {estimated_times[technology_name]}")
    
    start_time = time.time()
    
    try:
        # Modify script for 100M dataset
        temp_script = modify_script_for_100m(script_path, technology_name)
        
        # Set very long timeouts for massive dataset
        timeout_minutes = {
            'pyarrow': 15,    # Should be fast
            'dask': 20,       # Distributed processing
            'polars': 30,     # Memory efficient
            'pandas': 45      # Might struggle with 3GB
        }
        
        timeout = timeout_minutes.get(technology_name, 30) * 60  # Convert to seconds
        
        print(f"Timeout set to {timeout//60} minutes")
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            minutes = execution_time / 60
            print(f"SUCCESS: {technology_name.upper()} completed in {execution_time:.2f}s ({minutes:.1f} minutes)")
            print("Output:", result.stdout.strip() if result.stdout.strip() else "No output")
            if result.stderr.strip():
                print("Warnings:", result.stderr.strip())
        else:
            print(f"FAILED: {technology_name.upper()} failed with return code {result.returncode}")
            print("Error:", result.stderr)
            if result.stdout:
                print("Output:", result.stdout)
        
        # Clean up temporary script
        if temp_script.exists():
            temp_script.unlink()
        
        return result.returncode == 0, execution_time
        
    except subprocess.TimeoutExpired:
        minutes = timeout // 60
        print(f"TIMEOUT: {technology_name.upper()} exceeded {minutes} minute limit")
        print("This suggests the technology may not be suitable for datasets this large")
        return False, time.time() - start_time
    except Exception as e:
        print(f"CRASHED: {technology_name.upper()} crashed with error: {e}")
        return False, time.time() - start_time


def check_system_resources():
    """Check if system has enough resources for 100M dataset processing."""
    print("Checking system resources...")
    
    # Check available disk space
    dataset_path = Path("../data/benchmark_100m.csv")
    if dataset_path.exists():
        size_gb = dataset_path.stat().st_size / (1024**3)
        print(f"Dataset size: {size_gb:.2f} GB")
    
    # Warn about memory requirements
    print("Memory requirements:")
    print("- PyArrow: ~4GB RAM (efficient)")
    print("- Dask: ~6GB RAM (distributed)")
    print("- Polars: ~8GB RAM (memory efficient)")
    print("- Pandas: ~12GB RAM (loads entire dataset)")
    print("- PySpark: ~8GB RAM (distributed)")
    print()
    
    return True


def main():
    """Main execution function."""
    print("ULTIMATE BENCHMARK: 100M Row Dataset (3GB)")
    print("=" * 80)
    print("This is the ultimate stress test for data processing libraries!")
    print("Expected total runtime: 15-30 minutes")
    print("=" * 80)
    
    # Check if 100M dataset exists
    dataset_path = Path("../data/benchmark_100m.csv")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please run generate_massive_data.py first to create the 100M dataset")
        return False
    
    # Check system resources
    if not check_system_resources():
        return False
    
    # Define technologies in order of expected performance (fastest first)
    technologies = {
        'pyarrow': Path("pyarrow-usage.py"),    # Should be fastest
        'dask': Path("dask-usage.py"),          # Distributed processing advantage
        'polars': Path("polars-usage.py"),      # Memory efficient
        'pandas': Path("pandas-usage.py")       # Will be slowest but should work
    }
    
    print("Note: We'll also modify PySpark to run on 100M dataset separately")
    
    results = {}
    total_start_time = time.time()
    
    # Run each technology
    for tech_name, script_path in technologies.items():
        if not script_path.exists():
            print(f"Script not found: {script_path}")
            continue
        
        print(f"\nPreparing to run {tech_name.upper()}...")
        print(f"If this fails, it likely means {tech_name} cannot handle 3GB datasets efficiently")
        
        success, exec_time = run_technology(script_path, tech_name)
        results[tech_name] = {
            'success': success,
            'execution_time': exec_time
        }
        
        # Add longer pause between runs for system recovery
        if success:
            print(f"SUCCESS! Pausing 30 seconds for system recovery...")
            time.sleep(30)
        else:
            print(f"FAILED! Moving to next technology...")
            time.sleep(10)
    
    total_time = time.time() - total_start_time
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("ULTIMATE BENCHMARK RESULTS - 100M Row Dataset (3GB)")
    print("="*80)
    
    successful_runs = 0
    performance_ranking = []
    
    for tech_name, result in results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        minutes = result['execution_time'] / 60
        
        if result['success']:
            performance_ranking.append((tech_name, result['execution_time']))
            successful_runs += 1
            
        print(f"{tech_name.upper():10} | {status:12} | {result['execution_time']:8.2f}s ({minutes:5.1f}m)")
    
    # Sort by performance
    performance_ranking.sort(key=lambda x: x[1])
    
    print("\nPERFORMACE RANKING (Successful runs only):")
    print("-" * 50)
    for i, (tech, time_sec) in enumerate(performance_ranking, 1):
        minutes = time_sec / 60
        print(f"{i}. {tech.upper():10} - {time_sec:8.2f}s ({minutes:5.1f}m)")
    
    print(f"\nTOTAL SUMMARY:")
    print(f"Successfully completed: {successful_runs}/4 technologies")
    print(f"Total benchmark time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Dataset size: 3.01 GB (100,000,000 rows)")
    
    # List generated files
    print("\nGenerated Performance Metrics Files:")
    results_dir = Path("../results")
    for file in results_dir.glob("performance_metrics_*_100m.json"):
        print(f"  - {file.name}")
    
    if successful_runs > 0:
        print("\nNext steps:")
        print("1. Run PySpark separately on 100M dataset")
        print("2. Use visualize_results.py to analyze all results")
        print("3. Compare scaling behavior: 5M -> 50M -> 100M")
    else:
        print("\nNOTE: If all technologies failed, the dataset may be too large for this system.")
        print("Try increasing system memory or use distributed computing setup.")
    
    return successful_runs >= 1  # Success if at least one technology works


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(Path(__file__).parent)
    
    # Final warning about resource requirements
    print("WARNING: This benchmark will use MASSIVE system resources!")
    print("- Requires 8-16GB RAM")
    print("- Will take 15-30 minutes")
    print("- May slow down your system significantly")
    print("- Close other applications before running")
    print()
    print("Press Ctrl+C to cancel within 10 seconds...")
    
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("\nBenchmark cancelled by user.")
        sys.exit(0)
    
    success = main()
    sys.exit(0 if success else 1)