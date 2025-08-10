"""
Generate simple, easy-to-read charts for thesis - one chart per concept.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set clean, professional styling
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

def load_data():
    """Load benchmark data from JSON files."""
    data = {}
    json_files = list(Path('.').glob('performance_metrics_*_*.json'))
    
    for file_path in json_files:
        filename = file_path.stem
        parts = filename.split('_')
        if len(parts) >= 3:
            library = parts[2]
            size = parts[3]
            
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            if library not in data:
                data[library] = {}
            data[library][size] = json_data
    
    return data

def chart_1_total_execution_time():
    """Chart 1: Total execution time comparison - simple bar chart."""
    data = load_data()
    libraries = list(data.keys())
    sizes = ['100K', '500K']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, size in enumerate(sizes):
        ax = ax1 if i == 0 else ax2
        
        lib_names = []
        total_times = []
        std_devs = []
        
        for library in libraries:
            if size in data[library]:
                # Calculate total time
                operations = ['loading', 'cleaning', 'aggregation', 'sorting', 'filtering', 'correlation']
                total_time = sum(data[library][size].get(f"{op}_time_mean", 0) for op in operations)
                total_std = np.sqrt(sum(data[library][size].get(f"{op}_time_std", 0)**2 for op in operations))
                
                lib_names.append(library.capitalize())
                total_times.append(total_time)
                std_devs.append(total_std)
        
        bars = ax.bar(lib_names, total_times, yerr=std_devs, capsize=5, 
                     color=colors[:len(lib_names)], alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_title(f'{size} Dataset', fontweight='bold')
        ax.set_ylabel('Total Execution Time (seconds)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time in zip(bars, total_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Total Execution Time Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('01_total_execution_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 01_total_execution_time.png")

def chart_2_loading_time_detailed():
    """Chart 2: Loading time only - focused comparison."""
    data = load_data()
    libraries = list(data.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(libraries))
    width = 0.35
    
    means_100k = []
    stds_100k = []
    means_500k = []
    stds_500k = []
    
    for library in libraries:
        # 100K data
        if '100K' in data[library]:
            means_100k.append(data[library]['100K'].get('loading_time_mean', 0))
            stds_100k.append(data[library]['100K'].get('loading_time_std', 0))
        else:
            means_100k.append(0)
            stds_100k.append(0)
        
        # 500K data
        if '500K' in data[library]:
            means_500k.append(data[library]['500K'].get('loading_time_mean', 0))
            stds_500k.append(data[library]['500K'].get('loading_time_std', 0))
        else:
            means_500k.append(0)
            stds_500k.append(0)
    
    bars1 = ax.bar(x - width/2, means_100k, width, yerr=stds_100k, capsize=5,
                   label='100K rows', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, means_500k, width, yerr=stds_500k, capsize=5,
                   label='500K rows', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Library')
    ax.set_ylabel('Loading Time (seconds)')
    ax.set_title('CSV Loading Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([lib.capitalize() for lib in libraries])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('02_loading_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 02_loading_time_comparison.png")

def chart_3_memory_usage_simple():
    """Chart 3: Peak memory usage - simple comparison."""
    data = load_data()
    libraries = list(data.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(libraries))
    width = 0.35
    
    peak_100k = []
    peak_500k = []
    
    for library in libraries:
        operations = ['loading', 'cleaning', 'aggregation', 'sorting', 'filtering', 'correlation']
        
        # 100K peak memory
        if '100K' in data[library]:
            peak_mem = max(data[library]['100K'].get(f"{op}_memory_mean", 0) for op in operations)
            peak_100k.append(peak_mem)
        else:
            peak_100k.append(0)
        
        # 500K peak memory
        if '500K' in data[library]:
            peak_mem = max(data[library]['500K'].get(f"{op}_memory_mean", 0) for op in operations)
            peak_500k.append(peak_mem)
        else:
            peak_500k.append(0)
    
    bars1 = ax.bar(x - width/2, peak_100k, width, label='100K rows', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, peak_500k, width, label='500K rows', 
                   color='#f39c12', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Library')
    ax.set_ylabel('Peak Memory Usage (MB)')
    ax.set_title('Peak Memory Usage Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([lib.capitalize() for lib in libraries])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}MB', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('03_memory_usage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 03_memory_usage_comparison.png")

def chart_4_aggregation_performance():
    """Chart 4: Aggregation operation focus."""
    data = load_data()
    libraries = list(data.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(libraries))
    width = 0.35
    
    agg_100k = []
    agg_500k = []
    std_100k = []
    std_500k = []
    
    for library in libraries:
        # 100K aggregation
        if '100K' in data[library]:
            agg_100k.append(data[library]['100K'].get('aggregation_time_mean', 0))
            std_100k.append(data[library]['100K'].get('aggregation_time_std', 0))
        else:
            agg_100k.append(0)
            std_100k.append(0)
        
        # 500K aggregation
        if '500K' in data[library]:
            agg_500k.append(data[library]['500K'].get('aggregation_time_mean', 0))
            std_500k.append(data[library]['500K'].get('aggregation_time_std', 0))
        else:
            agg_500k.append(0)
            std_500k.append(0)
    
    bars1 = ax.bar(x - width/2, agg_100k, width, yerr=std_100k, capsize=5,
                   label='100K rows', color='#9b59b6', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, agg_500k, width, yerr=std_500k, capsize=5,
                   label='500K rows', color='#34495e', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Library')
    ax.set_ylabel('Aggregation Time (seconds)')
    ax.set_title('Group-By Aggregation Performance', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([lib.capitalize() for lib in libraries])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('04_aggregation_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 04_aggregation_performance.png")

def chart_5_scalability_lines():
    """Chart 5: Simple scalability - how time increases with data size."""
    data = load_data()
    libraries = list(data.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sizes = [100000, 500000]
    size_labels = ['100K', '500K']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, library in enumerate(libraries):
        total_times = []
        
        for size_label in size_labels:
            if size_label in data[library]:
                operations = ['loading', 'cleaning', 'aggregation', 'sorting', 'filtering', 'correlation']
                total_time = sum(data[library][size_label].get(f"{op}_time_mean", 0) for op in operations)
                total_times.append(total_time)
            else:
                total_times.append(0)
        
        if len(total_times) == 2 and total_times[1] > 0:
            ax.plot(sizes, total_times, marker='o', linewidth=3, markersize=8,
                   label=library.capitalize(), color=colors[i % len(colors)])
            
            # Add value labels
            for x, y in zip(sizes, total_times):
                ax.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontweight='bold')
    
    ax.set_xlabel('Dataset Size (rows)')
    ax.set_ylabel('Total Execution Time (seconds)')
    ax.set_title('Performance Scalability', fontsize=16, fontweight='bold')
    ax.set_xticks(sizes)
    ax.set_xticklabels(size_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('05_scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 05_scalability_analysis.png")

def chart_6_operation_ranking():
    """Chart 6: Ranking libraries by operation - who wins what."""
    data = load_data()
    operations = ['loading', 'cleaning', 'aggregation', 'sorting', 'filtering', 'correlation']
    
    # Use 500K dataset for ranking
    size = '500K'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, op in enumerate(operations):
        ax = axes[i]
        
        lib_times = []
        lib_names = []
        lib_stds = []
        
        for library in data.keys():
            if size in data[library]:
                time_mean = data[library][size].get(f"{op}_time_mean", 0)
                time_std = data[library][size].get(f"{op}_time_std", 0)
                if time_mean > 0:
                    lib_times.append(time_mean)
                    lib_names.append(library.capitalize())
                    lib_stds.append(time_std)
        
        # Sort by performance (fastest first)
        sorted_data = sorted(zip(lib_times, lib_names, lib_stds))
        lib_times, lib_names, lib_stds = zip(*sorted_data) if sorted_data else ([], [], [])
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']  # Green = best, Red = worst
        bars = ax.bar(lib_names, lib_times, yerr=lib_stds, capsize=4,
                     color=colors[:len(lib_names)], alpha=0.8, edgecolor='black')
        
        ax.set_title(f'{op.capitalize()} Operation', fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, time in zip(bars, lib_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Add ranking badges
        if lib_names:
            ax.text(0.02, 0.95, 'FASTEST', transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.7),
                   verticalalignment='top')
    
    plt.suptitle('Individual Operation Performance Ranking (500K Dataset)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('06_operation_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 06_operation_rankings.png")

def generate_simple_charts():
    """Generate all simple, focused charts."""
    print("Generating simple, easy-to-read charts for your thesis...")
    print("=" * 60)
    
    chart_1_total_execution_time()
    chart_2_loading_time_detailed()
    chart_3_memory_usage_simple()
    chart_4_aggregation_performance()
    chart_5_scalability_lines()
    chart_6_operation_ranking()
    
    print("=" * 60)
    print("ALL CHARTS GENERATED SUCCESSFULLY!")
    print("\nThesis-ready visualizations:")
    print("01_total_execution_time.png - Overall performance comparison")
    print("02_loading_time_comparison.png - CSV loading performance")
    print("03_memory_usage_comparison.png - Memory efficiency")
    print("04_aggregation_performance.png - Group-by operations")
    print("05_scalability_analysis.png - How performance scales")
    print("06_operation_rankings.png - Who wins each operation")
    print("\nReady for your university thesis!")

if __name__ == "__main__":
    generate_simple_charts()