#!/usr/bin/env python3
"""
Create visualization charts from 5M benchmark results
"""
import matplotlib.pyplot as plt
import numpy as np

# Results from the benchmark
results = {
    'Pandas': {
        'load_time': 1.33,
        'clean_time': 0.06,
        'agg_time': 0.40,
        'sort_time': 0.82,
        'filter_time': 0.07,
        'total_time': 2.69,
        'memory_mb': 822.6
    },
    'Polars': {
        'load_time': 0.15,
        'clean_time': 0.00,
        'agg_time': 0.14,
        'sort_time': 0.21,
        'filter_time': 0.02,
        'total_time': 0.52,
        'memory_mb': 1664.8
    }
}

def create_performance_charts():
    """Create performance comparison charts"""

    # Set up the figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('5M Dataset Performance Comparison: Pandas vs Polars', fontsize=16, fontweight='bold')

    libraries = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e']  # Blue for Pandas, Orange for Polars

    # 1. Total Execution Time
    total_times = [results[lib]['total_time'] for lib in libraries]
    bars1 = ax1.bar(libraries, total_times, color=colors)
    ax1.set_title('Total Execution Time', fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, time in zip(bars1, total_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')

    # 2. Memory Usage
    memory_usage = [results[lib]['memory_mb'] for lib in libraries]
    bars2 = ax2.bar(libraries, memory_usage, color=colors)
    ax2.set_title('Memory Usage', fontweight='bold')
    ax2.set_ylabel('Memory (MB)')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mem in zip(bars2, memory_usage):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{mem:.1f} MB', ha='center', va='bottom', fontweight='bold')

    # 3. Operation Breakdown
    operations = ['load_time', 'clean_time', 'agg_time', 'sort_time', 'filter_time']
    op_labels = ['Load', 'Clean', 'Aggregate', 'Sort', 'Filter']

    x = np.arange(len(op_labels))
    width = 0.35

    pandas_times = [results['Pandas'][op] for op in operations]
    polars_times = [results['Polars'][op] for op in operations]

    bars3_1 = ax3.bar(x - width/2, pandas_times, width, label='Pandas', color=colors[0])
    bars3_2 = ax3.bar(x + width/2, polars_times, width, label='Polars', color=colors[1])

    ax3.set_title('Operation Breakdown', fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(op_labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars3_1, bars3_2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # 4. Speed Comparison (Polars vs Pandas speedup)
    speedups = []
    speed_labels = []

    for op in operations:
        pandas_time = results['Pandas'][op]
        polars_time = results['Polars'][op]
        if polars_time > 0:
            speedup = pandas_time / polars_time
            speedups.append(speedup)
            speed_labels.append(op.replace('_time', '').title())

    # Add total speedup
    total_speedup = results['Pandas']['total_time'] / results['Polars']['total_time']
    speedups.append(total_speedup)
    speed_labels.append('Total')

    bars4 = ax4.bar(speed_labels, speedups, color=['#2ca02c'] * len(speedups))
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax4.set_title('Polars Speedup Factor vs Pandas', fontweight='bold')
    ax4.set_ylabel('Speedup Factor (x times faster)')
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend()

    # Add value labels on bars
    for bar, speedup in zip(bars4, speedups):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('charts/5m_benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("Chart saved as: charts/5m_benchmark_comparison.png")

def create_summary_table():
    """Create a summary table visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for table
    table_data = []
    headers = ['Library', 'Total Time', 'Load Time', 'Agg Time', 'Sort Time', 'Memory (MB)', 'Winner']

    for lib in ['Pandas', 'Polars']:
        data = results[lib]
        row = [
            lib,
            f"{data['total_time']:.2f}s",
            f"{data['load_time']:.2f}s",
            f"{data['agg_time']:.2f}s",
            f"{data['sort_time']:.2f}s",
            f"{data['memory_mb']:.1f}",
            ""
        ]
        table_data.append(row)

    # Determine winners
    if results['Polars']['total_time'] < results['Pandas']['total_time']:
        table_data[1][6] = "Fastest Overall"
    else:
        table_data[0][6] = "Fastest Overall"

    if results['Pandas']['memory_mb'] < results['Polars']['memory_mb']:
        table_data[0][6] += (" + " if table_data[0][6] else "") + "Lower Memory"
    else:
        table_data[1][6] += (" + " if table_data[1][6] else "") + "Lower Memory"

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f7f7f7')

    plt.title('5M Dataset Benchmark Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('charts/5m_benchmark_table.png', dpi=300, bbox_inches='tight')
    print("Table saved as: charts/5m_benchmark_table.png")

def main():
    """Generate all visualizations"""
    print("Creating 5M benchmark visualizations...")

    # Ensure charts directory exists
    import os
    os.makedirs('charts', exist_ok=True)

    try:
        create_performance_charts()
        create_summary_table()
        print("\nVisualization Summary:")
        print("- Polars is 5.2x faster overall than Pandas")
        print("- Polars is 8.9x faster at loading data")
        print("- Polars uses 2x more memory but provides significant speed gains")
        print("- All charts saved to charts/ directory")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main()