"""
Visualization of scikit-learn anomaly detection results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(results_file):
    """Load results from JSON"""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_comparison_charts(results, output_dir="charts"):
    """Create comparison visualizations"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if_results = results['isolation_forest']
    lof_results = results['local_outlier_factor']

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Scikit-learn Anomaly Detection Comparison\n(1M Sample from Japanese Trade Data)',
                 fontsize=16, fontweight='bold')

    # 1. Training Time Comparison
    ax = axes[0, 0]
    models = ['Isolation\nForest', 'LOF']
    train_times = [if_results['training_time'], lof_results['training_time']]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(models, train_times, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax.set_title('Training Time Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontweight='bold')

    # 2. Inference Speed Comparison
    ax = axes[0, 1]
    inference_speeds = [if_results['inference_speed'], lof_results['inference_speed']]
    bars = ax.bar(models, inference_speeds, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Inference Speed (samples/sec)', fontweight='bold')
    ax.set_title('Inference Speed Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 3. Anomalies Detected
    ax = axes[1, 0]
    anomalies = [if_results['n_anomalies'], lof_results['n_anomalies']]
    bars = ax.bar(models, anomalies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Anomalies Detected', fontweight='bold')
    ax.set_title('Anomalies Detected (1% contamination)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')

    # 4. Memory Usage
    ax = axes[1, 1]
    memory = [if_results['memory_usage_gb'], lof_results['memory_usage_gb']]
    bars = ax.bar(models, memory, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Memory Usage (GB)', fontweight='bold')
    ax.set_title('Memory Usage Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} GB',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_file = f"{output_dir}/sklearn_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Chart saved: {output_file}")
    plt.close()

    # Create a summary metrics chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Metrics table
    metrics_data = [
        ['Training Time', f"{if_results['training_time']:.2f}s", f"{lof_results['training_time']:.2f}s"],
        ['Inference Time', f"{if_results['inference_time']:.2f}s", f"{lof_results['inference_time']:.2f}s"],
        ['Memory Usage', f"{if_results['memory_usage_gb']:.2f} GB", f"{lof_results['memory_usage_gb']:.2f} GB"],
        ['Anomalies Detected', f"{if_results['n_anomalies']:,}", f"{lof_results['n_anomalies']:,}"],
        ['Anomaly Rate', f"{if_results['anomaly_rate']:.2f}%", f"{lof_results['anomaly_rate']:.2f}%"],
        ['Inference Speed', f"{if_results['inference_speed']:,.0f} samples/s", f"{lof_results['inference_speed']:,.0f} samples/s"]
    ]

    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=metrics_data,
                     colLabels=['Metric', 'Isolation Forest', 'Local Outlier Factor'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.35, 0.35])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(metrics_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('Scikit-learn Anomaly Detection: Complete Performance Metrics\n(1M Sample from Japanese Trade Data)',
              fontsize=14, fontweight='bold', pad=20)

    output_file = f"{output_dir}/sklearn_metrics_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Metrics table saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    results_file = "results/sklearn_anomaly_detection_results.json"
    output_dir = "charts"

    print("Loading results...")
    results = load_results(results_file)

    print("Creating visualizations...")
    create_comparison_charts(results, output_dir)

    print("\n[SUCCESS] Visualizations created!")
    print(f"Output directory: {output_dir}/")
