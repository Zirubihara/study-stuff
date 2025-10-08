"""
Compare results across all anomaly detection frameworks
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results():
    """Load all available results"""
    results = {}

    # Load scikit-learn results
    sklearn_file = "results/sklearn_anomaly_detection_results.json"
    if Path(sklearn_file).exists():
        with open(sklearn_file, 'r') as f:
            results['sklearn'] = json.load(f)

    # Load PyTorch results
    pytorch_file = "results/pytorch_anomaly_detection_results.json"
    if Path(pytorch_file).exists():
        with open(pytorch_file, 'r') as f:
            results['pytorch'] = json.load(f)

    return results


def create_comparison_charts(results, output_dir="charts"):
    """Create comprehensive comparison visualizations"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract data for comparison
    models = []
    train_times = []
    inference_times = []
    anomalies = []
    inference_speeds = []
    memory_usage = []

    # Scikit-learn models
    if 'sklearn' in results:
        sklearn = results['sklearn']

        # Isolation Forest
        if_data = sklearn['isolation_forest']
        models.append('Isolation\nForest')
        train_times.append(if_data['training_time'])
        inference_times.append(if_data['inference_time'])
        anomalies.append(if_data['n_anomalies'])
        inference_speeds.append(if_data['inference_speed'])
        memory_usage.append(if_data['memory_usage_gb'])

        # LOF
        lof_data = sklearn['local_outlier_factor']
        models.append('LOF')
        train_times.append(lof_data['training_time'])
        inference_times.append(lof_data['inference_time'])
        anomalies.append(lof_data['n_anomalies'])
        inference_speeds.append(lof_data['inference_speed'])
        memory_usage.append(lof_data['memory_usage_gb'])

    # PyTorch model
    if 'pytorch' in results:
        pytorch_data = results['pytorch']['pytorch_autoencoder']
        models.append('PyTorch\nAutoencoder')
        train_times.append(pytorch_data['training_time'])
        inference_times.append(pytorch_data['inference_time'])
        anomalies.append(pytorch_data['n_anomalies'])
        inference_speeds.append(pytorch_data['inference_speed'])
        memory_usage.append(pytorch_data['memory_usage_gb'])

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Anomaly Detection Framework Comparison\n(Japanese Trade Data)',
                 fontsize=16, fontweight='bold')

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

    # 1. Training Time
    ax = axes[0, 0]
    bars = ax.bar(models, train_times, color=colors[:len(models)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax.set_title('Training Time Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 2. Inference Time
    ax = axes[0, 1]
    bars = ax.bar(models, inference_times, color=colors[:len(models)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Inference Time (seconds)', fontweight='bold')
    ax.set_title('Inference Time Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 3. Inference Speed
    ax = axes[0, 2]
    bars = ax.bar(models, inference_speeds, color=colors[:len(models)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Inference Speed (samples/sec)', fontweight='bold')
    ax.set_title('Inference Speed Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height/1000:.0f}K', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 4. Anomalies Detected
    ax = axes[1, 0]
    bars = ax.bar(models, anomalies, color=colors[:len(models)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Anomalies', fontweight='bold')
    ax.set_title('Anomalies Detected', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 5. Memory Usage
    ax = axes[1, 1]
    bars = ax.bar(models, memory_usage, color=colors[:len(models)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Memory Usage (GB)', fontweight='bold')
    ax.set_title('Memory Usage Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 6. Overall Ranking (normalized scores)
    ax = axes[1, 2]
    # Lower is better for time/memory, higher is better for speed
    # Normalize and create composite score
    norm_train = 1 - np.array(train_times) / max(train_times)  # Inverted
    norm_inference = 1 - np.array(inference_times) / max(inference_times)  # Inverted
    norm_speed = np.array(inference_speeds) / max(inference_speeds)
    norm_memory = 1 - np.array(memory_usage) / max(memory_usage)  # Inverted

    composite_scores = (norm_train + norm_inference + norm_speed + norm_memory) / 4 * 100

    bars = ax.barh(models, composite_scores, color=colors[:len(models)], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Composite Score (%)', fontweight='bold')
    ax.set_title('Overall Performance Ranking', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (bar, score) in enumerate(zip(bars, composite_scores)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{score:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    output_file = f"{output_dir}/framework_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison chart saved: {output_file}")
    plt.close()

    # Create summary table
    fig, ax = plt.subplots(figsize=(14, 6))

    table_data = []
    for i, model in enumerate(models):
        row = [
            model.replace('\n', ' '),
            f"{train_times[i]:.2f}s",
            f"{inference_times[i]:.2f}s",
            f"{inference_speeds[i]:,.0f}",
            f"{anomalies[i]:,}",
            f"{memory_usage[i]:.2f} GB",
            f"{composite_scores[i]:.1f}%"
        ]
        table_data.append(row)

    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'Training Time', 'Inference Time', 'Speed (samples/s)', 'Anomalies', 'Memory', 'Score'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.15, 0.12, 0.12, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    # Highlight best scores
    best_train_idx = train_times.index(min(train_times)) + 1
    best_inference_idx = inference_times.index(min(inference_times)) + 1
    best_speed_idx = inference_speeds.index(max(inference_speeds)) + 1
    best_score_idx = list(composite_scores).index(max(composite_scores)) + 1

    table[(best_train_idx, 1)].set_facecolor('#2ecc71')
    table[(best_inference_idx, 2)].set_facecolor('#2ecc71')
    table[(best_speed_idx, 3)].set_facecolor('#2ecc71')
    table[(best_score_idx, 6)].set_facecolor('#2ecc71')

    plt.title('Complete Framework Performance Comparison\n(Best values highlighted in green)',
              fontsize=14, fontweight='bold', pad=20)

    output_file = f"{output_dir}/framework_comparison_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison table saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    print("Loading all results...")
    results = load_results()

    if not results:
        print("No results found! Please run anomaly detection scripts first.")
    else:
        print(f"Found results for: {', '.join(results.keys())}")
        print("\nCreating comparison visualizations...")
        create_comparison_charts(results)
        print("\n[SUCCESS] Framework comparison complete!")
