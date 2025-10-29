"""
ML/DL Frameworks Visualization using Matplotlib
Compares scikit-learn, PyTorch, TensorFlow, XGBoost, and JAX for anomaly detection
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class MLFrameworkVisualizerMatplotlib:
    """Matplotlib visualizations for ML/DL framework comparison"""

    def __init__(self, results_dir="../../models/results", output_dir="./output"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frameworks = ['sklearn', 'pytorch', 'tensorflow', 'xgboost', 'jax']
        self.framework_names = {
            'sklearn': 'Scikit-learn',
            'pytorch': 'PyTorch',
            'tensorflow': 'TensorFlow',
            'xgboost': 'XGBoost',
            'jax': 'JAX'
        }

    def load_ml_results(self):
        """Load ML/DL framework results"""
        print("Loading ML/DL framework results...")
        data = {}

        for framework in self.frameworks:
            filename = f"{framework}_anomaly_detection_results.json"
            filepath = self.results_dir / filename

            if filepath.exists():
                with open(filepath, 'r') as f:
                    data[framework] = json.load(f)
                print(f"  Loaded: {filename}")
            else:
                print(f"  Missing: {filename}")

        return data

    def plot_training_time_comparison(self, data):
        """Compare training times across frameworks"""
        print("\nCreating training time comparison chart...")

        fig, ax = plt.subplots(figsize=(10, 6))

        frameworks = []
        times = []

        for fw in self.frameworks:
            if fw not in data:
                continue

            # Get training time from first model (e.g., isolation_forest, autoencoder)
            if fw == 'sklearn':
                time = data[fw].get('isolation_forest', {}).get('training_time', 0)
            elif fw == 'pytorch':
                time = data[fw].get('pytorch_autoencoder', {}).get('training_time', 0)
            elif fw == 'tensorflow':
                time = data[fw].get('tensorflow_autoencoder', {}).get('training_time', 0)
            elif fw == 'jax':
                time = data[fw].get('jax_autoencoder', {}).get('training_time', 0)
            elif fw == 'xgboost':
                time = data[fw].get('xgboost_detector', {}).get('training_time', 0)
            else:
                continue

            if time > 0:
                frameworks.append(self.framework_names[fw])
                times.append(time)

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(frameworks)))
        bars = ax.bar(frameworks, times, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('ML/DL Framework Training Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = self.output_dir / 'ml_training_time.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_inference_speed_comparison(self, data):
        """Compare inference speeds"""
        print("\nCreating inference speed comparison chart...")

        fig, ax = plt.subplots(figsize=(10, 6))

        frameworks = []
        speeds = []

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == 'sklearn':
                speed = data[fw].get('isolation_forest', {}).get('inference_speed', 0)
            elif fw == 'pytorch':
                speed = data[fw].get('pytorch_autoencoder', {}).get('inference_speed', 0)
            elif fw == 'tensorflow':
                speed = data[fw].get('tensorflow_autoencoder', {}).get('inference_speed', 0)
            elif fw == 'jax':
                speed = data[fw].get('jax_autoencoder', {}).get('inference_speed', 0)
            elif fw == 'xgboost':
                speed = data[fw].get('xgboost_detector', {}).get('inference_speed', 0)
            else:
                continue

            if speed > 0:
                frameworks.append(self.framework_names[fw])
                speeds.append(speed)

        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(frameworks)))
        bars = ax.bar(frameworks, speeds, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Inference Speed (samples/sec)', fontsize=12, fontweight='bold')
        ax.set_title('ML/DL Framework Inference Speed Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = self.output_dir / 'ml_inference_speed.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_memory_usage_comparison(self, data):
        """Compare memory usage across frameworks"""
        print("\nCreating memory usage comparison chart...")

        fig, ax = plt.subplots(figsize=(10, 6))

        frameworks = []
        memory = []

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == 'sklearn':
                mem = data[fw].get('isolation_forest', {}).get('memory_usage_gb', 0)
            elif fw == 'pytorch':
                mem = data[fw].get('pytorch_autoencoder', {}).get('memory_usage_gb', 0)
            elif fw == 'tensorflow':
                mem = data[fw].get('tensorflow_autoencoder', {}).get('memory_usage_gb', 0)
            elif fw == 'jax':
                mem = data[fw].get('jax_autoencoder', {}).get('memory_usage_gb', 0)
            elif fw == 'xgboost':
                mem = data[fw].get('xgboost_detector', {}).get('memory_usage_gb', 0)
            else:
                continue

            if mem > 0:
                frameworks.append(self.framework_names[fw])
                memory.append(mem)

        colors = plt.cm.coolwarm(np.linspace(0.2, 0.9, len(frameworks)))
        bars = ax.bar(frameworks, memory, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}GB',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Memory Usage (GB)', fontsize=12, fontweight='bold')
        ax.set_title('ML/DL Framework Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = self.output_dir / 'ml_memory_usage.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_anomaly_detection_rate(self, data):
        """Compare anomaly detection rates"""
        print("\nCreating anomaly detection rate chart...")

        fig, ax = plt.subplots(figsize=(10, 6))

        frameworks = []
        rates = []

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == 'sklearn':
                rate = data[fw].get('isolation_forest', {}).get('anomaly_rate', 0)
            elif fw == 'pytorch':
                rate = data[fw].get('pytorch_autoencoder', {}).get('anomaly_rate', 0)
            elif fw == 'tensorflow':
                rate = data[fw].get('tensorflow_autoencoder', {}).get('anomaly_rate', 0)
            elif fw == 'jax':
                rate = data[fw].get('jax_autoencoder', {}).get('anomaly_rate', 0)
            elif fw == 'xgboost':
                rate = data[fw].get('xgboost_detector', {}).get('anomaly_rate', 0)
            else:
                continue

            if rate > 0:
                frameworks.append(self.framework_names[fw])
                rates.append(rate)

        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(frameworks)))
        bars = ax.bar(frameworks, rates, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Anomaly Detection Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('ML/DL Framework Anomaly Detection Rate', fontsize=14, fontweight='bold')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Expected (1%)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = self.output_dir / 'ml_anomaly_rate.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_framework_comparison_matrix(self, data):
        """Create comparison matrix for all metrics"""
        print("\nCreating framework comparison matrix...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics = [
            ('training_time', 'Training Time (s)', 'viridis'),
            ('inference_speed', 'Inference Speed (samples/s)', 'plasma'),
            ('memory_usage_gb', 'Memory Usage (GB)', 'coolwarm'),
            ('anomaly_rate', 'Anomaly Detection Rate (%)', 'RdYlGn_r')
        ]

        for idx, (metric, title, cmap) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            frameworks = []
            values = []

            for fw in self.frameworks:
                if fw not in data:
                    continue

                if fw == 'sklearn':
                    val = data[fw].get('isolation_forest', {}).get(metric, 0)
                elif fw in ['pytorch', 'tensorflow', 'jax']:
                    val = data[fw].get('autoencoder', {}).get(metric, 0)
                elif fw == 'xgboost':
                    val = data[fw].get('xgboost_detector', {}).get(metric, 0)
                else:
                    continue

                if val > 0 or metric == 'memory_usage_gb':  # Allow negative memory
                    frameworks.append(self.framework_names[fw])
                    values.append(val)

            if frameworks:
                colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.9, len(frameworks)))
                bars = ax.bar(frameworks, values, color=colors, edgecolor='black', linewidth=1.5)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if metric == 'inference_speed':
                        label = f'{height:,.0f}'
                    elif metric == 'memory_usage_gb':
                        label = f'{height:.2f}'
                    else:
                        label = f'{height:.2f}'

                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=9, fontweight='bold')

                ax.set_ylabel(title, fontsize=11, fontweight='bold')
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.tick_params(axis='x', rotation=45)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle('ML/DL Framework Comprehensive Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / 'ml_comparison_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_performance_summary_table(self, data):
        """Create comprehensive summary table"""
        print("\nCreating performance summary table...")

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')

        headers = ['Framework', 'Training Time', 'Inference Speed', 'Memory Usage', 'Anomaly Rate']
        table_data = []

        for fw in self.frameworks:
            if fw not in data:
                continue

            row = [self.framework_names[fw]]

            if fw == 'sklearn':
                model_data = data[fw].get('isolation_forest', {})
            elif fw in ['pytorch', 'tensorflow', 'jax']:
                model_data = data[fw].get('autoencoder', {})
            elif fw == 'xgboost':
                model_data = data[fw].get('xgboost_detector', {})
            else:
                continue

            row.append(f"{model_data.get('training_time', 0):.2f}s")
            row.append(f"{model_data.get('inference_speed', 0):,.0f} s/s")
            row.append(f"{model_data.get('memory_usage_gb', 0):.2f} GB")
            row.append(f"{model_data.get('anomaly_rate', 0):.2f}%")

            table_data.append(row)

        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('ML/DL Framework Performance Summary', fontsize=16, fontweight='bold', pad=20)

        output_file = self.output_dir / 'ml_summary_table.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_ml_training_vs_inference(self, data):
        """Scatter plot: Training time vs Inference speed"""
        print("\nCreating training vs inference scatter plot...")

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.Set2(np.linspace(0, 1, len(self.frameworks)))

        for i, fw in enumerate(self.frameworks):
            if fw not in data:
                continue

            if fw == 'sklearn':
                model_data = data[fw].get('isolation_forest', {})
            elif fw in ['pytorch', 'tensorflow', 'jax']:
                model_data = data[fw].get(f'{fw}_autoencoder', {})
            elif fw == 'xgboost':
                model_data = data[fw].get('xgboost_detector', {})
            else:
                continue

            train = model_data.get('training_time', 0)
            infer = model_data.get('inference_speed', 0)

            if train > 0 and infer > 0:
                ax.scatter([train], [infer], s=150, c=[colors[i]],
                          label=self.framework_names[fw], alpha=0.7)
                ax.annotate(self.framework_names[fw], (train, infer),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('Inference Speed (samples/sec)', fontsize=12)
        ax.set_title('ML/DL Training vs Inference Trade-off',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_file = self.output_dir / 'ml_training_vs_inference.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_ml_framework_radar(self, data):
        """Radar chart for ML framework comparison"""
        print("\nCreating ML framework radar chart...")

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        metrics = ['Training', 'Inference', 'Memory']
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        colors = plt.cm.Set2(np.linspace(0, 1, len(self.frameworks)))

        for i, fw in enumerate(self.frameworks):
            if fw not in data:
                continue

            if fw == 'sklearn':
                model_data = data[fw].get('isolation_forest', {})
            elif fw in ['pytorch', 'tensorflow', 'jax']:
                model_data = data[fw].get(f'{fw}_autoencoder', {})
            elif fw == 'xgboost':
                model_data = data[fw].get('xgboost_detector', {})
            else:
                continue

            train_time = model_data.get('training_time', 0)
            infer_speed = model_data.get('inference_speed', 0)
            memory = abs(model_data.get('memory_usage_gb', 0))

            values = [train_time, infer_speed / 1000, memory]

            if max(values) > 0:
                max_val = max(values)
                values_norm = [v / max_val for v in values]
                values_norm += values_norm[:1]

                ax.plot(angles, values_norm, 'o-', linewidth=2,
                       label=self.framework_names[fw], color=colors[i])
                ax.fill(angles, values_norm, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('ML Framework Radar Comparison',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        output_file = self.output_dir / 'ml_framework_radar.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_ml_framework_ranking(self, data):
        """ML framework ranking (horizontal bars)"""
        print("\nCreating ML framework ranking...")

        # Calculate composite scores (higher inference / lower training = better)
        scores = {}
        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == 'sklearn':
                model_data = data[fw].get('isolation_forest', {})
            elif fw in ['pytorch', 'tensorflow', 'jax']:
                model_data = data[fw].get(f'{fw}_autoencoder', {})
            elif fw == 'xgboost':
                model_data = data[fw].get('xgboost_detector', {})
            else:
                continue

            train = model_data.get('training_time', 0)
            infer = model_data.get('inference_speed', 0)

            if train > 0 and infer > 0:
                scores[self.framework_names[fw]] = infer / train

        # Sort by score
        sorted_fws = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fw_names = [fw[0] for fw in sorted_fws]
        score_values = [fw[1] for fw in sorted_fws]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(fw_names, score_values, color='#e74c3c')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.2f}',
                   ha='left', va='center', fontsize=10)

        ax.set_xlabel('Performance Score (Inference/Training)', fontsize=12)
        ax.set_ylabel('Framework', fontsize=12)
        ax.set_title('ML Framework Performance Ranking',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        output_file = self.output_dir / 'ml_framework_ranking.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def generate_all_visualizations(self):
        """Generate all Matplotlib ML/DL visualizations"""
        print("="*80)
        print("MATPLOTLIB ML/DL FRAMEWORK VISUALIZATIONS")
        print("="*80)

        data = self.load_ml_results()

        if not data:
            print("No ML/DL results found!")
            return

        self.plot_training_time_comparison(data)
        self.plot_inference_speed_comparison(data)
        self.plot_memory_usage_comparison(data)
        self.plot_anomaly_detection_rate(data)
        self.plot_framework_comparison_matrix(data)
        self.plot_performance_summary_table(data)
        self.plot_ml_training_vs_inference(data)
        self.plot_ml_framework_radar(data)
        self.plot_ml_framework_ranking(data)

        print("\n" + "="*80)
        print("MATPLOTLIB ML/DL COMPLETE - 9 STANDARDIZED CHARTS")
        print(f"Charts saved to: {self.output_dir}")
        print("  1. ml_training_time.png")
        print("  2. ml_inference_speed.png")
        print("  3. ml_memory_usage.png")
        print("  4. ml_anomaly_rate.png")
        print("  5. ml_comparison_matrix.png")
        print("  6. ml_summary_table.png")
        print("="*80)


if __name__ == "__main__":
    visualizer = MLFrameworkVisualizerMatplotlib()
    visualizer.generate_all_visualizations()
