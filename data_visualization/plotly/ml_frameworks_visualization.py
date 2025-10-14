"""
ML/DL Frameworks Visualization using Plotly
Interactive visualizations for framework comparison
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

class MLFrameworkVisualizerPlotly:
    """Plotly interactive visualizations for ML/DL frameworks"""

    def __init__(self, results_dir="../models/results", output_dir="./charts_plotly"):
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
        self.colors = {
            'sklearn': '#1f77b4',
            'pytorch': '#ff7f0e',
            'tensorflow': '#2ca02c',
            'xgboost': '#d62728',
            'jax': '#9467bd'
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

    def plot_training_vs_inference_interactive(self, data):
        """Interactive scatter plot: Training time vs Inference speed"""
        print("\nCreating training vs inference scatter plot...")

        fig = go.Figure()

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == 'sklearn':
                model_data = data[fw].get('isolation_forest', {})
            elif fw == 'pytorch':
                model_data = data[fw].get('pytorch_autoencoder', {})
            elif fw == 'tensorflow':
                model_data = data[fw].get('tensorflow_autoencoder', {})
            elif fw == 'jax':
                model_data = data[fw].get('jax_autoencoder', {})
            elif fw == 'xgboost':
                model_data = data[fw].get('xgboost_detector', {})
            else:
                continue

            train_time = model_data.get('training_time', 0)
            infer_speed = model_data.get('inference_speed', 0)
            memory = model_data.get('memory_usage_gb', 0)

            if train_time > 0 and infer_speed > 0:
                fig.add_trace(go.Scatter(
                    x=[train_time],
                    y=[infer_speed],
                    mode='markers+text',
                    name=self.framework_names[fw],
                    marker=dict(
                        size=15 + abs(memory) * 5,  # Size based on memory
                        color=self.colors[fw],
                        line=dict(width=2, color='white')
                    ),
                    text=[self.framework_names[fw]],
                    textposition="top center",
                    hovertemplate=f'<b>{self.framework_names[fw]}</b><br>' +
                                f'Training Time: {train_time:.2f}s<br>' +
                                f'Inference Speed: {infer_speed:,.0f} samples/s<br>' +
                                f'Memory: {memory:.2f} GB<extra></extra>'
                ))

        fig.update_layout(
            title='ML/DL Framework: Training Time vs Inference Speed Trade-off',
            xaxis_title='Training Time (seconds) - Lower is Better',
            yaxis_title='Inference Speed (samples/sec) - Higher is Better',
            hovermode='closest',
            template='plotly_white',
            height=600,
            font=dict(size=12),
            showlegend=True
        )

        output_file = self.output_dir / 'ml_training_vs_inference_interactive.html'
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_framework_radar_interactive(self, data):
        """Interactive radar chart for framework comparison"""
        print("\nCreating interactive radar chart...")

        fig = go.Figure()

        metrics = ['Training Speed', 'Inference Speed', 'Memory Efficiency', 'Anomaly Accuracy']

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == 'sklearn':
                model_data = data[fw].get('isolation_forest', {})
            elif fw == 'pytorch':
                model_data = data[fw].get('pytorch_autoencoder', {})
            elif fw == 'tensorflow':
                model_data = data[fw].get('tensorflow_autoencoder', {})
            elif fw == 'jax':
                model_data = data[fw].get('jax_autoencoder', {})
            elif fw == 'xgboost':
                model_data = data[fw].get('xgboost_detector', {})
            else:
                continue

            train_time = model_data.get('training_time', 0)
            infer_speed = model_data.get('inference_speed', 0)
            memory = model_data.get('memory_usage_gb', 0)
            anomaly_rate = model_data.get('anomaly_rate', 0)

            # Normalize to 0-10 scale
            # Training: faster = better (inverted)
            train_score = 10 / (1 + train_time / 100)
            # Inference: faster = better
            infer_score = min(10, infer_speed / 10000)
            # Memory: less = better (inverted)
            mem_score = 10 / (1 + abs(memory))
            # Anomaly: closer to expected 1% = better
            anomaly_score = 10 * (1 - abs(anomaly_rate - 1.0) / 5)

            scores = [train_score, infer_score, mem_score, anomaly_score]

            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=metrics + [metrics[0]],
                fill='toself',
                name=self.framework_names[fw],
                line=dict(color=self.colors[fw], width=2),
                hovertemplate=f'<b>{self.framework_names[fw]}</b><br>' +
                            '%{theta}: %{r:.2f}<extra></extra>'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            title='ML/DL Framework Comprehensive Performance Comparison<br>(Higher = Better)',
            showlegend=True,
            height=700,
            template='plotly_white'
        )

        output_file = self.output_dir / 'ml_framework_radar_interactive.html'
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_multi_metric_comparison(self, data):
        """Multi-metric bar chart comparison"""
        print("\nCreating multi-metric comparison chart...")

        metrics = [
            ('training_time', 'Training Time (s)'),
            ('inference_speed', 'Inference Speed (samples/s)'),
            ('memory_usage_gb', 'Memory Usage (GB)'),
            ('anomaly_rate', 'Anomaly Rate (%)')
        ]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m[1] for m in metrics]
        )

        for idx, (metric, title) in enumerate(metrics):
            row = (idx // 2) + 1
            col = (idx % 2) + 1

            frameworks = []
            values = []

            for fw in self.frameworks:
                if fw not in data:
                    continue

                if fw == 'sklearn':
                    model_data = data[fw].get('isolation_forest', {})
                elif fw in ['pytorch', 'tensorflow', 'jax']:
                    model_data = data[fw].get('autoencoder', {})
                elif fw == 'xgboost':
                    model_data = data[fw].get('xgboost_detector', {})
                else:
                    continue

                val = model_data.get(metric, 0)
                if val != 0 or metric == 'memory_usage_gb':
                    frameworks.append(self.framework_names[fw])
                    values.append(val)

            if frameworks:
                fig.add_trace(
                    go.Bar(
                        x=frameworks,
                        y=values,
                        marker_color=[self.colors[fw.lower()] for fw in self.frameworks if self.framework_names[fw] in frameworks],
                        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=col
                )

        fig.update_layout(
            title_text='ML/DL Framework Multi-Metric Comparison',
            height=800,
            template='plotly_white',
            showlegend=False
        )

        output_file = self.output_dir / 'ml_multi_metric_comparison.html'
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_framework_ranking(self, data):
        """Create interactive ranking visualization"""
        print("\nCreating framework ranking visualization...")

        metrics = {
            'Training Speed': 'training_time',
            'Inference Speed': 'inference_speed',
            'Memory Efficiency': 'memory_usage_gb',
            'Anomaly Accuracy': 'anomaly_rate'
        }

        fig = go.Figure()

        for metric_name, metric_key in metrics.items():
            rankings = []

            for fw in self.frameworks:
                if fw not in data:
                    continue

                if fw == 'sklearn':
                    model_data = data[fw].get('isolation_forest', {})
                elif fw in ['pytorch', 'tensorflow', 'jax']:
                    model_data = data[fw].get('autoencoder', {})
                elif fw == 'xgboost':
                    model_data = data[fw].get('xgboost_detector', {})
                else:
                    continue

                val = model_data.get(metric_key, 0)
                if val != 0 or metric_key == 'memory_usage_gb':
                    rankings.append((self.framework_names[fw], val))

            # Sort rankings (lower is better for time/memory, higher for speed)
            if metric_key in ['training_time', 'memory_usage_gb']:
                rankings.sort(key=lambda x: abs(x[1]))
            elif metric_key == 'inference_speed':
                rankings.sort(key=lambda x: x[1], reverse=True)
            else:  # anomaly_rate - closer to 1.0 is better
                rankings.sort(key=lambda x: abs(x[1] - 1.0))

            # Assign ranks
            frameworks_ranked = [r[0] for r in rankings]
            values = [r[1] for r in rankings]

            fig.add_trace(go.Bar(
                name=metric_name,
                x=frameworks_ranked,
                y=list(range(len(frameworks_ranked), 0, -1)),  # Reverse rank (5=best, 1=worst)
                orientation='v',
                marker_color=self.colors.get(rankings[0][0].lower(), '#888'),
                visible=False,
                hovertemplate='<b>%{x}</b><br>Rank: %{y}<br>Value: ' +
                            ', '.join([f'{v:.2f}' for v in values]) + '<extra></extra>'
            ))

        # Make first trace visible
        fig.data[0].visible = True

        # Create buttons for metric selection
        buttons = []
        for i, metric_name in enumerate(metrics.keys()):
            visibility = [i == j for j in range(len(metrics))]
            buttons.append(dict(
                label=metric_name,
                method='update',
                args=[{'visible': visibility},
                     {'title': f'Framework Ranking: {metric_name}'}]
            ))

        fig.update_layout(
            updatemenus=[dict(
                type='buttons',
                direction='left',
                x=0.5,
                xanchor='center',
                y=1.15,
                yanchor='top',
                buttons=buttons
            )],
            title='Framework Ranking: Training Speed',
            xaxis_title='Framework',
            yaxis_title='Rank (Higher = Better)',
            template='plotly_white',
            height=600
        )

        output_file = self.output_dir / 'ml_framework_ranking_interactive.html'
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def generate_all_visualizations(self):
        """Generate all Plotly ML/DL visualizations"""
        print("="*80)
        print("PLOTLY ML/DL FRAMEWORK VISUALIZATIONS")
        print("="*80)

        data = self.load_ml_results()

        if not data:
            print("No ML/DL results found!")
            return

        self.plot_training_vs_inference_interactive(data)
        self.plot_framework_radar_interactive(data)
        self.plot_multi_metric_comparison(data)
        self.plot_framework_ranking(data)

        print("\n" + "="*80)
        print("PLOTLY ML/DL VISUALIZATIONS COMPLETE")
        print(f"Interactive charts saved to: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    visualizer = MLFrameworkVisualizerPlotly()
    visualizer.generate_all_visualizations()
