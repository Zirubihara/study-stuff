"""
Combined Visualizations using Holoviews
High-level declarative visualizations for both data processing and ML/DL comparisons
"""

import json
from pathlib import Path
import holoviews as hv
from holoviews import opts
import pandas as pd

hv.extension('bokeh')  # Use Bokeh backend

class HoloviewsVisualizer:
    """Holoviews visualizations for thesis comparisons"""

    def __init__(self,
                 dp_results_dir="../../results",
                 ml_results_dir="../../models/results",
                 output_dir="./output"):
        self.dp_results_dir = Path(dp_results_dir)
        self.ml_results_dir = Path(ml_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data processing
        self.libraries = ['pandas', 'polars', 'pyarrow', 'dask', 'spark']
        self.dataset_sizes = ['5M', '10M', '50M']
        self.operations = ['loading', 'cleaning', 'aggregation', 'sorting', 'filtering', 'correlation']

        # ML/DL
        self.frameworks = ['sklearn', 'pytorch', 'tensorflow', 'xgboost', 'jax']
        self.framework_names = {
            'sklearn': 'Scikit-learn',
            'pytorch': 'PyTorch',
            'tensorflow': 'TensorFlow',
            'xgboost': 'XGBoost',
            'jax': 'JAX'
        }

    def load_data_processing_results(self):
        """Load data processing benchmark results"""
        print("Loading data processing results...")
        data = {}

        for lib in self.libraries:
            data[lib] = {}
            for size in self.dataset_sizes:
                filename = f"performance_metrics_{lib}_{size}.json"
                filepath = self.dp_results_dir / filename

                if filepath.exists():
                    with open(filepath, 'r') as f:
                        data[lib][size] = json.load(f)
                    print(f"  Loaded: {filename}")

        return data

    def load_ml_results(self):
        """Load ML/DL framework results"""
        print("Loading ML/DL results...")
        data = {}

        for fw in self.frameworks:
            filepath = self.ml_results_dir / f"{fw}_anomaly_detection_results.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data[fw] = json.load(f)
                print(f"  Loaded: {fw}_anomaly_detection_results.json")

        return data

    def plot_dp_execution_time(self, data, dataset_size='10M'):
        """Data processing execution time comparison"""
        print(f"\nCreating data processing execution time chart ({dataset_size})...")

        # Prepare data
        chart_data = []
        for lib in self.libraries:
            if dataset_size in data[lib]:
                time = data[lib][dataset_size].get('total_operation_time_mean', 0)
                chart_data.append({'Library': lib.capitalize(), 'Time': time})

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=['Library'], vdims=['Time'])
        bars.opts(
            opts.Bars(
                width=800, height=500,
                title=f'Data Processing Performance - {dataset_size} Dataset',
                xlabel='Library',
                ylabel='Total Execution Time (seconds)',
                color='Library',
                cmap='Category10',
                tools=['hover'],
                show_legend=False
            )
        )

        # Save
        output_file = self.output_dir / f'dp_execution_time_{dataset_size}.html'
        hv.save(bars, output_file)
        print(f"  Saved: dp_execution_time_{dataset_size}.html")

        return bars

    def plot_dp_operation_breakdown(self, data, dataset_size='10M'):
        """Operation breakdown for data processing"""
        print(f"\nCreating operation breakdown chart ({dataset_size})...")

        # Prepare data
        chart_data = []
        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue

            for op in self.operations:
                key = f"{op}_time_mean"
                time = data[lib][dataset_size].get(key, 0)
                chart_data.append({
                    'Library': lib.capitalize(),
                    'Operation': op.capitalize(),
                    'Time': time
                })

        df = pd.DataFrame(chart_data)

        # Create grouped bar chart
        bars = hv.Bars(df, kdims=['Operation', 'Library'], vdims=['Time'])
        bars.opts(
            opts.Bars(
                width=1000, height=500,
                title=f'Operation Breakdown - {dataset_size} Dataset',
                xlabel='Operation',
                ylabel='Time (seconds)',
                color='Library',
                cmap='Category10',
                tools=['hover'],
                legend_position='top_left',
                xrotation=45
            )
        )

        # Save
        output_file = self.output_dir / f'dp_operation_breakdown_{dataset_size}.html'
        hv.save(bars, output_file)
        print(f"  Saved: dp_operation_breakdown_{dataset_size}.html")

        return bars

    def plot_dp_scalability(self, data):
        """Scalability analysis for data processing"""
        print("\nCreating scalability analysis chart...")

        sizes_numeric = [5, 10, 50]

        # Prepare data
        chart_data = []
        for lib in self.libraries:
            for i, size in enumerate(self.dataset_sizes):
                if size in data[lib]:
                    time = data[lib][size].get('total_operation_time_mean', 0)
                    if time > 0:
                        chart_data.append({
                            'Library': lib.capitalize(),
                            'Dataset Size (M)': sizes_numeric[i],
                            'Time': time
                        })

        df = pd.DataFrame(chart_data)

        # Create curve chart
        curves = hv.Curve(df, kdims=['Dataset Size (M)'], vdims=['Time'], label=self.libraries[0].capitalize())

        overlay = hv.Overlay([
            hv.Curve(df[df['Library'] == lib.capitalize()],
                    kdims=['Dataset Size (M)'], vdims=['Time'],
                    label=lib.capitalize())
            for lib in self.libraries
        ])

        overlay.opts(
            opts.Curve(
                width=900, height=600,
                title='Scalability Analysis: Performance vs Dataset Size',
                xlabel='Dataset Size (Million Rows)',
                ylabel='Execution Time (seconds)',
                logx=True, logy=True,
                tools=['hover'],
                show_grid=True
            ),
            opts.Overlay(legend_position='top_left')
        )

        # Save
        output_file = self.output_dir / 'dp_scalability_analysis.html'
        hv.save(overlay, output_file)
        print("  Saved: dp_scalability_analysis.html")

        return overlay

    def plot_ml_training_time(self, data):
        """ML/DL training time comparison"""
        print("\nCreating ML/DL training time chart...")

        # Prepare data
        chart_data = []
        for fw in self.frameworks:
            if fw not in data:
                continue

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
                chart_data.append({
                    'Framework': self.framework_names[fw],
                    'Training Time': time
                })

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=['Framework'], vdims=['Training Time'])
        bars.opts(
            opts.Bars(
                width=800, height=500,
                title='ML/DL Framework Training Time Comparison',
                xlabel='Framework',
                ylabel='Training Time (seconds)',
                color='Framework',
                cmap='Set2',
                tools=['hover'],
                show_legend=False,
                xrotation=45
            )
        )

        # Save
        output_file = self.output_dir / 'ml_training_time.html'
        hv.save(bars, output_file)
        print("  Saved: ml_training_time.html")

        return bars

    def plot_ml_inference_speed(self, data):
        """ML/DL inference speed comparison"""
        print("\nCreating ML/DL inference speed chart...")

        # Prepare data
        chart_data = []
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
                chart_data.append({
                    'Framework': self.framework_names[fw],
                    'Inference Speed': speed
                })

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=['Framework'], vdims=['Inference Speed'])
        bars.opts(
            opts.Bars(
                width=800, height=500,
                title='ML/DL Framework Inference Speed Comparison',
                xlabel='Framework',
                ylabel='Inference Speed (samples/sec)',
                color='Framework',
                cmap='Set2',
                tools=['hover'],
                show_legend=False,
                xrotation=45
            )
        )

        # Save
        output_file = self.output_dir / 'ml_inference_speed.html'
        hv.save(bars, output_file)
        print("  Saved: ml_inference_speed.html")

        return bars

    def plot_ml_comparison_heatmap(self, data):
        """ML/DL framework comparison heatmap"""
        print("\nCreating ML/DL comparison heatmap...")

        # Prepare data
        metrics = ['Training Time', 'Inference Speed', 'Memory Usage']
        chart_data = []

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

            # Normalize values for heatmap
            train_time = model_data.get('training_time', 0)
            infer_speed = model_data.get('inference_speed', 0) / 1000  # Scale down
            memory = abs(model_data.get('memory_usage_gb', 0))

            chart_data.append({
                'Framework': self.framework_names[fw],
                'Metric': 'Training Time',
                'Value': train_time
            })
            chart_data.append({
                'Framework': self.framework_names[fw],
                'Metric': 'Inference Speed',
                'Value': infer_speed
            })
            chart_data.append({
                'Framework': self.framework_names[fw],
                'Metric': 'Memory Usage',
                'Value': memory
            })

        df = pd.DataFrame(chart_data)

        # Create heatmap
        heatmap = hv.HeatMap(df, kdims=['Metric', 'Framework'], vdims=['Value'])
        heatmap.opts(
            opts.HeatMap(
                width=700, height=500,
                title='ML/DL Framework Multi-Metric Comparison',
                xlabel='Metric',
                ylabel='Framework',
                colorbar=True,
                cmap='RdYlGn_r',
                tools=['hover'],
                xrotation=45
            )
        )

        # Save
        output_file = self.output_dir / 'ml_comparison_heatmap.html'
        hv.save(heatmap, output_file)
        print("  Saved: ml_comparison_heatmap.html")

        return heatmap

    def generate_all_visualizations(self):
        """Generate all Holoviews visualizations"""
        print("="*80)
        print("HOLOVIEWS DECLARATIVE VISUALIZATIONS")
        print("="*80)

        # Load data
        dp_data = self.load_data_processing_results()
        ml_data = self.load_ml_results()

        if not dp_data or not ml_data:
            print("Warning: Some data not found!")
            return

        # Data Processing visualizations
        self.plot_dp_execution_time(dp_data, '10M')
        self.plot_dp_operation_breakdown(dp_data, '10M')
        self.plot_dp_scalability(dp_data)

        # ML/DL visualizations
        if ml_data:
            self.plot_ml_training_time(ml_data)
            self.plot_ml_inference_speed(ml_data)
            self.plot_ml_comparison_heatmap(ml_data)

        print("\n" + "="*80)
        print("HOLOVIEWS VISUALIZATIONS COMPLETE")
        print(f"Charts saved to: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    visualizer = HoloviewsVisualizer()
    visualizer.generate_all_visualizations()
