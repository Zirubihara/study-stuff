"""
Combined Visualizations using Bokeh
Interactive browser-based visualizations for both data processing and ML/DL comparisons
"""

import json
from pathlib import Path
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, HoverTool, Panel, Tabs
from bokeh.transform import dodge
import numpy as np

class BokehVisualizer:
    """Bokeh visualizations for thesis comparisons"""

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

        # Colors
        self.lib_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.fw_colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']

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

        lib_names = []
        times = []

        for lib in self.libraries:
            if dataset_size in data[lib]:
                lib_names.append(lib.capitalize())
                times.append(data[lib][dataset_size].get('total_operation_time_mean', 0))

        source = ColumnDataSource(data=dict(
            libraries=lib_names,
            times=times,
            colors=self.lib_colors[:len(lib_names)]
        ))

        p = figure(x_range=lib_names,
                   title=f'Data Processing Performance - {dataset_size} Dataset',
                   toolbar_location="above",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=800, height=500)

        p.vbar(x='libraries', top='times', width=0.7, color='colors', source=source,
               legend_field="libraries")

        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Library", "@libraries"),
            ("Time", "@times{0.00} seconds")
        ])
        p.add_tools(hover)

        p.xaxis.axis_label = "Library"
        p.yaxis.axis_label = "Total Execution Time (seconds)"
        p.xgrid.grid_line_color = None
        p.legend.visible = False

        output_file(self.output_dir / f'dp_execution_time_{dataset_size}.html')
        save(p)
        print(f"  Saved: dp_execution_time_{dataset_size}.html")

        return p

    def plot_dp_operation_breakdown(self, data, dataset_size='10M'):
        """Operation breakdown for data processing"""
        print(f"\nCreating operation breakdown chart ({dataset_size})...")

        # Prepare data for grouped bar chart
        x_offset = [-0.3, -0.15, 0, 0.15, 0.3]  # Offsets for 5 libraries

        p = figure(x_range=self.operations,
                   title=f'Operation Breakdown - {dataset_size} Dataset',
                   toolbar_location="above",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=1000, height=500)

        for idx, lib in enumerate(self.libraries):
            if dataset_size not in data[lib]:
                continue

            times = []
            for op in self.operations:
                key = f"{op}_time_mean"
                times.append(data[lib][dataset_size].get(key, 0))

            # Dodge positions
            x_positions = [i + x_offset[idx] for i in range(len(self.operations))]

            p.vbar(x=x_positions, top=times, width=0.12,
                   color=self.lib_colors[idx], legend_label=lib.capitalize())

        p.xaxis.axis_label = "Operation"
        p.yaxis.axis_label = "Time (seconds)"
        p.xgrid.grid_line_color = None
        p.legend.location = "top_left"

        output_file(self.output_dir / f'dp_operation_breakdown_{dataset_size}.html')
        save(p)
        print(f"  Saved: dp_operation_breakdown_{dataset_size}.html")

        return p

    def plot_dp_scalability(self, data):
        """Scalability analysis for data processing"""
        print("\nCreating scalability analysis chart...")

        sizes_numeric = [5, 10, 50]

        p = figure(title='Scalability Analysis: Performance vs Dataset Size',
                   x_axis_type="log", y_axis_type="log",
                   toolbar_location="above",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=900, height=600)

        for idx, lib in enumerate(self.libraries):
            times = []
            valid_sizes = []

            for i, size in enumerate(self.dataset_sizes):
                if size in data[lib]:
                    time = data[lib][size].get('total_operation_time_mean', 0)
                    if time > 0:
                        times.append(time)
                        valid_sizes.append(sizes_numeric[i])

            if valid_sizes:
                p.line(valid_sizes, times, legend_label=lib.capitalize(),
                       line_width=2, color=self.lib_colors[idx])
                p.circle(valid_sizes, times, size=8, color=self.lib_colors[idx])

        p.xaxis.axis_label = "Dataset Size (Million Rows)"
        p.yaxis.axis_label = "Execution Time (seconds)"
        p.legend.location = "top_left"

        output_file(self.output_dir / 'dp_scalability_analysis.html')
        save(p)
        print("  Saved: dp_scalability_analysis.html")

        return p

    def plot_ml_training_time(self, data):
        """ML/DL training time comparison"""
        print("\nCreating ML/DL training time chart...")

        fw_names = []
        times = []

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
                fw_names.append(self.framework_names[fw])
                times.append(time)

        source = ColumnDataSource(data=dict(
            frameworks=fw_names,
            times=times,
            colors=self.fw_colors[:len(fw_names)]
        ))

        p = figure(x_range=fw_names,
                   title='ML/DL Framework Training Time Comparison',
                   toolbar_location="above",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=800, height=500)

        p.vbar(x='frameworks', top='times', width=0.7, color='colors', source=source)

        hover = HoverTool(tooltips=[
            ("Framework", "@frameworks"),
            ("Training Time", "@times{0.00} seconds")
        ])
        p.add_tools(hover)

        p.xaxis.axis_label = "Framework"
        p.yaxis.axis_label = "Training Time (seconds)"
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 0.8

        output_file(self.output_dir / 'ml_training_time.html')
        save(p)
        print("  Saved: ml_training_time.html")

        return p

    def plot_ml_inference_speed(self, data):
        """ML/DL inference speed comparison"""
        print("\nCreating ML/DL inference speed chart...")

        fw_names = []
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
                fw_names.append(self.framework_names[fw])
                speeds.append(speed)

        source = ColumnDataSource(data=dict(
            frameworks=fw_names,
            speeds=speeds,
            colors=self.fw_colors[:len(fw_names)]
        ))

        p = figure(x_range=fw_names,
                   title='ML/DL Framework Inference Speed Comparison',
                   toolbar_location="above",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=800, height=500)

        p.vbar(x='frameworks', top='speeds', width=0.7, color='colors', source=source)

        hover = HoverTool(tooltips=[
            ("Framework", "@frameworks"),
            ("Inference Speed", "@speeds{0,0} samples/sec")
        ])
        p.add_tools(hover)

        p.xaxis.axis_label = "Framework"
        p.yaxis.axis_label = "Inference Speed (samples/sec)"
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 0.8

        output_file(self.output_dir / 'ml_inference_speed.html')
        save(p)
        print("  Saved: ml_inference_speed.html")

        return p

    def plot_ml_memory_usage(self, data):
        """ML/DL memory usage comparison"""
        print("\nCreating ML/DL memory usage chart...")

        fw_names = []
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

            fw_names.append(self.framework_names[fw])
            memory.append(abs(mem))  # Use absolute value for display

        source = ColumnDataSource(data=dict(
            frameworks=fw_names,
            memory=memory,
            colors=self.fw_colors[:len(fw_names)]
        ))

        p = figure(x_range=fw_names,
                   title='ML/DL Framework Memory Usage Comparison',
                   toolbar_location="above",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=800, height=500)

        p.vbar(x='frameworks', top='memory', width=0.7, color='colors', source=source)

        hover = HoverTool(tooltips=[
            ("Framework", "@frameworks"),
            ("Memory Usage", "@memory{0.00} GB")
        ])
        p.add_tools(hover)

        p.xaxis.axis_label = "Framework"
        p.yaxis.axis_label = "Memory Usage (GB)"
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 0.8

        output_file(self.output_dir / 'ml_memory_usage.html')
        save(p)
        print("  Saved: ml_memory_usage.html")

        return p

    def create_combined_dashboard(self, dp_data, ml_data):
        """Create a combined dashboard with multiple charts"""
        print("\nCreating combined dashboard...")

        # Data Processing charts
        dp_exec = self.plot_dp_execution_time(dp_data, '10M')
        dp_ops = self.plot_dp_operation_breakdown(dp_data, '10M')
        dp_scale = self.plot_dp_scalability(dp_data)

        # ML/DL charts
        ml_train = self.plot_ml_training_time(ml_data)
        ml_infer = self.plot_ml_inference_speed(ml_data)
        ml_mem = self.plot_ml_memory_usage(ml_data)

        # Note: Bokeh dashboard layout would be more complex
        # Individual files already created above
        print("\n  Individual chart files created. For full dashboard, use Streamlit.")

    def generate_all_visualizations(self):
        """Generate all Bokeh visualizations"""
        print("="*80)
        print("BOKEH INTERACTIVE VISUALIZATIONS")
        print("="*80)

        # Load data
        dp_data = self.load_data_processing_results()
        ml_data = self.load_ml_results()

        # Generate charts
        self.create_combined_dashboard(dp_data, ml_data)

        print("\n" + "="*80)
        print("BOKEH VISUALIZATIONS COMPLETE")
        print(f"Charts saved to: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    visualizer = BokehVisualizer()
    visualizer.generate_all_visualizations()
