"""
Combined Visualizations using Bokeh
Interactive browser-based visualizations for both data processing and ML/DL comparisons
"""

import json
from pathlib import Path

import numpy as np
from bokeh.layouts import column, gridplot, row
from bokeh.models import ColumnDataSource, HoverTool, Panel, Tabs
from bokeh.plotting import figure, output_file, save
from bokeh.transform import dodge


class BokehVisualizer:
    """Bokeh visualizations for thesis comparisons"""

    def __init__(
        self,
        dp_results_dir="../../results",
        ml_results_dir="../../models/results",
        output_dir="./output",
    ):
        self.dp_results_dir = Path(dp_results_dir)
        self.ml_results_dir = Path(ml_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data processing
        self.libraries = ["pandas", "polars", "pyarrow", "dask", "spark"]
        self.dataset_sizes = ["5M", "10M", "50M"]
        self.operations = [
            "loading",
            "cleaning",
            "aggregation",
            "sorting",
            "filtering",
            "correlation",
        ]

        # ML/DL
        self.frameworks = ["sklearn", "pytorch", "tensorflow", "xgboost", "jax"]
        self.framework_names = {
            "sklearn": "Scikit-learn",
            "pytorch": "PyTorch",
            "tensorflow": "TensorFlow",
            "xgboost": "XGBoost",
            "jax": "JAX",
        }

        # Colors
        self.lib_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        self.fw_colors = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff9896"]

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
                    with open(filepath, "r") as f:
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
                with open(filepath, "r") as f:
                    data[fw] = json.load(f)
                print(f"  Loaded: {fw}_anomaly_detection_results.json")

        return data

    def plot_dp_execution_time(self, data, dataset_size="10M"):
        """Data processing execution time comparison"""
        print(f"\nCreating data processing execution time chart ({dataset_size})...")

        lib_names = []
        times = []

        for lib in self.libraries:
            if dataset_size in data[lib]:
                lib_names.append(lib.capitalize())
                times.append(
                    data[lib][dataset_size].get("total_operation_time_mean", 0)
                )

        source = ColumnDataSource(
            data=dict(
                libraries=lib_names,
                times=times,
                colors=self.lib_colors[: len(lib_names)],
            )
        )

        p = figure(
            x_range=lib_names,
            title=f"Data Processing Performance - {dataset_size} Dataset",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
        )

        p.vbar(
            x="libraries",
            top="times",
            width=0.7,
            color="colors",
            source=source,
            legend_field="libraries",
        )

        # Add hover tool
        hover = HoverTool(
            tooltips=[("Library", "@libraries"), ("Time", "@times{0.00} seconds")]
        )
        p.add_tools(hover)

        p.xaxis.axis_label = "Library"
        p.yaxis.axis_label = "Total Execution Time (seconds)"
        p.xgrid.grid_line_color = None
        p.legend.visible = False

        output_file(self.output_dir / f"dp_execution_time_{dataset_size}.html")
        save(p)
        print(f"  Saved: dp_execution_time_{dataset_size}.html")

        return p

    def plot_dp_operation_breakdown(self, data, dataset_size="10M"):
        """Operation breakdown for data processing"""
        print(f"\nCreating operation breakdown chart ({dataset_size})...")

        # Prepare data for grouped bar chart
        x_offset = [-0.3, -0.15, 0, 0.15, 0.3]  # Offsets for 5 libraries

        p = figure(
            x_range=self.operations,
            title=f"Operation Breakdown - {dataset_size} Dataset",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=1000,
            height=500,
        )

        for idx, lib in enumerate(self.libraries):
            if dataset_size not in data[lib]:
                continue

            times = []
            for op in self.operations:
                key = f"{op}_time_mean"
                times.append(data[lib][dataset_size].get(key, 0))

            # Dodge positions
            x_positions = [i + x_offset[idx] for i in range(len(self.operations))]

            p.vbar(
                x=x_positions,
                top=times,
                width=0.12,
                color=self.lib_colors[idx],
                legend_label=lib.capitalize(),
            )

        p.xaxis.axis_label = "Operation"
        p.yaxis.axis_label = "Time (seconds)"
        p.xgrid.grid_line_color = None
        p.legend.location = "top_left"

        output_file(self.output_dir / f"dp_operation_breakdown_{dataset_size}.html")
        save(p)
        print(f"  Saved: dp_operation_breakdown_{dataset_size}.html")

        return p

    def plot_dp_scalability(self, data):
        """Scalability analysis for data processing"""
        print("\nCreating scalability analysis chart...")

        sizes_numeric = [5, 10, 50]

        p = figure(
            title="Scalability Analysis: Performance vs Dataset Size",
            x_axis_type="log",
            y_axis_type="log",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=900,
            height=600,
        )

        for idx, lib in enumerate(self.libraries):
            times = []
            valid_sizes = []

            for i, size in enumerate(self.dataset_sizes):
                if size in data[lib]:
                    time = data[lib][size].get("total_operation_time_mean", 0)
                    if time > 0:
                        times.append(time)
                        valid_sizes.append(sizes_numeric[i])

            if valid_sizes:
                p.line(
                    valid_sizes,
                    times,
                    legend_label=lib.capitalize(),
                    line_width=2,
                    color=self.lib_colors[idx],
                )
                p.circle(valid_sizes, times, size=8, color=self.lib_colors[idx])

        p.xaxis.axis_label = "Dataset Size (Million Rows)"
        p.yaxis.axis_label = "Execution Time (seconds)"
        p.legend.location = "top_left"

        output_file(self.output_dir / "dp_scalability_analysis.html")
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

            if fw == "sklearn":
                time = data[fw].get("isolation_forest", {}).get("training_time", 0)
            elif fw == "pytorch":
                time = data[fw].get("pytorch_autoencoder", {}).get("training_time", 0)
            elif fw == "tensorflow":
                time = (
                    data[fw].get("tensorflow_autoencoder", {}).get("training_time", 0)
                )
            elif fw == "jax":
                time = data[fw].get("jax_autoencoder", {}).get("training_time", 0)
            elif fw == "xgboost":
                time = data[fw].get("xgboost_detector", {}).get("training_time", 0)
            else:
                continue

            if time > 0:
                fw_names.append(self.framework_names[fw])
                times.append(time)

        source = ColumnDataSource(
            data=dict(
                frameworks=fw_names, times=times, colors=self.fw_colors[: len(fw_names)]
            )
        )

        p = figure(
            x_range=fw_names,
            title="ML/DL Framework Training Time Comparison",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
        )

        p.vbar(x="frameworks", top="times", width=0.7, color="colors", source=source)

        hover = HoverTool(
            tooltips=[
                ("Framework", "@frameworks"),
                ("Training Time", "@times{0.00} seconds"),
            ]
        )
        p.add_tools(hover)

        p.xaxis.axis_label = "Framework"
        p.yaxis.axis_label = "Training Time (seconds)"
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 0.8

        output_file(self.output_dir / "ml_training_time.html")
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

            if fw == "sklearn":
                speed = data[fw].get("isolation_forest", {}).get("inference_speed", 0)
            elif fw == "pytorch":
                speed = (
                    data[fw].get("pytorch_autoencoder", {}).get("inference_speed", 0)
                )
            elif fw == "tensorflow":
                speed = (
                    data[fw].get("tensorflow_autoencoder", {}).get("inference_speed", 0)
                )
            elif fw == "jax":
                speed = data[fw].get("jax_autoencoder", {}).get("inference_speed", 0)
            elif fw == "xgboost":
                speed = data[fw].get("xgboost_detector", {}).get("inference_speed", 0)
            else:
                continue

            if speed > 0:
                fw_names.append(self.framework_names[fw])
                speeds.append(speed)

        source = ColumnDataSource(
            data=dict(
                frameworks=fw_names,
                speeds=speeds,
                colors=self.fw_colors[: len(fw_names)],
            )
        )

        p = figure(
            x_range=fw_names,
            title="ML/DL Framework Inference Speed Comparison",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
        )

        p.vbar(x="frameworks", top="speeds", width=0.7, color="colors", source=source)

        hover = HoverTool(
            tooltips=[
                ("Framework", "@frameworks"),
                ("Inference Speed", "@speeds{0,0} samples/sec"),
            ]
        )
        p.add_tools(hover)

        p.xaxis.axis_label = "Framework"
        p.yaxis.axis_label = "Inference Speed (samples/sec)"
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 0.8

        output_file(self.output_dir / "ml_inference_speed.html")
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

            if fw == "sklearn":
                mem = data[fw].get("isolation_forest", {}).get("memory_usage_gb", 0)
            elif fw == "pytorch":
                mem = data[fw].get("pytorch_autoencoder", {}).get("memory_usage_gb", 0)
            elif fw == "tensorflow":
                mem = (
                    data[fw].get("tensorflow_autoencoder", {}).get("memory_usage_gb", 0)
                )
            elif fw == "jax":
                mem = data[fw].get("jax_autoencoder", {}).get("memory_usage_gb", 0)
            elif fw == "xgboost":
                mem = data[fw].get("xgboost_detector", {}).get("memory_usage_gb", 0)
            else:
                continue

            fw_names.append(self.framework_names[fw])
            memory.append(abs(mem))  # Use absolute value for display

        source = ColumnDataSource(
            data=dict(
                frameworks=fw_names,
                memory=memory,
                colors=self.fw_colors[: len(fw_names)],
            )
        )

        p = figure(
            x_range=fw_names,
            title="ML/DL Framework Memory Usage Comparison",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
        )

        p.vbar(x="frameworks", top="memory", width=0.7, color="colors", source=source)

        hover = HoverTool(
            tooltips=[
                ("Framework", "@frameworks"),
                ("Memory Usage", "@memory{0.00} GB"),
            ]
        )
        p.add_tools(hover)

        p.xaxis.axis_label = "Framework"
        p.yaxis.axis_label = "Memory Usage (GB)"
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 0.8

        output_file(self.output_dir / "ml_memory_usage.html")
        save(p)
        print("  Saved: ml_memory_usage.html")

        return p

    def plot_dp_memory_usage(self, data, dataset_size="10M"):
        """Data processing memory usage comparison"""
        print(f"\nCreating data processing memory usage chart ({dataset_size})...")

        lib_names = []
        memory = []

        for lib in self.libraries:
            if dataset_size in data[lib]:
                load_mem = data[lib][dataset_size].get("loading_memory_mean", 0)
                clean_mem = data[lib][dataset_size].get("cleaning_memory_mean", 0)
                total_mem = (load_mem + clean_mem) / 1024  # Convert to GB
                lib_names.append(lib.capitalize())
                memory.append(total_mem)

        source = ColumnDataSource(
            data=dict(
                libraries=lib_names,
                memory=memory,
                colors=self.lib_colors[: len(lib_names)],
            )
        )

        p = figure(
            x_range=lib_names,
            title=f"Memory Usage Comparison - {dataset_size} Dataset",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
        )

        p.vbar(x="libraries", top="memory", width=0.7, color="colors", source=source)

        hover = HoverTool(
            tooltips=[("Library", "@libraries"), ("Memory Usage", "@memory{0.00} GB")]
        )
        p.add_tools(hover)

        p.xaxis.axis_label = "Library"
        p.yaxis.axis_label = "Memory Usage (GB)"
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 0.8

        output_file(self.output_dir / f"dp_memory_usage_{dataset_size}.html")
        save(p)
        print(f"  Saved: dp_memory_usage_{dataset_size}.html")

        return p

    def plot_ml_anomaly_rate(self, data):
        """ML/DL anomaly detection rate comparison"""
        print("\nCreating ML/DL anomaly rate chart...")

        fw_names = []
        rates = []

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == "sklearn":
                rate = data[fw].get("isolation_forest", {}).get("anomaly_rate", 0)
            elif fw == "pytorch":
                rate = data[fw].get("pytorch_autoencoder", {}).get("anomaly_rate", 0)
            elif fw == "tensorflow":
                rate = data[fw].get("tensorflow_autoencoder", {}).get("anomaly_rate", 0)
            elif fw == "jax":
                rate = data[fw].get("jax_autoencoder", {}).get("anomaly_rate", 0)
            elif fw == "xgboost":
                rate = data[fw].get("xgboost_detector", {}).get("anomaly_rate", 0)
            else:
                continue

            if rate > 0:
                fw_names.append(self.framework_names[fw])
                rates.append(rate)

        source = ColumnDataSource(
            data=dict(
                frameworks=fw_names, rates=rates, colors=self.fw_colors[: len(fw_names)]
            )
        )

        p = figure(
            x_range=fw_names,
            title="Anomaly Detection Rate Comparison",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
        )

        p.vbar(x="frameworks", top="rates", width=0.7, color="colors", source=source)

        hover = HoverTool(
            tooltips=[("Framework", "@frameworks"), ("Anomaly Rate", "@rates{0.00}%")]
        )
        p.add_tools(hover)

        p.xaxis.axis_label = "Framework"
        p.yaxis.axis_label = "Anomaly Detection Rate (%)"
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 0.8

        output_file(self.output_dir / "ml_anomaly_rate.html")
        save(p)
        print("  Saved: ml_anomaly_rate.html")

        return p

    def plot_dp_operation_breakdown_stacked(self, data, dataset_size="10M"):
        """Stacked bar chart for operation breakdown"""
        print(f"\nCreating stacked operation breakdown chart ({dataset_size})...")

        from bokeh.models import FactorRange

        lib_names = []
        for lib in self.libraries:
            if dataset_size in data[lib]:
                lib_names.append(lib.capitalize())

        operation_data = {op: [] for op in self.operations}

        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue
            for op in self.operations:
                key = f"{op}_time_mean"
                time = data[lib][dataset_size].get(key, 0)
                operation_data[op].append(time)

        p = figure(
            x_range=lib_names,
            title=f"Stacked Operation Breakdown - {dataset_size} Dataset",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=1000,
            height=600,
        )

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        bottom = [0] * len(lib_names)

        for i, op in enumerate(self.operations):
            p.vbar(
                x=lib_names,
                top=operation_data[op],
                width=0.7,
                bottom=bottom,
                color=colors[i],
                legend_label=op.capitalize(),
            )
            bottom = [b + t for b, t in zip(bottom, operation_data[op])]

        p.xaxis.axis_label = "Library"
        p.yaxis.axis_label = "Time (seconds)"
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        output_file(
            self.output_dir / f"operation_breakdown_stacked_{dataset_size}.html"
        )
        save(p)
        print(f"  Saved: operation_breakdown_stacked_{dataset_size}.html")

        return p

    def plot_dp_memory_vs_time_scatter(self, data):
        """Scatter plot: Memory vs Time trade-off"""
        print("\nCreating memory vs time scatter plot...")

        p = figure(
            title="Memory vs Time Trade-off Analysis",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=900,
            height=600,
        )

        for size in self.dataset_sizes:
            for i, lib in enumerate(self.libraries):
                if size not in data[lib]:
                    continue

                d = data[lib][size]
                total_time = d.get("total_operation_time_mean", 0)
                load_mem = d.get("loading_memory_mean", 0)
                clean_mem = d.get("cleaning_memory_mean", 0)
                total_mem = (load_mem + clean_mem) / 1024  # GB

                if total_time > 0 and total_mem > 0:
                    p.circle(
                        [total_mem],
                        [total_time],
                        size=12,
                        color=self.lib_colors[i],
                        legend_label=f"{lib.capitalize()} ({size})",
                        alpha=0.7,
                    )

        p.xaxis.axis_label = "Memory Usage (GB)"
        p.yaxis.axis_label = "Execution Time (seconds)"
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        output_file(self.output_dir / "memory_vs_time_scatter.html")
        save(p)
        print("  Saved: memory_vs_time_scatter.html")

        return p

    def plot_ml_training_vs_inference(self, data):
        """Scatter plot: Training time vs Inference speed"""
        print("\nCreating training vs inference scatter plot...")

        p = figure(
            title="ML/DL Training vs Inference Trade-off",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=900,
            height=600,
        )

        fw_names = []
        train_times = []
        infer_speeds = []

        for i, fw in enumerate(self.frameworks):
            if fw not in data:
                continue

            if fw == "sklearn":
                train = data[fw].get("isolation_forest", {}).get("training_time", 0)
                infer = data[fw].get("isolation_forest", {}).get("inference_speed", 0)
            elif fw == "pytorch":
                train = data[fw].get("pytorch_autoencoder", {}).get("training_time", 0)
                infer = (
                    data[fw].get("pytorch_autoencoder", {}).get("inference_speed", 0)
                )
            elif fw == "tensorflow":
                train = (
                    data[fw].get("tensorflow_autoencoder", {}).get("training_time", 0)
                )
                infer = (
                    data[fw].get("tensorflow_autoencoder", {}).get("inference_speed", 0)
                )
            elif fw == "jax":
                train = data[fw].get("jax_autoencoder", {}).get("training_time", 0)
                infer = data[fw].get("jax_autoencoder", {}).get("inference_speed", 0)
            elif fw == "xgboost":
                train = data[fw].get("xgboost_detector", {}).get("training_time", 0)
                infer = data[fw].get("xgboost_detector", {}).get("inference_speed", 0)
            else:
                continue

            if train > 0 and infer > 0:
                p.circle(
                    [train],
                    [infer],
                    size=15,
                    color=self.fw_colors[i],
                    legend_label=self.framework_names[fw],
                    alpha=0.7,
                )

        p.xaxis.axis_label = "Training Time (seconds)"
        p.yaxis.axis_label = "Inference Speed (samples/sec)"
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        output_file(self.output_dir / "ml_training_vs_inference_interactive.html")
        save(p)
        print("  Saved: ml_training_vs_inference_interactive.html")

        return p

    def plot_dp_performance_radar(self, data, dataset_size="10M"):
        """Radar chart for data processing performance"""
        print(f"\nCreating performance radar chart ({dataset_size})...")

        # Simplified version: Use a polar plot
        from math import pi

        categories = [op.capitalize() for op in self.operations]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        p = figure(
            title=f"Performance Radar - {dataset_size}",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=800,
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 1.5),
        )

        for i, lib in enumerate(self.libraries):
            if dataset_size not in data[lib]:
                continue

            values = []
            for op in self.operations:
                key = f"{op}_time_mean"
                values.append(data[lib][dataset_size].get(key, 0))

            # Normalize values (0-1 scale)
            max_val = max(values) if max(values) > 0 else 1
            values_norm = [v / max_val for v in values]
            values_norm += values_norm[:1]

            xs = [val * np.cos(ang) for val, ang in zip(values_norm, angles)]
            ys = [val * np.sin(ang) for val, ang in zip(values_norm, angles)]

            p.line(
                xs,
                ys,
                color=self.lib_colors[i],
                alpha=0.6,
                line_width=2,
                legend_label=lib.capitalize(),
            )

        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        output_file(self.output_dir / f"performance_radar_{dataset_size}.html")
        save(p)
        print(f"  Saved: performance_radar_{dataset_size}.html")

        return p

    def plot_ml_framework_radar(self, data):
        """Radar chart for ML framework comparison"""
        print("\nCreating ML framework radar chart...")

        from math import pi

        metrics = ["Training", "Inference", "Memory"]
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        p = figure(
            title="ML Framework Radar Comparison",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=800,
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 1.5),
        )

        for i, fw in enumerate(self.frameworks):
            if fw not in data:
                continue

            if fw == "sklearn":
                model_data = data[fw].get("isolation_forest", {})
            elif fw in ["pytorch", "tensorflow", "jax"]:
                model_data = data[fw].get(f"{fw}_autoencoder", {})
            elif fw == "xgboost":
                model_data = data[fw].get("xgboost_detector", {})
            else:
                continue

            values = [
                model_data.get("training_time", 0),
                model_data.get("inference_speed", 0) / 1000,  # Scale down
                abs(model_data.get("memory_usage_gb", 0)),
            ]

            if max(values) > 0:
                max_val = max(values)
                values_norm = [v / max_val for v in values]
                values_norm += values_norm[:1]

                xs = [val * np.cos(ang) for val, ang in zip(values_norm, angles)]
                ys = [val * np.sin(ang) for val, ang in zip(values_norm, angles)]

                p.line(
                    xs,
                    ys,
                    color=self.fw_colors[i],
                    alpha=0.6,
                    line_width=2,
                    legend_label=self.framework_names[fw],
                )

        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        output_file(self.output_dir / "ml_framework_radar_interactive.html")
        save(p)
        print("  Saved: ml_framework_radar_interactive.html")

        return p

    def plot_dp_performance_rankings(self, data, dataset_size="10M"):
        """Performance rankings as horizontal bars"""
        print(f"\nCreating performance rankings ({dataset_size})...")

        # Calculate composite scores
        scores = {}
        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue
            total_time = data[lib][dataset_size].get("total_operation_time_mean", 0)
            if total_time > 0:
                scores[lib.capitalize()] = total_time

        # Sort by score (lower is better, so reverse)
        sorted_libs = sorted(scores.items(), key=lambda x: x[1])
        lib_names = [lib[0] for lib in sorted_libs]
        times = [lib[1] for lib in sorted_libs]

        p = figure(
            y_range=lib_names,
            title=f"Performance Rankings - {dataset_size}",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
        )

        p.hbar(y=lib_names, right=times, height=0.7, color="#3498db")

        p.xaxis.axis_label = "Total Execution Time (seconds)"
        p.yaxis.axis_label = "Library"

        output_file(self.output_dir / f"dp_performance_rankings_{dataset_size}.html")
        save(p)
        print(f"  Saved: dp_performance_rankings_{dataset_size}.html")

        return p

    def plot_ml_multi_metric_comparison(self, data):
        """Multi-metric comparison for ML frameworks"""
        print("\nCreating ML multi-metric comparison...")

        from bokeh.layouts import gridplot

        # Create subplots for different metrics
        plots = []

        # Training time
        p1 = self.plot_ml_training_time(data)
        plots.append(p1)

        # Inference speed
        p2 = self.plot_ml_inference_speed(data)
        plots.append(p2)

        # Memory usage
        p3 = self.plot_ml_memory_usage(data)
        plots.append(p3)

        # Anomaly rate
        p4 = self.plot_ml_anomaly_rate(data)
        plots.append(p4)

        # Create grid
        grid = gridplot([[p1, p2], [p3, p4]], width=400, height=350)

        output_file(self.output_dir / "ml_multi_metric_comparison.html")
        save(grid)
        print("  Saved: ml_multi_metric_comparison.html")

        return grid

    def plot_dp_summary_table(self, data, dataset_size="10M"):
        """Summary table for data processing (text-based visualization)"""
        print(f"\nCreating data processing summary table ({dataset_size})...")

        from bokeh.layouts import column
        from bokeh.models import Div

        # Build HTML table
        html = f"""
        <h2>Data Processing Summary - {dataset_size} Dataset</h2>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #3498db; color: white;">
                <th style="border: 1px solid #ddd; padding: 12px;">Library</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Total Time (s)</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Memory (GB)</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Loading (s)</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Aggregation (s)</th>
            </tr>
        """

        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue

            d = data[lib][dataset_size]
            total_time = d.get("total_operation_time_mean", 0)
            load_mem = d.get("loading_memory_mean", 0)
            clean_mem = d.get("cleaning_memory_mean", 0)
            total_mem = (load_mem + clean_mem) / 1024
            loading = d.get("loading_time_mean", 0)
            aggregation = d.get("aggregation_time_mean", 0)

            html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>{lib.capitalize()}</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{total_time:.2f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{total_mem:.2f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{loading:.2f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{aggregation:.2f}</td>
            </tr>
            """

        html += "</table>"

        div = Div(text=html, width=900, height=400)

        output_file(self.output_dir / f"dp_summary_table_{dataset_size}.html")
        save(div)
        print(f"  Saved: dp_summary_table_{dataset_size}.html")

        return div

    def plot_ml_summary_table(self, data):
        """Summary table for ML frameworks (text-based visualization)"""
        print("\nCreating ML summary table...")

        from bokeh.models import Div

        # Build HTML table
        html = """
        <h2>ML/DL Framework Summary</h2>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #e74c3c; color: white;">
                <th style="border: 1px solid #ddd; padding: 12px;">Framework</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Training (s)</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Inference (samp/s)</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Memory (GB)</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Anomaly Rate (%)</th>
            </tr>
        """

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == "sklearn":
                model_data = data[fw].get("isolation_forest", {})
            elif fw in ["pytorch", "tensorflow", "jax"]:
                model_data = data[fw].get(f"{fw}_autoencoder", {})
            elif fw == "xgboost":
                model_data = data[fw].get("xgboost_detector", {})
            else:
                continue

            train = model_data.get("training_time", 0)
            infer = model_data.get("inference_speed", 0)
            mem = abs(model_data.get("memory_usage_gb", 0))
            anomaly = model_data.get("anomaly_rate", 0)

            html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>{self.framework_names[fw]}</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train:.2f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{infer:.0f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{mem:.2f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{anomaly:.2f}</td>
            </tr>
            """

        html += "</table>"

        div = Div(text=html, width=900, height=400)

        output_file(self.output_dir / "ml_summary_table.html")
        save(div)
        print("  Saved: ml_summary_table.html")

        return div

    def plot_ml_framework_ranking(self, data):
        """ML framework ranking (horizontal bars)"""
        print("\nCreating ML framework ranking...")

        # Calculate composite scores (lower training time + higher inference = better)
        scores = {}
        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == "sklearn":
                model_data = data[fw].get("isolation_forest", {})
            elif fw in ["pytorch", "tensorflow", "jax"]:
                model_data = data[fw].get(f"{fw}_autoencoder", {})
            elif fw == "xgboost":
                model_data = data[fw].get("xgboost_detector", {})
            else:
                continue

            train = model_data.get("training_time", 0)
            infer = model_data.get("inference_speed", 0)

            # Composite score (normalize and combine)
            if train > 0 and infer > 0:
                scores[self.framework_names[fw]] = infer / train  # Higher is better

        # Sort by score
        sorted_fws = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fw_names = [fw[0] for fw in sorted_fws]
        score_values = [fw[1] for fw in sorted_fws]

        p = figure(
            y_range=fw_names,
            title="ML Framework Performance Ranking",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
        )

        p.hbar(y=fw_names, right=score_values, height=0.7, color="#e74c3c")

        p.xaxis.axis_label = "Performance Score (Inference/Training)"
        p.yaxis.axis_label = "Framework"

        output_file(self.output_dir / "ml_framework_ranking_interactive.html")
        save(p)
        print("  Saved: ml_framework_ranking_interactive.html")

        return p

    def create_combined_dashboard(self, dp_data, ml_data):
        """Create a combined dashboard with multiple charts"""
        print("\nCreating combined dashboard...")

        # Data Processing charts (basic)
        dp_exec = self.plot_dp_execution_time(dp_data, "10M")
        dp_ops = self.plot_dp_operation_breakdown(dp_data, "10M")
        dp_scale = self.plot_dp_scalability(dp_data)
        dp_mem = self.plot_dp_memory_usage(dp_data, "10M")

        # Data Processing charts (advanced)
        dp_stacked = self.plot_dp_operation_breakdown_stacked(dp_data, "10M")
        dp_scatter = self.plot_dp_memory_vs_time_scatter(dp_data)
        dp_radar = self.plot_dp_performance_radar(dp_data, "10M")
        dp_rankings = self.plot_dp_performance_rankings(dp_data, "10M")
        dp_table = self.plot_dp_summary_table(dp_data, "10M")

        # ML/DL charts (basic)
        ml_train = self.plot_ml_training_time(ml_data)
        ml_infer = self.plot_ml_inference_speed(ml_data)
        ml_mem = self.plot_ml_memory_usage(ml_data)
        ml_anomaly = self.plot_ml_anomaly_rate(ml_data)

        # ML/DL charts (advanced)
        ml_scatter = self.plot_ml_training_vs_inference(ml_data)
        ml_radar = self.plot_ml_framework_radar(ml_data)
        ml_multi = self.plot_ml_multi_metric_comparison(ml_data)
        ml_ranking = self.plot_ml_framework_ranking(ml_data)
        ml_table = self.plot_ml_summary_table(ml_data)

        # Note: Bokeh dashboard layout would be more complex
        # Individual files already created above
        print("\n  Individual chart files created. For full dashboard, use Streamlit.")
        print(f"  Total charts generated: 24")

    def generate_all_visualizations(self):
        """Generate all Bokeh visualizations"""
        print("=" * 80)
        print("BOKEH INTERACTIVE VISUALIZATIONS")
        print("=" * 80)

        # Load data
        dp_data = self.load_data_processing_results()
        ml_data = self.load_ml_results()

        # Generate charts
        self.create_combined_dashboard(dp_data, ml_data)

        print("\n" + "=" * 80)
        print("BOKEH VISUALIZATIONS COMPLETE")
        print(f"Charts saved to: {self.output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    visualizer = BokehVisualizer()
    visualizer.generate_all_visualizations()
