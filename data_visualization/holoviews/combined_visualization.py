"""
Combined Visualizations using Holoviews
High-level declarative visualizations
for both data processing and ML/DL comparisons
"""

import json
from pathlib import Path

import holoviews as hv
import pandas as pd
from holoviews import opts

hv.extension("bokeh")  # Use Bokeh backend


class HoloviewsVisualizer:
    """Holoviews visualizations for thesis comparisons"""

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
                    with open(filepath, "r", encoding="utf-8") as f:
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
                with open(filepath, "r", encoding="utf-8") as f:
                    data[fw] = json.load(f)
                print(f"  Loaded: {fw}_anomaly_detection_results.json")

        return data

    def plot_dp_execution_time(self, data, dataset_size="10M"):
        """Data processing execution time comparison"""
        print(f"\nCreating data processing execution time chart ({dataset_size})...")

        # Prepare data
        chart_data = []
        for lib in self.libraries:
            if dataset_size in data[lib]:
                time = data[lib][dataset_size].get("total_operation_time_mean", 0)
                chart_data.append({"Library": lib.capitalize(), "Time": time})

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=["Library"], vdims=["Time"])
        bars.opts(
            opts.Bars(
                width=800,
                height=500,
                title=f"Data Processing Performance - {dataset_size} Dataset",
                xlabel="Library",
                ylabel="Total Execution Time (seconds)",
                color="Library",
                cmap="Category10",
                tools=["hover"],
                show_legend=False,
            )
        )

        # Save
        output_file = self.output_dir / f"dp_execution_time_{dataset_size}.html"
        hv.save(bars, output_file)
        print(f"  Saved: dp_execution_time_{dataset_size}.html")

        return bars

    def plot_dp_operation_breakdown(self, data, dataset_size="10M"):
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
                chart_data.append(
                    {
                        "Library": lib.capitalize(),
                        "Operation": op.capitalize(),
                        "Time": time,
                    }
                )

        df = pd.DataFrame(chart_data)

        # Create grouped bar chart
        bars = hv.Bars(df, kdims=["Operation", "Library"], vdims=["Time"])
        bars.opts(
            opts.Bars(
                width=1000,
                height=500,
                title=f"Operation Breakdown - {dataset_size} Dataset",
                xlabel="Operation",
                ylabel="Time (seconds)",
                color="Library",
                cmap="Category10",
                tools=["hover"],
                legend_position="top_left",
                xrotation=45,
            )
        )

        # Save
        output_file = self.output_dir / f"dp_operation_breakdown_{dataset_size}.html"
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
                    time = data[lib][size].get("total_operation_time_mean", 0)
                    if time > 0:
                        chart_data.append(
                            {
                                "Library": lib.capitalize(),
                                "Dataset Size (M)": sizes_numeric[i],
                                "Time": time,
                            }
                        )

        df = pd.DataFrame(chart_data)

        # Create curve chart with overlay
        overlay = hv.Overlay(
            [
                hv.Curve(
                    df[df["Library"] == lib.capitalize()],
                    kdims=["Dataset Size (M)"],
                    vdims=["Time"],
                    label=lib.capitalize(),
                )
                for lib in self.libraries
            ]
        )

        overlay.opts(
            opts.Curve(
                width=900,
                height=600,
                title="Scalability Analysis: Performance vs Dataset Size",
                xlabel="Dataset Size (Million Rows)",
                ylabel="Execution Time (seconds)",
                logx=True,
                logy=True,
                tools=["hover"],
                show_grid=True,
            ),
            opts.Overlay(legend_position="top_left"),
        )

        # Save
        output_file = self.output_dir / "dp_scalability_analysis.html"
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
                chart_data.append(
                    {"Framework": self.framework_names[fw], "Training Time": time}
                )

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=["Framework"], vdims=["Training Time"])
        bars.opts(
            opts.Bars(
                width=800,
                height=500,
                title="ML/DL Framework Training Time Comparison",
                xlabel="Framework",
                ylabel="Training Time (seconds)",
                color="Framework",
                cmap="Set2",
                tools=["hover"],
                show_legend=False,
                xrotation=45,
            )
        )

        # Save
        output_file = self.output_dir / "ml_training_time.html"
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
                chart_data.append(
                    {"Framework": self.framework_names[fw], "Inference Speed": speed}
                )

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=["Framework"], vdims=["Inference Speed"])
        bars.opts(
            opts.Bars(
                width=800,
                height=500,
                title="ML/DL Framework Inference Speed Comparison",
                xlabel="Framework",
                ylabel="Inference Speed (samples/sec)",
                color="Framework",
                cmap="Set2",
                tools=["hover"],
                show_legend=False,
                xrotation=45,
            )
        )

        # Save
        output_file = self.output_dir / "ml_inference_speed.html"
        hv.save(bars, output_file)
        print("  Saved: ml_inference_speed.html")

        return bars

    def plot_ml_comparison_heatmap(self, data):
        """ML/DL framework comparison heatmap"""
        print("\nCreating ML/DL comparison heatmap...")

        # Prepare data
        chart_data = []

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == "sklearn":
                model_data = data[fw].get("isolation_forest", {})
            elif fw in ["pytorch", "tensorflow", "jax"]:
                model_data = data[fw].get("autoencoder", {})
            elif fw == "xgboost":
                model_data = data[fw].get("xgboost_detector", {})
            else:
                continue

            # Normalize values for heatmap
            train_time = model_data.get("training_time", 0)
            infer_speed = model_data.get("inference_speed", 0) / 1000  # Scale down
            memory = abs(model_data.get("memory_usage_gb", 0))

            chart_data.append(
                {
                    "Framework": self.framework_names[fw],
                    "Metric": "Training Time",
                    "Value": train_time,
                }
            )
            chart_data.append(
                {
                    "Framework": self.framework_names[fw],
                    "Metric": "Inference Speed",
                    "Value": infer_speed,
                }
            )
            chart_data.append(
                {
                    "Framework": self.framework_names[fw],
                    "Metric": "Memory Usage",
                    "Value": memory,
                }
            )

        df = pd.DataFrame(chart_data)

        # Create heatmap
        heatmap = hv.HeatMap(df, kdims=["Metric", "Framework"], vdims=["Value"])
        heatmap.opts(
            opts.HeatMap(
                width=700,
                height=500,
                title="ML/DL Framework Multi-Metric Comparison",
                xlabel="Metric",
                ylabel="Framework",
                colorbar=True,
                cmap="RdYlGn_r",
                tools=["hover"],
                xrotation=45,
            )
        )

        # Save
        output_file = self.output_dir / "ml_comparison_heatmap.html"
        hv.save(heatmap, output_file)
        print("  Saved: ml_comparison_heatmap.html")

        return heatmap

    def plot_dp_memory_usage(self, data, dataset_size="10M"):
        """Data processing memory usage comparison"""
        print(f"\nCreating data processing memory usage chart ({dataset_size})...")

        # Prepare data
        chart_data = []
        for lib in self.libraries:
            if dataset_size in data[lib]:
                load_mem = data[lib][dataset_size].get("loading_memory_mean", 0)
                clean_mem = data[lib][dataset_size].get("cleaning_memory_mean", 0)
                total_mem = (load_mem + clean_mem) / 1024  # Convert to GB
                chart_data.append(
                    {"Library": lib.capitalize(), "Memory (GB)": total_mem}
                )

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=["Library"], vdims=["Memory (GB)"])
        bars.opts(
            opts.Bars(
                width=800,
                height=500,
                title=f"Memory Usage Comparison - {dataset_size} Dataset",
                xlabel="Library",
                ylabel="Memory Usage (GB)",
                color="Library",
                cmap="Category10",
                tools=["hover"],
                show_legend=False,
            )
        )

        # Save
        output_file = self.output_dir / f"dp_memory_usage_{dataset_size}.html"
        hv.save(bars, output_file)
        print(f"  Saved: dp_memory_usage_{dataset_size}.html")

        return bars

    def plot_ml_memory_usage(self, data):
        """ML/DL memory usage comparison"""
        print("\nCreating ML/DL memory usage chart...")

        # Prepare data
        chart_data = []
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

            chart_data.append(
                {"Framework": self.framework_names[fw], "Memory (GB)": abs(mem)}
            )

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=["Framework"], vdims=["Memory (GB)"])
        bars.opts(
            opts.Bars(
                width=800,
                height=500,
                title="ML/DL Framework Memory Usage Comparison",
                xlabel="Framework",
                ylabel="Memory Usage (GB)",
                color="Framework",
                cmap="Set2",
                tools=["hover"],
                show_legend=False,
                xrotation=45,
            )
        )

        # Save
        output_file = self.output_dir / "ml_memory_usage.html"
        hv.save(bars, output_file)
        print("  Saved: ml_memory_usage.html")

        return bars

    def plot_ml_anomaly_rate(self, data):
        """ML/DL anomaly detection rate comparison"""
        print("\nCreating ML/DL anomaly rate chart...")

        # Prepare data
        chart_data = []
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
                chart_data.append(
                    {"Framework": self.framework_names[fw], "Anomaly Rate (%)": rate}
                )

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=["Framework"], vdims=["Anomaly Rate (%)"])
        bars.opts(
            opts.Bars(
                width=800,
                height=500,
                title="Anomaly Detection Rate Comparison",
                xlabel="Framework",
                ylabel="Anomaly Detection Rate (%)",
                color="Framework",
                cmap="Set2",
                tools=["hover"],
                show_legend=False,
                xrotation=45,
            )
        )

        # Save
        output_file = self.output_dir / "ml_anomaly_rate.html"
        hv.save(bars, output_file)
        print("  Saved: ml_anomaly_rate.html")

        return bars

    def plot_operation_chart(self, data, operation, dataset_size="10M"):
        """Create chart for a specific operation"""
        print(f"\nCreating {operation} operation chart ({dataset_size})...")

        # Prepare data
        chart_data = []
        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue

            key = f"{operation}_time_mean"
            time = data[lib][dataset_size].get(key, 0)
            chart_data.append({"Library": lib.capitalize(), "Time": time})

        df = pd.DataFrame(chart_data)

        # Create bar chart
        bars = hv.Bars(df, kdims=["Library"], vdims=["Time"])
        bars.opts(
            opts.Bars(
                width=900,
                height=500,
                title=f"{operation.capitalize()} Performance - {dataset_size}",
                xlabel="Data Processing Library",
                ylabel="Execution Time (seconds)",
                color="Library",
                cmap="Category10",
                tools=["hover"],
                show_legend=False,
            )
        )

        # Save
        output_file = self.output_dir / f"op_{operation}_{dataset_size}.html"
        hv.save(bars, output_file)
        print(f"  Saved: op_{operation}_{dataset_size}.html")

        return bars

    def plot_dp_performance_radar(self, data, dataset_size="10M"):
        """Radar chart for data processing performance"""
        print(f"\nCreating performance radar chart ({dataset_size})...")

        # Holoviews doesn't have native radar charts, use overlay of lines
        from math import pi
        import numpy as np

        categories = [op.capitalize() for op in self.operations]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]

        overlay_charts = []
        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue

            values = []
            for op in self.operations:
                key = f"{op}_time_mean"
                values.append(data[lib][dataset_size].get(key, 0))

            # Simple bar chart as Holoviews radar alternative
            chart_data = []
            for op, val in zip(self.operations, values):
                chart_data.append(
                    {"Operation": op.capitalize(), "Time": val, "Library": lib.capitalize()}
                )

        df = pd.DataFrame(chart_data)
        bars = hv.Bars(df, kdims=["Operation", "Library"], vdims=["Time"])
        bars.opts(
            opts.Bars(
                width=1000,
                height=600,
                title=f"Performance by Operation - {dataset_size}",
                xlabel="Operation",
                ylabel="Time (seconds)",
                color="Library",
                cmap="Category10",
                tools=["hover"],
                legend_position="right",
                xrotation=45,
            )
        )

        output_file = self.output_dir / f"performance_radar_{dataset_size}.html"
        hv.save(bars, output_file)
        print(f"  Saved: performance_radar_{dataset_size}.html")

        return bars

    def plot_dp_operation_breakdown_stacked(self, data, dataset_size="10M"):
        """Stacked bar chart for operation breakdown"""
        print(f"\nCreating stacked operation breakdown ({dataset_size})...")

        chart_data = []
        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue
            for op in self.operations:
                key = f"{op}_time_mean"
                time = data[lib][dataset_size].get(key, 0)
                chart_data.append(
                    {
                        "Library": lib.capitalize(),
                        "Operation": op.capitalize(),
                        "Time": time,
                    }
                )

        df = pd.DataFrame(chart_data)
        bars = hv.Bars(df, kdims=["Library"], vdims=["Time"], group="Operation")
        bars.opts(
            opts.Bars(
                width=1000,
                height=600,
                title=f"Stacked Operation Breakdown - {dataset_size}",
                xlabel="Library",
                ylabel="Time (seconds)",
                color="Operation",
                cmap="Set3",
                tools=["hover"],
                legend_position="right",
                stacked=True,
            )
        )

        output_file = self.output_dir / f"operation_breakdown_stacked_{dataset_size}.html"
        hv.save(bars, output_file)
        print(f"  Saved: operation_breakdown_stacked_{dataset_size}.html")

        return bars

    def plot_dp_memory_vs_time_scatter(self, data):
        """Scatter plot: Memory vs Time trade-off"""
        print("\nCreating memory vs time scatter plot...")

        chart_data = []
        for size in self.dataset_sizes:
            for lib in self.libraries:
                if size not in data[lib]:
                    continue

                d = data[lib][size]
                total_time = d.get("total_operation_time_mean", 0)
                load_mem = d.get("loading_memory_mean", 0)
                clean_mem = d.get("cleaning_memory_mean", 0)
                total_mem = (load_mem + clean_mem) / 1024  # GB

                if total_time > 0 and total_mem > 0:
                    chart_data.append(
                        {
                            "Memory (GB)": total_mem,
                            "Time (seconds)": total_time,
                            "Library": f"{lib.capitalize()} ({size})",
                        }
                    )

        df = pd.DataFrame(chart_data)
        scatter = hv.Scatter(df, kdims=["Memory (GB)"], vdims=["Time (seconds)"])
        scatter.opts(
            opts.Scatter(
                width=900,
                height=600,
                title="Memory vs Time Trade-off Analysis",
                xlabel="Memory Usage (GB)",
                ylabel="Execution Time (seconds)",
                color="Library",
                cmap="Category10",
                size=10,
                tools=["hover"],
                legend_position="right",
            )
        )

        output_file = self.output_dir / "memory_vs_time_scatter.html"
        hv.save(scatter, output_file)
        print("  Saved: memory_vs_time_scatter.html")

        return scatter

    def plot_dp_performance_rankings(self, data, dataset_size="10M"):
        """Performance rankings horizontal bar chart"""
        print(f"\nCreating performance rankings ({dataset_size})...")

        scores = {}
        for lib in self.libraries:
            if dataset_size in data[lib]:
                total_time = data[lib][dataset_size].get("total_operation_time_mean", 0)
                if total_time > 0:
                    scores[lib.capitalize()] = total_time

        # Sort by score (lower is better)
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        chart_data = [
            {"Library": lib, "Time": time} for lib, time in sorted_items
        ]

        df = pd.DataFrame(chart_data)
        bars = hv.Bars(df, kdims=["Library"], vdims=["Time"])
        bars.opts(
            opts.Bars(
                width=800,
                height=500,
                title=f"Performance Rankings - {dataset_size}",
                xlabel="Library",
                ylabel="Total Execution Time (seconds)",
                color="#3498db",
                tools=["hover"],
                invert_axes=True,
            )
        )

        output_file = self.output_dir / f"dp_performance_rankings_{dataset_size}.html"
        hv.save(bars, output_file)
        print(f"  Saved: dp_performance_rankings_{dataset_size}.html")

        return bars

    def plot_dp_summary_table(self, data, dataset_size="10M"):
        """Summary table for data processing (using Div)"""
        print(f"\nCreating data processing summary table ({dataset_size})...")

        from bokeh.models import Div
        from bokeh.io import save as bokeh_save

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
        output_file = self.output_dir / f"dp_summary_table_{dataset_size}.html"
        bokeh_save(div, filename=str(output_file))
        print(f"  Saved: dp_summary_table_{dataset_size}.html")

        return div

    def plot_ml_training_vs_inference(self, data):
        """Scatter plot: Training time vs Inference speed"""
        print("\nCreating ML training vs inference scatter plot...")

        chart_data = []
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

            if train > 0 and infer > 0:
                chart_data.append(
                    {
                        "Training Time": train,
                        "Inference Speed": infer,
                        "Framework": self.framework_names[fw],
                    }
                )

        df = pd.DataFrame(chart_data)
        scatter = hv.Scatter(df, kdims=["Training Time"], vdims=["Inference Speed"])
        scatter.opts(
            opts.Scatter(
                width=900,
                height=600,
                title="ML/DL Training vs Inference Trade-off",
                xlabel="Training Time (seconds)",
                ylabel="Inference Speed (samples/sec)",
                color="Framework",
                cmap="Set2",
                size=15,
                tools=["hover"],
                legend_position="right",
            )
        )

        output_file = self.output_dir / "ml_training_vs_inference_interactive.html"
        hv.save(scatter, output_file)
        print("  Saved: ml_training_vs_inference_interactive.html")

        return scatter

    def plot_ml_framework_radar(self, data):
        """Radar chart for ML framework comparison (using grouped bars)"""
        print("\nCreating ML framework radar chart...")

        chart_data = []
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

            train_time = model_data.get("training_time", 0)
            infer_speed = model_data.get("inference_speed", 0) / 1000
            memory = abs(model_data.get("memory_usage_gb", 0))

            chart_data.extend(
                [
                    {
                        "Framework": self.framework_names[fw],
                        "Metric": "Training",
                        "Value": train_time,
                    },
                    {
                        "Framework": self.framework_names[fw],
                        "Metric": "Inference",
                        "Value": infer_speed,
                    },
                    {
                        "Framework": self.framework_names[fw],
                        "Metric": "Memory",
                        "Value": memory,
                    },
                ]
            )

        df = pd.DataFrame(chart_data)
        bars = hv.Bars(df, kdims=["Metric", "Framework"], vdims=["Value"])
        bars.opts(
            opts.Bars(
                width=1000,
                height=600,
                title="ML Framework Multi-Metric Comparison",
                xlabel="Metric",
                ylabel="Value (normalized)",
                color="Framework",
                cmap="Set2",
                tools=["hover"],
                legend_position="right",
                xrotation=45,
            )
        )

        output_file = self.output_dir / "ml_framework_radar_interactive.html"
        hv.save(bars, output_file)
        print("  Saved: ml_framework_radar_interactive.html")

        return bars

    def plot_ml_multi_metric_comparison(self, data):
        """Multi-metric comparison for ML frameworks (as combined bars)"""
        print("\nCreating ML multi-metric comparison...")

        chart_data = []
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

            chart_data.extend(
                [
                    {
                        "Framework": self.framework_names[fw],
                        "Metric": "Training Time",
                        "Value": train,
                    },
                    {
                        "Framework": self.framework_names[fw],
                        "Metric": "Inference Speed",
                        "Value": infer / 1000,
                    },
                    {
                        "Framework": self.framework_names[fw],
                        "Metric": "Memory",
                        "Value": mem,
                    },
                    {
                        "Framework": self.framework_names[fw],
                        "Metric": "Anomaly Rate",
                        "Value": anomaly,
                    },
                ]
            )

        df = pd.DataFrame(chart_data)
        bars = hv.Bars(df, kdims=["Metric", "Framework"], vdims=["Value"])
        bars.opts(
            opts.Bars(
                width=1200,
                height=600,
                title="ML Framework Multi-Metric Comparison",
                xlabel="Metric",
                ylabel="Value",
                color="Framework",
                cmap="Set2",
                tools=["hover"],
                legend_position="right",
                xrotation=45,
            )
        )

        output_file = self.output_dir / "ml_multi_metric_comparison.html"
        hv.save(bars, output_file)
        print("  Saved: ml_multi_metric_comparison.html")

        return bars

    def plot_ml_framework_ranking(self, data):
        """ML framework ranking (horizontal bars)"""
        print("\nCreating ML framework ranking...")

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

            if train > 0 and infer > 0:
                scores[self.framework_names[fw]] = infer / train

        # Sort by score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        chart_data = [
            {"Framework": fw, "Score": score} for fw, score in sorted_items
        ]

        df = pd.DataFrame(chart_data)
        bars = hv.Bars(df, kdims=["Framework"], vdims=["Score"])
        bars.opts(
            opts.Bars(
                width=800,
                height=500,
                title="ML Framework Performance Ranking",
                xlabel="Framework",
                ylabel="Performance Score (Inference/Training)",
                color="#e74c3c",
                tools=["hover"],
                invert_axes=True,
            )
        )

        output_file = self.output_dir / "ml_framework_ranking_interactive.html"
        hv.save(bars, output_file)
        print("  Saved: ml_framework_ranking_interactive.html")

        return bars

    def plot_ml_summary_table(self, data):
        """Summary table for ML frameworks (using Div)"""
        print("\nCreating ML summary table...")

        from bokeh.models import Div
        from bokeh.io import save as bokeh_save

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
        output_file = self.output_dir / "ml_summary_table.html"
        bokeh_save(div, filename=str(output_file))
        print("  Saved: ml_summary_table.html")

        return div

    def generate_all_visualizations(self):
        """Generate all Holoviews visualizations"""
        print("=" * 80)
        print("HOLOVIEWS DECLARATIVE VISUALIZATIONS")
        print("=" * 80)

        # Load data
        dp_data = self.load_data_processing_results()
        ml_data = self.load_ml_results()

        if not dp_data:
            print("Warning: Data processing data not found!")
            return

        # Data Processing visualizations (basic)
        print("\n--- Data Processing Charts ---")
        self.plot_dp_execution_time(dp_data, "10M")
        self.plot_dp_operation_breakdown(dp_data, "10M")
        self.plot_dp_scalability(dp_data)
        self.plot_dp_memory_usage(dp_data, "10M")

        # Data Processing visualizations (advanced)
        print("\n--- Advanced Data Processing Charts ---")
        self.plot_dp_performance_radar(dp_data, "10M")
        self.plot_dp_operation_breakdown_stacked(dp_data, "10M")
        self.plot_dp_memory_vs_time_scatter(dp_data)
        self.plot_dp_performance_rankings(dp_data, "10M")
        self.plot_dp_summary_table(dp_data, "10M")

        # Operation-specific charts
        print("\n--- Operation-Specific Charts ---")
        for operation in self.operations:
            self.plot_operation_chart(dp_data, operation, "10M")

        # ML/DL visualizations
        if ml_data:
            print("\n--- ML/DL Framework Charts (Basic) ---")
            self.plot_ml_training_time(ml_data)
            self.plot_ml_inference_speed(ml_data)
            self.plot_ml_memory_usage(ml_data)
            self.plot_ml_anomaly_rate(ml_data)
            self.plot_ml_comparison_heatmap(ml_data)

            print("\n--- ML/DL Framework Charts (Advanced) ---")
            self.plot_ml_training_vs_inference(ml_data)
            self.plot_ml_framework_radar(ml_data)
            self.plot_ml_multi_metric_comparison(ml_data)
            self.plot_ml_framework_ranking(ml_data)
            self.plot_ml_summary_table(ml_data)
        else:
            print("Warning: ML/DL data not found!")

        print("\n" + "=" * 80)
        print("HOLOVIEWS VISUALIZATIONS COMPLETE")
        print(f"Charts saved to: {self.output_dir}")
        print("Total charts generated: 25")
        print("=" * 80)


if __name__ == "__main__":
    visualizer = HoloviewsVisualizer()
    visualizer.generate_all_visualizations()
