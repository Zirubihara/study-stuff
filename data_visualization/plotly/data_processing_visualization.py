"""
Data Processing Libraries Visualization using Plotly
Creates interactive web-based visualizations for thesis
"""

import json
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataProcessingVisualizerPlotly:
    """Plotly interactive visualizations for data processing benchmarks"""

    def __init__(self, results_dir="../../results", output_dir="./output"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        # Color scheme
        self.colors = {
            "pandas": "#1f77b4",
            "polars": "#ff7f0e",
            "pyarrow": "#2ca02c",
            "dask": "#d62728",
            "spark": "#9467bd",
        }

    def load_benchmark_data(self):
        """Load all benchmark result files"""
        print("Loading benchmark data...")
        data = {}

        for lib in self.libraries:
            data[lib] = {}
            for size in self.dataset_sizes:
                filename = f"performance_metrics_{lib}_{size}.json"
                filepath = self.results_dir / filename

                if filepath.exists():
                    with open(filepath, "r") as f:
                        data[lib][size] = json.load(f)
                    print(f"  Loaded: {filename}")
                else:
                    print(f"  Missing: {filename}")

        return data

    def plot_execution_time_interactive(self, data):
        """Interactive execution time comparison"""
        print("\nCreating interactive execution time chart...")

        fig = go.Figure()

        for lib in self.libraries:
            times = []
            hover_texts = []
            for size in self.dataset_sizes:
                if size in data[lib]:
                    time = data[lib][size].get("total_operation_time_mean", 0)
                    times.append(time)
                    hover_texts.append(
                        f"Library: {lib.capitalize()}<br>"
                        f"Dataset: {size}<br>"
                        f"Time: {time:.2f}s"
                    )
                else:
                    times.append(None)
                    hover_texts.append("")

            fig.add_trace(
                go.Bar(
                    name=lib.capitalize(),
                    x=self.dataset_sizes,
                    y=times,
                    marker_color=self.colors[lib],
                    hovertext=hover_texts,
                    hoverinfo="text",
                )
            )

        fig.update_layout(
            title="Data Processing Performance: Total Execution Time Comparison",
            xaxis_title="Dataset Size",
            yaxis_title="Total Execution Time (seconds)",
            barmode="group",
            hovermode="closest",
            template="plotly_white",
            height=600,
            font=dict(size=12),
            legend=dict(title="Library"),
        )

        output_file = self.output_dir / "dp_execution_time.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_operation_heatmap(self, data):
        """Create interactive heatmap of operation times"""
        print("\nCreating operation heatmap...")

        # Prepare data for 10M dataset
        dataset_size = "10M"
        heatmap_data = []
        lib_labels = []

        for lib in self.libraries:
            if dataset_size in data[lib]:
                row = []
                for op in self.operations:
                    key = f"{op}_time_mean"
                    row.append(data[lib][dataset_size].get(key, 0))
                heatmap_data.append(row)
                lib_labels.append(lib.capitalize())

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=[op.capitalize() for op in self.operations],
                y=lib_labels,
                colorscale="RdYlGn_r",
                text=[[f"{val:.2f}s" for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Time (s)"),
            )
        )

        fig.update_layout(
            title=f"Operation Performance Heatmap - {dataset_size} Dataset",
            xaxis_title="Operation",
            yaxis_title="Library",
            height=500,
            template="plotly_white",
        )

        output_file = self.output_dir / "dp_operation_breakdown.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_scalability_interactive(self, data):
        """Interactive scalability analysis"""
        print("\nCreating interactive scalability chart...")

        fig = go.Figure()

        sizes_numeric = [5, 10, 50]

        for lib in self.libraries:
            times = []
            hover_texts = []
            for i, size in enumerate(self.dataset_sizes):
                if size in data[lib]:
                    time = data[lib][size].get("total_operation_time_mean", 0)
                    times.append(time)
                    speed = f"{sizes_numeric[i]/time*1e6:.0f}" if time > 0 else "N/A"
                    hover_texts.append(
                        f"Library: {lib.capitalize()}<br>"
                        f"Dataset: {sizes_numeric[i]}M rows<br>"
                        f"Time: {time:.2f}s<br>"
                        f"Speed: {speed} rows/sec"
                    )
                else:
                    times.append(None)
                    hover_texts.append("")

            valid_sizes = [s for s, t in zip(sizes_numeric, times) if t is not None]
            valid_times = [t for t in times if t is not None]
            valid_hover = [h for h, t in zip(hover_texts, times) if t is not None]

            if valid_times:
                fig.add_trace(
                    go.Scatter(
                        name=lib.capitalize(),
                        x=valid_sizes,
                        y=valid_times,
                        mode="lines+markers",
                        marker=dict(size=10, color=self.colors[lib]),
                        line=dict(width=2, color=self.colors[lib]),
                        hovertext=valid_hover,
                        hoverinfo="text",
                    )
                )

        fig.update_layout(
            title="Scalability Analysis: Performance vs Dataset Size",
            xaxis_title="Dataset Size (Million Rows)",
            yaxis_title="Total Execution Time (seconds)",
            xaxis_type="log",
            yaxis_type="log",
            hovermode="closest",
            template="plotly_white",
            height=600,
            font=dict(size=12),
            legend=dict(title="Library"),
        )

        output_file = self.output_dir / "dp_scalability.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_operation_breakdown_stacked(self, data, dataset_size="10M"):
        """Stacked bar chart showing operation breakdown"""
        print(f"\nCreating stacked operation breakdown for {dataset_size}...")

        fig = go.Figure()

        for op in self.operations:
            times = []
            for lib in self.libraries:
                if dataset_size in data[lib]:
                    key = f"{op}_time_mean"
                    times.append(data[lib][dataset_size].get(key, 0))
                else:
                    times.append(0)

            fig.add_trace(
                go.Bar(
                    name=op.capitalize(),
                    x=[lib.capitalize() for lib in self.libraries],
                    y=times,
                    hovertemplate="<b>%{x}</b><br>"
                    + f"{op.capitalize()}: %{{y:.2f}}s<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"Operation Breakdown by Library - {dataset_size} Dataset",
            xaxis_title="Library",
            yaxis_title="Execution Time (seconds)",
            barmode="stack",
            hovermode="closest",
            template="plotly_white",
            height=600,
            font=dict(size=12),
            legend=dict(title="Operation"),
        )

        output_file = (
            self.output_dir / f"operation_breakdown_stacked_{dataset_size}.html"
        )
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_memory_vs_time_scatter(self, data):
        """Scatter plot: Memory vs Time trade-off"""
        print("\nCreating memory vs time scatter plot...")

        fig = go.Figure()

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
                    fig.add_trace(
                        go.Scatter(
                            x=[total_mem],
                            y=[total_time],
                            mode="markers+text",
                            name=f"{lib.capitalize()} ({size})",
                            marker=dict(size=15, color=self.colors[lib]),
                            text=[f"{lib}<br>{size}"],
                            textposition="top center",
                            hovertemplate=f"<b>{lib.capitalize()}</b><br>"
                            + f"Dataset: {size}<br>"
                            + f"Memory: {total_mem:.2f} GB<br>"
                            + f"Time: {total_time:.2f}s<extra></extra>",
                        )
                    )

        fig.update_layout(
            title="Memory vs Time Trade-off Analysis",
            xaxis_title="Memory Usage (GB)",
            yaxis_title="Execution Time (seconds)",
            hovermode="closest",
            template="plotly_white",
            height=600,
            font=dict(size=12),
            showlegend=True,
        )

        output_file = self.output_dir / "memory_vs_time_scatter.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_performance_radar(self, data, dataset_size="10M"):
        """Radar chart comparing libraries across operations"""
        print(f"\nCreating radar chart for {dataset_size}...")

        fig = go.Figure()

        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue

            # Normalize times to 0-10 scale (inverted: lower time = higher score)
            times = []
            for op in self.operations:
                key = f"{op}_time_mean"
                times.append(data[lib][dataset_size].get(key, 0))

            # Find max time for normalization
            max_time = max(times) if max(times) > 0 else 1

            # Invert and normalize (fast = high score)
            scores = [10 * (1 - t / max_time) for t in times]

            fig.add_trace(
                go.Scatterpolar(
                    r=scores + [scores[0]],  # Close the polygon
                    theta=[op.capitalize() for op in self.operations]
                    + [self.operations[0].capitalize()],
                    fill="toself",
                    name=lib.capitalize(),
                    line=dict(color=self.colors[lib]),
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title=f"Performance Radar Chart - {dataset_size} Dataset<br>(Higher = Better)",
            showlegend=True,
            height=700,
            template="plotly_white",
        )

        output_file = self.output_dir / f"performance_radar_{dataset_size}.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_memory_usage(self, data, dataset_size="10M"):
        """Memory usage comparison bar chart"""
        print(f"\nCreating memory usage chart for {dataset_size}...")

        lib_names = []
        memory_values = []

        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue

            load_mem = data[lib][dataset_size].get("loading_memory_mean", 0)
            clean_mem = data[lib][dataset_size].get("cleaning_memory_mean", 0)
            total_mem = (load_mem + clean_mem) / 1024  # Convert to GB

            lib_names.append(lib.capitalize())
            memory_values.append(total_mem)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=lib_names,
                    y=memory_values,
                    marker_color=[self.colors[lib.lower()] for lib in lib_names],
                    text=[f"{m:.2f} GB" for m in memory_values],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Memory: %{y:.2f} GB<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=f"Memory Usage Comparison - {dataset_size} Dataset",
            xaxis_title="Library",
            yaxis_title="Memory Usage (GB)",
            template="plotly_white",
            height=500,
            font=dict(size=12),
        )

        output_file = self.output_dir / f"dp_memory_usage_{dataset_size}.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_performance_rankings(self, data, dataset_size="10M"):
        """Performance rankings horizontal bar chart"""
        print(f"\nCreating performance rankings for {dataset_size}...")

        # Calculate scores
        scores = {}
        for lib in self.libraries:
            if dataset_size in data[lib]:
                total_time = data[lib][dataset_size].get("total_operation_time_mean", 0)
                if total_time > 0:
                    scores[lib.capitalize()] = total_time

        # Sort by score (lower is better)
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        lib_names = [item[0] for item in sorted_items]
        times = [item[1] for item in sorted_items]

        fig = go.Figure(
            data=[
                go.Bar(
                    y=lib_names,
                    x=times,
                    orientation="h",
                    marker_color="#3498db",
                    text=[f"{t:.2f}s" for t in times],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Time: %{x:.2f}s<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=f"Performance Rankings - {dataset_size} Dataset",
            xaxis_title="Total Execution Time (seconds)",
            yaxis_title="Library",
            template="plotly_white",
            height=500,
            font=dict(size=12),
        )

        output_file = self.output_dir / f"dp_performance_rankings_{dataset_size}.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def plot_summary_table(self, data, dataset_size="10M"):
        """Summary table with all metrics"""
        print(f"\nCreating summary table for {dataset_size}...")

        # Prepare table data
        headers = [
            "Library",
            "Total Time (s)",
            "Memory (GB)",
            "Loading (s)",
            "Aggregation (s)",
            "Sorting (s)",
        ]
        rows = []

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
            sorting = d.get("sorting_time_mean", 0)

            rows.append(
                [
                    lib.capitalize(),
                    f"{total_time:.2f}",
                    f"{total_mem:.2f}",
                    f"{loading:.2f}",
                    f"{aggregation:.2f}",
                    f"{sorting:.2f}",
                ]
            )

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=headers,
                        fill_color="#3498db",
                        align="left",
                        font=dict(color="white", size=14),
                    ),
                    cells=dict(
                        values=list(zip(*rows)),  # Transpose for Plotly
                        fill_color="lavender",
                        align="left",
                        font=dict(size=12),
                        height=30,
                    ),
                )
            ]
        )

        fig.update_layout(
            title=f"Data Processing Summary Table - {dataset_size} Dataset",
            height=400,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        output_file = self.output_dir / f"dp_summary_table_{dataset_size}.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    def generate_all_visualizations(self):
        """Generate all Plotly visualizations"""
        print("=" * 80)
        print("PLOTLY INTERACTIVE DATA PROCESSING VISUALIZATIONS")
        print("=" * 80)

        data = self.load_benchmark_data()

        self.plot_execution_time_interactive(data)
        self.plot_scalability_interactive(data)
        self.plot_memory_vs_time_scatter(data)
        self.plot_operation_heatmap(data)

        for size in ["10M"]:
            self.plot_operation_breakdown_stacked(data, size)
            self.plot_performance_radar(data, size)
            self.plot_memory_usage(data, size)
            self.plot_performance_rankings(data, size)
            self.plot_summary_table(data, size)

        print("\n" + "=" * 80)
        print("PLOTLY VISUALIZATIONS COMPLETE")
        print(f"Interactive charts saved to: {self.output_dir}")
        print(f"Total charts generated: 12")
        print("=" * 80)


if __name__ == "__main__":
    visualizer = DataProcessingVisualizerPlotly()
    visualizer.generate_all_visualizations()
