"""
Generate professional charts and visualizations for thesis from benchmark results.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend

# Set professional styling
try:
    plt.style.use("seaborn-v0_8")
except Exception:
    plt.style.use("default")

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 11


def load_benchmark_data():
    """Load all benchmark JSON files and organize data."""
    data = {}

    # Find all performance metric files
    json_files = list(Path("../results").glob("performance_metrics_*_*.json"))

    for file_path in json_files:
        filename = file_path.stem
        parts = filename.split("_")
        if len(parts) >= 3:
            library = parts[2]  # e.g., 'pandas'
            size = parts[3]  # e.g., '100K'

            with open(file_path, "r") as f:
                json_data = json.load(f)

            if library not in data:
                data[library] = {}
            data[library][size] = json_data

    return data


def create_execution_time_comparison():
    """Create bar chart comparing execution times with error bars."""
    data = load_benchmark_data()

    operations = [
        "loading",
        "cleaning",
        "aggregation",
        "sorting",
        "filtering",
        "correlation",
    ]
    libraries = list(data.keys())
    dataset_sizes = ["100K", "500K"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for idx, size in enumerate(dataset_sizes):
        ax = axes[idx]

        # Prepare data for plotting
        x = np.arange(len(operations))
        width = 0.2
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, library in enumerate(libraries):
            if size in data[library]:
                means = []
                stds = []

                for op in operations:
                    mean_key = f"{op}_time_mean"
                    std_key = f"{op}_time_std"
                    means.append(data[library][size].get(mean_key, 0))
                    stds.append(data[library][size].get(std_key, 0))

                bars = ax.bar(
                    x + i * width,
                    means,
                    width,
                    yerr=stds,
                    capsize=3,
                    label=library.capitalize(),
                    color=colors[i % len(colors)],
                    alpha=0.8,
                )

        ax.set_xlabel("Operations")
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_title(f"Execution Time Comparison - {size} Dataset")
        ax.set_xticks(x + width * (len(libraries) - 1) / 2)
        ax.set_xticklabels([op.capitalize() for op in operations], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../charts/execution_time_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_memory_usage_comparison():
    """Create bar chart comparing memory usage with error bars."""
    data = load_benchmark_data()

    operations = [
        "loading",
        "cleaning",
        "aggregation",
        "sorting",
        "filtering",
        "correlation",
    ]
    libraries = list(data.keys())
    dataset_sizes = ["100K", "500K"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for idx, size in enumerate(dataset_sizes):
        ax = axes[idx]

        # Prepare data for plotting
        x = np.arange(len(operations))
        width = 0.2
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, library in enumerate(libraries):
            if size in data[library]:
                means = []
                stds = []

                for op in operations:
                    mean_key = f"{op}_memory_mean"
                    std_key = f"{op}_memory_std"
                    means.append(data[library][size].get(mean_key, 0))
                    stds.append(data[library][size].get(std_key, 0))

                bars = ax.bar(
                    x + i * width,
                    means,
                    width,
                    yerr=stds,
                    capsize=3,
                    label=library.capitalize(),
                    color=colors[i % len(colors)],
                    alpha=0.8,
                )

        ax.set_xlabel("Operations")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title(f"Memory Usage Comparison - {size} Dataset")
        ax.set_xticks(x + width * (len(libraries) - 1) / 2)
        ax.set_xticklabels([op.capitalize() for op in operations], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../charts/memory_usage_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_scalability_analysis():
    """Create line chart showing how performance scales with dataset size."""
    data = load_benchmark_data()

    # Total execution time scalability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    libraries = list(data.keys())
    sizes = ["100K", "500K"]
    size_values = [100000, 500000]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Execution time scalability
    for i, library in enumerate(libraries):
        total_times = []
        total_stds = []

        for size in sizes:
            if size in data[library]:
                # Sum all operation times
                operations = [
                    "loading",
                    "cleaning",
                    "aggregation",
                    "sorting",
                    "filtering",
                    "correlation",
                ]
                total_time = sum(
                    data[library][size].get(f"{op}_time_mean", 0) for op in operations
                )
                total_std = np.sqrt(
                    sum(
                        data[library][size].get(f"{op}_time_std", 0) ** 2
                        for op in operations
                    )
                )

                total_times.append(total_time)
                total_stds.append(total_std)

        if len(total_times) == len(sizes):
            ax1.errorbar(
                size_values,
                total_times,
                yerr=total_stds,
                marker="o",
                linewidth=2,
                markersize=8,
                label=library.capitalize(),
                color=colors[i % len(colors)],
            )

    ax1.set_xlabel("Dataset Size (rows)")
    ax1.set_ylabel("Total Execution Time (seconds)")
    ax1.set_title("Execution Time Scalability")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    # Memory usage scalability
    for i, library in enumerate(libraries):
        peak_memory = []

        for size in sizes:
            if size in data[library]:
                # Get peak memory usage across all operations
                operations = [
                    "loading",
                    "cleaning",
                    "aggregation",
                    "sorting",
                    "filtering",
                    "correlation",
                ]
                peak_mem = max(
                    data[library][size].get(f"{op}_memory_mean", 0) for op in operations
                )
                peak_memory.append(peak_mem)

        if len(peak_memory) == len(sizes):
            ax2.plot(
                size_values,
                peak_memory,
                marker="o",
                linewidth=2,
                markersize=8,
                label=library.capitalize(),
                color=colors[i % len(colors)],
            )

    ax2.set_xlabel("Dataset Size (rows)")
    ax2.set_ylabel("Peak Memory Usage (MB)")
    ax2.set_title("Memory Usage Scalability")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.savefig("../charts/scalability_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_operation_breakdown_heatmap():
    """Create heatmap showing operation breakdown for each library."""
    data = load_benchmark_data()

    operations = [
        "loading",
        "cleaning",
        "aggregation",
        "sorting",
        "filtering",
        "correlation",
    ]
    libraries = list(data.keys())

    # Use 100K dataset for detailed breakdown
    size = "100K"

    # Create matrix for heatmap
    matrix = []
    for library in libraries:
        if size in data[library]:
            row = []
            for op in operations:
                mean_time = data[library][size].get(f"{op}_time_mean", 0)
                row.append(mean_time)
            matrix.append(row)
        else:
            matrix.append([0] * len(operations))

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Execution Time (seconds)", rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(operations)))
    ax.set_yticks(np.arange(len(libraries)))
    ax.set_xticklabels([op.capitalize() for op in operations])
    ax.set_yticklabels([lib.capitalize() for lib in libraries])

    # Add text annotations
    for i in range(len(libraries)):
        for j in range(len(operations)):
            text = ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    ax.set_title(f"Operation Execution Time Breakdown - {size} Dataset")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    plt.savefig(
        "../charts/operation_breakdown_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_statistical_summary_table():
    """Create a summary table with key statistics."""
    data = load_benchmark_data()

    print("\\n=== BENCHMARK SUMMARY TABLE ===\\n")

    for size in ["100K", "500K"]:
        print(f"Dataset Size: {size} rows")
        print("-" * 80)
        print(
            f"{'Library':<10} {'Total Time (s)':<15} {'±STD':<10} {'Peak Memory (MB)':<20} {'Fastest Op':<15}"
        )
        print("-" * 80)

        for library in data.keys():
            if size in data[library]:
                # Calculate total time
                operations = [
                    "loading",
                    "cleaning",
                    "aggregation",
                    "sorting",
                    "filtering",
                    "correlation",
                ]
                total_time = sum(
                    data[library][size].get(f"{op}_time_mean", 0) for op in operations
                )
                total_std = np.sqrt(
                    sum(
                        data[library][size].get(f"{op}_time_std", 0) ** 2
                        for op in operations
                    )
                )

                # Find peak memory
                peak_memory = max(
                    data[library][size].get(f"{op}_memory_mean", 0) for op in operations
                )

                # Find fastest operation
                op_times = {
                    op: data[library][size].get(f"{op}_time_mean", float("inf"))
                    for op in operations
                }
                fastest_op = min(op_times.keys(), key=lambda x: op_times[x])

                print(
                    f"{library.capitalize():<10} {total_time:<15.3f} {total_std:<10.3f} {peak_memory:<20.1f} {fastest_op.capitalize():<15}"
                )

        print("\\n")


def generate_all_visualizations():
    """Generate all visualizations for the thesis."""
    print("Generating thesis visualizations...")

    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Set seaborn style
        try:
            plt.style.use("seaborn-v0_8")
        except Exception:
            plt.style.use("default")

        create_execution_time_comparison()
        create_memory_usage_comparison()
        create_scalability_analysis()
        create_operation_breakdown_heatmap()
        create_statistical_summary_table()

        print("\\nAll visualizations saved:")
        print("- execution_time_comparison.png")
        print("- memory_usage_comparison.png")
        print("- scalability_analysis.png")
        print("- operation_breakdown_heatmap.png")
        print("\\nReady for your thesis! 🎓")

    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install with: pip install matplotlib seaborn")


if __name__ == "__main__":
    generate_all_visualizations()
