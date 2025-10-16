"""
Data Processing Libraries Visualization using Matplotlib
Creates publication-quality static charts for thesis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class DataProcessingVisualizerMatplotlib:
    """Matplotlib visualizations for data processing benchmark comparison"""

    def __init__(self, results_dir="../results", output_dir="./output"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.libraries = ['pandas', 'polars', 'pyarrow', 'dask', 'spark']
        self.dataset_sizes = ['5M', '10M', '50M']
        self.operations = ['loading', 'cleaning', 'aggregation', 'sorting', 'filtering', 'correlation']
        self.default_size = '10M'  # Standardized dataset size for comparisons

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
                    with open(filepath, 'r') as f:
                        data[lib][size] = json.load(f)
                    print(f"  Loaded: {filename}")
                else:
                    print(f"  Missing: {filename}")

        return data

    def plot_execution_time_comparison(self, data):
        """Compare total execution times across libraries and dataset sizes"""
        print("\nCreating execution time comparison chart...")

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(self.dataset_sizes))
        width = 0.15

        for i, lib in enumerate(self.libraries):
            times = []
            for size in self.dataset_sizes:
                if size in data[lib]:
                    times.append(data[lib][size].get('total_operation_time_mean', 0))
                else:
                    times.append(0)

            offset = (i - len(self.libraries)/2) * width
            bars = ax.bar(x + offset, times, width, label=lib.capitalize())

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}s',
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Data Processing Performance: Total Execution Time Comparison',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.dataset_sizes)
        ax.legend(title='Library', loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_file = self.output_dir / 'dp_execution_time.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_operation_breakdown(self, data, dataset_size='10M'):
        """Show breakdown of time spent in each operation"""
        print(f"\nCreating operation breakdown chart for {dataset_size}...")

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(self.operations))
        width = 0.15

        for i, lib in enumerate(self.libraries):
            if dataset_size not in data[lib]:
                continue

            times = []
            for op in self.operations:
                key = f"{op}_time_mean"
                times.append(data[lib][dataset_size].get(key, 0))

            offset = (i - len(self.libraries)/2) * width
            ax.bar(x + offset, times, width, label=lib.capitalize())

        ax.set_xlabel('Operation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Operation Breakdown: {dataset_size} Dataset',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([op.capitalize() for op in self.operations], rotation=45, ha='right')
        ax.legend(title='Library', loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_file = self.output_dir / 'dp_operation_breakdown.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_memory_usage(self, data):
        """Compare memory usage across libraries"""
        print("\nCreating memory usage comparison chart...")

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(self.dataset_sizes))
        width = 0.15

        for i, lib in enumerate(self.libraries):
            memory = []
            for size in self.dataset_sizes:
                if size in data[lib]:
                    # Sum memory from loading and cleaning operations
                    load_mem = data[lib][size].get('loading_memory_mean', 0)
                    clean_mem = data[lib][size].get('cleaning_memory_mean', 0)
                    total_mem = (load_mem + clean_mem) / 1024  # Convert MB to GB
                    memory.append(total_mem)
                else:
                    memory.append(0)

            offset = (i - len(self.libraries)/2) * width
            bars = ax.bar(x + offset, memory, width, label=lib.capitalize())

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}GB',
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage (GB)', fontsize=12, fontweight='bold')
        ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.dataset_sizes)
        ax.legend(title='Library', loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_file = self.output_dir / 'dp_memory_usage.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_scalability_analysis(self, data):
        """Analyze how each library scales with data size"""
        print("\nCreating scalability analysis chart...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Dataset sizes in millions
        sizes_numeric = [5, 10, 50]

        for lib in self.libraries:
            times = []
            for size in self.dataset_sizes:
                if size in data[lib]:
                    times.append(data[lib][size].get('total_operation_time_mean', 0))
                else:
                    times.append(None)

            # Filter out None values
            valid_sizes = [s for s, t in zip(sizes_numeric, times) if t is not None]
            valid_times = [t for t in times if t is not None]

            if valid_times:
                ax.plot(valid_sizes, valid_times, marker='o', linewidth=2,
                       markersize=8, label=lib.capitalize())

        ax.set_xlabel('Dataset Size (Million Rows)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Scalability Analysis: Performance vs Dataset Size',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Library')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()
        output_file = self.output_dir / 'dp_scalability.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def plot_performance_rankings(self, data, dataset_size='10M'):
        """Create a ranking visualization for different operations"""
        print(f"\nCreating performance rankings for {dataset_size}...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, op in enumerate(self.operations):
            ax = axes[idx]

            lib_names = []
            times = []

            for lib in self.libraries:
                if dataset_size in data[lib]:
                    key = f"{op}_time_mean"
                    time = data[lib][dataset_size].get(key, 0)
                    if time > 0:
                        lib_names.append(lib.capitalize())
                        times.append(time)

            # Sort by time (fastest first)
            sorted_data = sorted(zip(lib_names, times), key=lambda x: x[1])
            if sorted_data:
                lib_names, times = zip(*sorted_data)

                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(lib_names)))
                bars = ax.barh(lib_names, times, color=colors)

                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                           f'{width:.2f}s',
                           ha='left', va='center', fontsize=9)

                ax.set_xlabel('Time (seconds)', fontsize=10)
                ax.set_title(f'{op.capitalize()} Performance', fontsize=11, fontweight='bold')
                ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.suptitle(f'Performance Rankings by Operation - {dataset_size} Dataset',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = self.output_dir / 'dp_performance_rankings.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def create_summary_table(self, data, dataset_size='10M'):
        """Create a summary table visualization"""
        print(f"\nCreating summary table for {dataset_size}...")

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        headers = ['Library', 'Total Time', 'Loading', 'Cleaning', 'Aggregation',
                  'Sorting', 'Filtering', 'Correlation']
        table_data = []

        for lib in self.libraries:
            if dataset_size not in data[lib]:
                continue

            row = [lib.capitalize()]
            d = data[lib][dataset_size]

            # Add times
            row.append(f"{d.get('total_operation_time_mean', 0):.2f}s")
            for op in self.operations:
                row.append(f"{d.get(f'{op}_time_mean', 0):.2f}s")

            table_data.append(row)

        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.12] * len(headers))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title(f'Performance Summary Table - {dataset_size} Dataset',
                 fontsize=14, fontweight='bold', pad=20)

        output_file = self.output_dir / 'dp_summary_table.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    def generate_all_visualizations(self):
        """Generate all Matplotlib visualizations"""
        print("="*80)
        print("MATPLOTLIB DATA PROCESSING VISUALIZATIONS")
        print("="*80)

        data = self.load_benchmark_data()

        self.plot_execution_time_comparison(data)
        self.plot_memory_usage(data)
        self.plot_scalability_analysis(data)

        for size in ['10M']:  # Focus on 10M for detailed analysis
            self.plot_operation_breakdown(data, size)
            self.plot_performance_rankings(data, size)
            try:
                self.create_summary_table(data, size)
            except Exception as e:
                print(f"  Warning: Could not create summary table: {e}")
                # Create placeholder
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.axis('off')
                ax.text(0.5, 0.5, 'Summary table data not available',
                        ha='center', va='center', fontsize=16)
                output_file = self.output_dir / 'dp_summary_table.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"  Saved placeholder: {output_file}")
                plt.close()

        print("\n" + "="*80)
        print("MATPLOTLIB DATA PROCESSING COMPLETE - 6 STANDARDIZED CHARTS")
        print(f"Charts saved to: {self.output_dir}")
        print("  1. dp_execution_time.png")
        print("  2. dp_operation_breakdown.png")
        print("  3. dp_memory_usage.png")
        print("  4. dp_scalability.png")
        print("  5. dp_performance_rankings.png")
        print("  6. dp_summary_table.png")
        print("="*80)


if __name__ == "__main__":
    visualizer = DataProcessingVisualizerMatplotlib()
    visualizer.generate_all_visualizations()
