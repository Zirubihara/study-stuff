"""
Create individual Matplotlib charts for each operation to better compare libraries
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class OperationSpecificChartsMatplotlib:
    """Generate separate Matplotlib charts for each operation"""

    def __init__(self, results_dir="../../results", output_dir="./output"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.libraries = ['pandas', 'polars', 'pyarrow', 'dask', 'spark']
        self.operations = ['loading', 'cleaning', 'aggregation',
                          'sorting', 'filtering', 'correlation']

        self.lib_colors = {
            'pandas': '#1f77b4',
            'polars': '#ff7f0e',
            'pyarrow': '#2ca02c',
            'dask': '#d62728',
            'spark': '#9467bd'
        }

    def load_data(self, dataset_size='10M'):
        """Load benchmark data for specified dataset size"""
        print(f"Loading data for {dataset_size} dataset...")
        data = {}

        for lib in self.libraries:
            filename = f"performance_metrics_{lib}_{dataset_size}.json"
            filepath = self.results_dir / filename

            if filepath.exists():
                with open(filepath, 'r') as f:
                    data[lib] = json.load(f)
                print(f"  Loaded: {lib}")
            else:
                print(f"  Missing: {lib}")

        return data

    def create_operation_chart(self, data, operation, dataset_size='10M'):
        """Create a single chart for one operation comparing all libraries"""
        print(f"\nCreating chart for {operation} operation...")

        lib_names = []
        times = []
        colors = []

        for lib in self.libraries:
            if lib not in data:
                continue

            key = f"{operation}_time_mean"
            time_val = data[lib].get(key, 0)

            lib_names.append(lib.capitalize())
            times.append(time_val)
            colors.append(self.lib_colors[lib])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bars
        bars = ax.bar(lib_names, times, color=colors, alpha=0.8, width=0.6)

        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.4f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Styling
        ax.set_xlabel('Data Processing Library', fontsize=13, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_title(f'{operation.capitalize()} Performance - {dataset_size} Dataset',
                    fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

        # Improve tick labels
        ax.tick_params(axis='both', labelsize=11)

        plt.tight_layout()

        # Save
        output_filename = f'op_{operation}_{dataset_size}.png'
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_filename}")

    def generate_all_operation_charts(self, dataset_size='10M'):
        """Generate separate charts for all operations"""
        print("="*80)
        print(f"GENERATING MATPLOTLIB OPERATION CHARTS - {dataset_size}")
        print("="*80)

        # Load data
        data = self.load_data(dataset_size)

        if not data:
            print("No data found!")
            return

        # Create a chart for each operation
        for operation in self.operations:
            self.create_operation_chart(data, operation, dataset_size)

        print("\n" + "="*80)
        print(f"COMPLETE - Generated {len(self.operations)} PNG charts")
        print(f"Charts saved to: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    generator = OperationSpecificChartsMatplotlib()
    generator.generate_all_operation_charts('10M')
