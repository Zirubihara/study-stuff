"""
Create individual Plotly charts for each operation to better compare libraries
"""

import json
from pathlib import Path
import plotly.graph_objects as go


class OperationSpecificChartsPlotly:
    """Generate separate Plotly charts for each operation"""

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
        hover_texts = []

        for lib in self.libraries:
            if lib not in data:
                continue

            key = f"{operation}_time_mean"
            time_val = data[lib].get(key, 0)

            lib_names.append(lib.capitalize())
            times.append(time_val)
            colors.append(self.lib_colors[lib])
            hover_texts.append(
                f"<b>{lib.capitalize()}</b><br>"
                f"Operation: {operation.capitalize()}<br>"
                f"Time: {time_val:.4f} seconds<br>"
                f"Dataset: {dataset_size}"
            )

        # Create figure
        fig = go.Figure(data=[
            go.Bar(
                x=lib_names,
                y=times,
                marker_color=colors,
                text=[f'{t:.3f}s' for t in times],
                textposition='outside',
                hovertext=hover_texts,
                hoverinfo='text'
            )
        ])

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{operation.capitalize()} Performance - {dataset_size}',
                font=dict(size=18)
            ),
            xaxis_title="Data Processing Library",
            yaxis_title="Execution Time (seconds)",
            template='plotly_white',
            height=550,
            width=1000,
            hovermode='closest',
            font=dict(size=13),
            showlegend=False,
            yaxis=dict(rangemode='tozero')
        )

        # Save
        output_filename = f'op_{operation}_{dataset_size}.html'
        output_path = self.output_dir / output_filename
        fig.write_html(str(output_path))
        print(f"  Saved: {output_filename}")

        return fig

    def generate_all_operation_charts(self, dataset_size='10M'):
        """Generate separate charts for all operations"""
        print("="*80)
        print(f"GENERATING PLOTLY OPERATION CHARTS - {dataset_size}")
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
        print(f"COMPLETE - Generated {len(self.operations)} interactive charts")
        print(f"Charts saved to: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    generator = OperationSpecificChartsPlotly()
    generator.generate_all_operation_charts('10M')
