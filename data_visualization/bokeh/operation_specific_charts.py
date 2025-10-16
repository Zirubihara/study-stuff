"""
Create individual charts for each operation to better compare libraries
"""

import json
from pathlib import Path
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool


class OperationSpecificCharts:
    """Generate separate charts for each operation"""

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

        # Create data source
        source = ColumnDataSource(data=dict(
            libraries=lib_names,
            times=times,
            colors=colors
        ))

        # Create figure
        p = figure(x_range=lib_names,
                   title=f'{operation.capitalize()} Performance - {dataset_size}',
                   toolbar_location="above",
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=900, height=500)

        p.vbar(x='libraries', top='times', width=0.7,
               color='colors', source=source, alpha=0.9)

        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Library", "@libraries"),
            ("Time", "@times{0.000} seconds")
        ])
        p.add_tools(hover)

        # Styling
        p.xaxis.axis_label = "Data Processing Library"
        p.yaxis.axis_label = "Execution Time (seconds)"
        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        # Add value labels on top of bars
        p.title.text_font_size = "14pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "11pt"

        # Save
        output_filename = f'op_{operation}_{dataset_size}.html'
        output_file(self.output_dir / output_filename)
        save(p)
        print(f"  Saved: {output_filename}")

        return p

    def generate_all_operation_charts(self, dataset_size='10M'):
        """Generate separate charts for all operations"""
        print("="*80)
        print(f"GENERATING INDIVIDUAL OPERATION CHARTS - {dataset_size}")
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
        print(f"COMPLETE - Generated {len(self.operations)} charts")
        print(f"Charts saved to: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    generator = OperationSpecificCharts()
    generator.generate_all_operation_charts('10M')
