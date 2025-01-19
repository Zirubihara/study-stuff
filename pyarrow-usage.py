import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import streamlit as st


@dataclass
class ArrowProcessingResults:
    """Store results of PyArrow data processing operations."""

    execution_time: float
    table: Optional[pa.Table] = None
    value: Optional[Any] = None
    metadata: Optional[Dict] = None


class ArrowDataProcessor:
    """Handle data processing operations using PyArrow with Streamlit visualization."""

    def __init__(self, file_path: str):
        """Initialize processor with file path."""
        self.file_path = Path(file_path)
        self.performance_metrics: Dict[str, Union[float, int, Dict]] = {}

    def _get_numeric_columns(self, table: pa.Table) -> List[str]:
        """Get list of numeric column names from table schema."""
        return [
            f
            for f in table.schema.names
            if pa.types.is_integer(table.schema.field(f).type)
            or pa.types.is_floating(table.schema.field(f).type)
        ]

    def _get_size_in_gb(self, table: pa.Table) -> float:
        """Calculate table size in gigabytes."""
        return table.nbytes / (1024**3)

    def _time_operation(self, operation_name: str):
        """Decorator to measure operation execution time."""

        def decorator(func):
            def wrapper(*args, **kwargs) -> ArrowProcessingResults:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Store timing in performance metrics
                self.performance_metrics[f"{operation_name}_time_seconds"] = (
                    execution_time
                )

                # Handle different return types
                if isinstance(result, tuple):
                    table = result[0] if isinstance(result[0], pa.Table) else None
                    value = result[1] if len(result) > 1 else None
                    metadata = result[2] if len(result) > 2 else None
                else:
                    table = result if isinstance(result, pa.Table) else None
                    value = result if not isinstance(result, pa.Table) else None
                    metadata = None

                return ArrowProcessingResults(
                    execution_time=execution_time,
                    table=table,
                    value=value,
                    metadata=metadata,
                )

            return wrapper

        return decorator

    @_time_operation("loading")
    def load_data(self) -> pa.Table:
        """Load CSV data into PyArrow table."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        table = csv.read_csv(self.file_path)

        self.performance_metrics["memory_size_gb"] = self._get_size_in_gb(table)
        self.performance_metrics["row_count"] = table.num_rows

        return table

    @_time_operation("cleaning")
    def clean_data(self, table: pa.Table) -> pa.Table:
        """Replace null values with zeros in numeric columns."""
        numeric_cols = self._get_numeric_columns(table)

        for col in numeric_cols:
            col_array = table[col]
            if col_array.null_count > 0:
                table = table.set_column(
                    table.schema.get_field_index(col), col, pc.fill_null(col_array, 0)
                )

        return table

    @_time_operation("aggregation")
    def aggregate_data(self, table: pa.Table) -> Tuple[pa.Table, Dict]:
        """Perform grouping and aggregation operations."""
        numeric_cols = self._get_numeric_columns(table)[:3]  # First 3 numeric columns
        group_col = table.column_names[0]

        aggs = []
        for ncol in numeric_cols:
            aggs.extend([(ncol, "mean"), (ncol, "max")])

        grouped = table.group_by(group_col).aggregate(aggs)

        return grouped, {"grouped_columns": numeric_cols}

    @_time_operation("sorting")
    def sort_data(self, table: pa.Table, n: int = 5) -> pa.Table:
        """Sort table by first numeric column and return top N rows."""
        numeric_cols = self._get_numeric_columns(table)

        if not numeric_cols:
            return pa.Table.from_arrays([], schema=pa.schema([]))

        sorted_indices = pc.sort_indices(
            table, sort_keys=[(numeric_cols[0], "descending")]
        )
        return table.take(sorted_indices).slice(0, n)

    @_time_operation("filtering")
    def filter_data(self, table: pa.Table) -> Tuple[pa.Table, float]:
        """Filter rows above mean value in first numeric column."""
        numeric_cols = self._get_numeric_columns(table)

        if not numeric_cols:
            return table, 0.0

        col = numeric_cols[0]
        mean_val = pc.mean(table[col]).as_py()
        condition = pc.greater(table[col], pa.scalar(mean_val))
        filtered = table.filter(condition)
        avg_filtered = pc.mean(filtered[col]).as_py()

        return filtered, avg_filtered

    @_time_operation("correlation")
    def calculate_correlation(self, table: pa.Table) -> Dict:
        """Calculate correlation matrix for numeric columns."""
        numeric_cols = self._get_numeric_columns(table)[:3]  # First 3 numeric columns

        if len(numeric_cols) < 2:
            return {}

        df_corr = table.select(numeric_cols).to_pandas()
        return df_corr.corr().to_dict()

    def save_performance_metrics(
        self, output_path: str = "performance_data_arrow.json"
    ):
        """Save performance metrics to JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(self.performance_metrics, f, indent=4)
            st.success(f"Performance metrics saved to: {output_path}")
        except Exception as e:
            st.error(f"Error saving performance metrics: {e}")

    def display_results(self, results: Dict[str, ArrowProcessingResults]):
        """Display processing results using Streamlit."""
        st.write("## Processing Results")

        # Display timing information
        st.write("### Performance Metrics")
        st.write(f"Loading time: {results['load'].execution_time:.2f} seconds")
        st.write(f"Memory size: {self.performance_metrics['memory_size_gb']:.2f} GB")
        st.write(f"Number of rows: {self.performance_metrics['row_count']}")

        # Display grouped data
        if results["aggregate"].table is not None:
            st.write("### Grouping Results")
            st.dataframe(results["aggregate"].table.to_pandas())

        # Display top sorted rows
        if results["sort"].table is not None:
            st.write("### Top 5 Sorted Rows")
            st.dataframe(results["sort"].table.to_pandas())

        # Display filtered average
        if results["filter"].value is not None:
            st.write("### Filtered Data Statistics")
            st.write(f"Average value for filtered data: {results['filter'].value:.2f}")

        # Display correlation matrix
        if results["correlation"].value:
            st.write("### Correlation Matrix")
            corr_df = pd.DataFrame(results["correlation"].value)
            st.dataframe(corr_df)

        # Display table schema
        if results["load"].table is not None:
            st.write("### Table Schema")
            st.write(results["load"].table.schema)

            # Display descriptive statistics
            st.write("### Descriptive Statistics")
            st.write(results["load"].table.to_pandas().describe())

    def process_data(self) -> Dict[str, ArrowProcessingResults]:
        """Execute complete data processing pipeline."""
        try:
            results = {}

            # Load and process data
            results["load"] = self.load_data()
            results["clean"] = self.clean_data(results["load"].table)
            results["aggregate"] = self.aggregate_data(results["clean"].table)
            results["sort"] = self.sort_data(results["clean"].table)
            results["filter"] = self.filter_data(results["clean"].table)
            results["correlation"] = self.calculate_correlation(results["clean"].table)

            # Update performance metrics
            self.performance_metrics.update(
                {
                    "average_filtered_value": results["filter"].value,
                    "grouping_results": results["aggregate"].table.to_pydict(),
                    "top_5_sorted_rows": results["sort"].table.to_pydict(),
                    "correlation_matrix": results["correlation"].value,
                }
            )

            return results

        except Exception as e:
            st.error(f"Error during data processing: {e}")
            raise


def main():
    """Main execution function."""
    st.title("PyArrow Data Processing Dashboard")

    csv_path = "/Users/krystianswiecicki/Downloads/custom_1988_2020.csv"

    try:
        processor = ArrowDataProcessor(csv_path)
        results = processor.process_data()
        processor.display_results(results)
        processor.save_performance_metrics()
        st.success("Data processing completed successfully!")

    except Exception as e:
        st.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
