import json
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np  # needed for correlation calculation

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv


@dataclass
class ArrowProcessingResults:
    """Store results of PyArrow data processing operations."""

    execution_time: float
    table: Optional[pa.Table] = None
    value: Optional[Any] = None
    metadata: Optional[Dict] = None


def time_operation(operation_name: str):
    """Decorator to measure operation execution time."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> ArrowProcessingResults:
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            self.performance_metrics[f"{operation_name}_time_seconds"] = execution_time

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


class ArrowDataProcessor:
    """Handle data processing operations using PyArrow."""

    COLUMN_NAMES = [
        "year_month",  # Format: YYYYMM
        "category1",  # Integer category
        "category2",  # Integer category
        "category3",  # Integer category
        "code",  # String code with leading zeros
        "flag",  # Integer flag
        "value1",  # Integer value
        "value2",  # Integer value
    ]

    SCHEMA = pa.schema(
        [
            ("year_month", pa.string()),
            ("category1", pa.int64()),
            ("category2", pa.int64()),
            ("category3", pa.int64()),
            ("code", pa.string()),
            ("flag", pa.int64()),
            ("value1", pa.int64()),
            ("value2", pa.int64()),
        ]
    )

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

    @time_operation("loading")
    def load_data(self) -> pa.Table:
        """Load CSV data into PyArrow table with explicit schema."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        read_options = csv.ReadOptions(column_names=self.COLUMN_NAMES)
        convert_options = csv.ConvertOptions(column_types=self.SCHEMA)

        table = csv.read_csv(
            self.file_path, read_options=read_options, convert_options=convert_options
        )

        self.performance_metrics["memory_size_gb"] = self._get_size_in_gb(table)
        self.performance_metrics["row_count"] = table.num_rows

        return table

    @time_operation("cleaning")
    def clean_data(self, table: pa.Table) -> pa.Table:
        """Replace null values with zeros in numeric columns."""
        numeric_cols = self._get_numeric_columns(table)
        arrays = []

        for col_name in table.column_names:
            col = table[col_name]
            if col_name in numeric_cols and col.null_count > 0:
                arrays.append(pc.fill_null(col, 0))
            else:
                arrays.append(col)

        return pa.Table.from_arrays(arrays, schema=table.schema)

    def _calculate_median(self, array: pa.Array) -> float:
        """Calculate median using PyArrow's approximate_median."""
        return pc.approximate_median(array).as_py()

    @time_operation("aggregation")
    def aggregate_data(self, table: pa.Table) -> pa.Table:
        """Group and aggregate data with mean, median, and max."""
        group_cols = ["year_month", "category1", "category2"]

        # Get unique combinations using groupby
        unique_groups = []
        for group_key in group_cols:
            unique_values = pc.unique(table[group_key])
            unique_groups.append(unique_values)

        # Create all combinations of unique values
        import itertools

        group_combinations = list(itertools.product(*unique_groups))

        # For each unique combination, calculate stats
        means = []
        medians = []
        maxes = []
        final_groups = {col: [] for col in group_cols}

        for combination in group_combinations:
            # Create mask for current group
            mask = None
            for col, val in zip(group_cols, combination):
                current_mask = pc.equal(table[col], val)
                mask = current_mask if mask is None else pc.and_(mask, current_mask)

            # Skip if no matching rows
            if pc.sum(pc.cast(mask, pa.int8())).as_py() == 0:
                continue

            # Get group data
            group_data = table["value2"].filter(mask)

            # Only add group if it has data
            if len(group_data) > 0:
                for col, val in zip(group_cols, combination):
                    final_groups[col].append(val)

                # Calculate statistics
                means.append(pc.mean(group_data).as_py())
                medians.append(pc.approximate_median(group_data).as_py())
                maxes.append(pc.max(group_data).as_py())

        # Combine results
        result_arrays = [
            pa.array(final_groups[col], type=table.schema.field(col).type)
            for col in group_cols
        ] + [
            pa.array(means, type=pa.float64()),
            pa.array(medians, type=pa.float64()),
            pa.array(maxes, type=pa.float64()),
        ]

        result_schema = pa.schema(
            [
                *((col, table.schema.field(col).type) for col in group_cols),
                ("value2_mean", pa.float64()),
                ("value2_median", pa.float64()),
                ("value2_max", pa.float64()),
            ]
        )

        return pa.Table.from_arrays(result_arrays, schema=result_schema)

    @time_operation("sorting")
    def sort_data(self, table: pa.Table) -> pa.Table:
        """Sort table by value2 column (full sort)."""
        indices = pc.sort_indices(table, sort_keys=[("value2", "descending")])
        return table.take(indices)

    @time_operation("filtering")
    def filter_data(self, table: pa.Table) -> Tuple[pa.Table, float]:
        """Filter rows above mean value in value2 column."""
        mean_val = pc.mean(table["value2"]).as_py()
        mask = pc.greater(table["value2"], mean_val)
        filtered = table.filter(mask)
        avg_filtered = pc.mean(filtered["value2"]).as_py()
        return filtered, avg_filtered

    @time_operation("correlation")
    def calculate_correlation(self, table: pa.Table) -> Dict:
        """Calculate full correlation matrix for all numeric columns."""
        numeric_cols = self._get_numeric_columns(table)
        n_cols = len(numeric_cols)
        corr_matrix = np.zeros((n_cols, n_cols))

        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i <= j:  # Correlation matrix is symmetric
                    correlation = pc.covariance(table[col1], table[col2]).as_py() / (
                        pc.stddev(table[col1]).as_py() * pc.stddev(table[col2]).as_py()
                    )
                    corr_matrix[i, j] = correlation
                    if i != j:
                        corr_matrix[j, i] = correlation

        return {
            col1: {col2: corr_matrix[i, j] for j, col2 in enumerate(numeric_cols)}
            for i, col1 in enumerate(numeric_cols)
        }

    def save_performance_metrics(
        self, output_path: str = "performance_metrics_arrow.json"
    ):
        """Save performance metrics to JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(self.performance_metrics, f, indent=4)
            print(f"Performance metrics saved to: {output_path}")
        except Exception as e:
            print(f"Error saving performance metrics: {e}")

    def process_data(self) -> Dict[str, ArrowProcessingResults]:
        """Execute complete data processing pipeline."""
        try:
            results = {}

            results["load"] = self.load_data()
            results["clean"] = self.clean_data(results["load"].table)
            results["aggregate"] = self.aggregate_data(results["clean"].table)
            results["sort"] = self.sort_data(results["clean"].table)
            results["filter"] = self.filter_data(results["clean"].table)
            results["correlation"] = self.calculate_correlation(results["clean"].table)

            self.performance_metrics["total_operation_time_seconds"] = sum(
                self.performance_metrics[f"{op}_time_seconds"]
                for op in [
                    "loading",
                    "cleaning",
                    "aggregation",
                    "sorting",
                    "filtering",
                    "correlation",
                ]
            )

            return results

        except Exception as e:
            print(f"Error during data processing: {e}")
            raise


def main():
    """Main execution function."""
    csv_path = "/Users/krystianswiecicki/Downloads/custom_1988_2020.csv"

    try:
        processor = ArrowDataProcessor(csv_path)
        results = processor.process_data()
        processor.save_performance_metrics()
        print("Data processing completed successfully!")
        return results

    except Exception as e:
        print(f"Error in main execution: {e}")
        return None


if __name__ == "__main__":
    main()
