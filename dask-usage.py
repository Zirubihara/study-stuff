import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dask.dataframe as dd
import numpy as np


@dataclass
class ProcessingResults:
    """Store results of data processing operations."""

    df: dd.DataFrame
    execution_time: float
    additional_info: Optional[Any] = None


class DaskDataProcessor:
    """Handle data processing operations on CSV files using Dask."""

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

    DTYPE_MAP = {
        "year_month": str,
        "category1": np.int64,
        "category2": np.int64,
        "category3": np.int64,
        "code": str,
        "flag": np.int64,
        "value1": np.int64,
        "value2": np.int64,
    }

    def __init__(self, file_path: str, n_partitions: int = 4):
        """Initialize processor with file path and number of partitions."""
        self.file_path = Path(file_path)
        self.n_partitions = n_partitions
        self.performance_metrics: Dict[str, float] = {}

    def _time_operation(self, operation_name: str):
        """Decorator to measure operation execution time."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.performance_metrics[f"{operation_name}_time_seconds"] = (
                    execution_time
                )

                # Handle tuple results for operations that return additional info
                if isinstance(result, tuple):
                    df, additional_info = result
                    return ProcessingResults(
                        df=df,
                        execution_time=execution_time,
                        additional_info=additional_info,
                    )

                return ProcessingResults(df=result, execution_time=execution_time)

            return wrapper

        return decorator

    @_time_operation("loading")
    def load_data(self) -> dd.DataFrame:
        """Load CSV data into Dask DataFrame with proper data types."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = dd.read_csv(
            self.file_path,
            names=self.COLUMN_NAMES,
            dtype=self.DTYPE_MAP,
            blocksize="64MB",  # Adjust based on your needs
        ).repartition(npartitions=self.n_partitions)

        # Calculate metrics (note: these operations will trigger computation)
        self.performance_metrics["memory_size_gb"] = (
            df.memory_usage(deep=True).sum().compute() / 1024**3
        )
        self.performance_metrics["row_count"] = len(df.compute())

        return df

    @_time_operation("cleaning")
    def clean_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Clean DataFrame by filling missing values."""
        return df.fillna(0)

    @_time_operation("aggregation")
    def aggregate_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Group and aggregate data."""
        group_cols = ["year_month", "category1", "category2"]

        # Dask requires explicit metadata for groupby operations
        return (
            df.groupby(group_cols)
            .agg({"value2": ["mean", "max"]})
            .reset_index()
            .map_partitions(
                lambda pdf: pdf.assign(value2_median=pdf["value2"].median())
            )
        )

    @_time_operation("sorting")
    def sort_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Sort DataFrame by value2 column."""
        return df.sort_values(by="value2", ascending=False)

    @_time_operation("filtering")
    def filter_data(self, df: dd.DataFrame) -> Tuple[dd.DataFrame, float]:
        """Filter rows where value2 is above mean."""
        mean_value2 = df["value2"].mean().compute()
        filtered_df = df[df["value2"] > mean_value2]
        avg_filtered = filtered_df["value2"].mean().compute()
        return filtered_df, avg_filtered

    @_time_operation("correlation")
    def calculate_correlation(self, df: dd.DataFrame) -> dd.DataFrame:
        """Calculate correlation matrix for numeric columns."""
        numeric_cols = [
            col for col, dtype in df.dtypes.items() if np.issubdtype(dtype, np.number)
        ]
        return df[numeric_cols].corr().compute()

    def save_performance_metrics(
        self, output_path: str = "dask_performance_metrics.json"
    ):
        """Save performance metrics to JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(self.performance_metrics, f, indent=4)
            print(f"Performance metrics saved to: {output_path}")
        except Exception as e:
            print(f"Error saving performance metrics: {e}")

    def process_data(self):
        """Execute complete data processing pipeline."""
        try:
            # Load and process data
            df_result = self.load_data()
            df_clean = self.clean_data(df_result.df)
            df_agg = self.aggregate_data(df_clean.df)
            df_sorted = self.sort_data(df_clean.df)
            df_filtered = self.filter_data(df_clean.df)
            correlation = self.calculate_correlation(df_clean.df)

            # Store filtered average
            self.performance_metrics["average_filtered_value"] = (
                df_filtered.additional_info
            )

            # Calculate total processing time
            self.performance_metrics["total_operation_time_seconds"] = sum(
                time
                for key, time in self.performance_metrics.items()
                if key.endswith("_time_seconds")
            )

            # Compute final results
            return {
                "aggregated": df_agg.df.compute(),
                "sorted": df_sorted.df.compute(),
                "filtered": df_filtered.df.compute(),
                "correlation": correlation.df,
            }

        except Exception as e:
            print(f"Error during data processing: {e}")
            raise


def main():
    """Main execution function."""
    csv_path = "/path/to/your/csv/file.csv"

    try:
        processor = DaskDataProcessor(csv_path, n_partitions=4)
        results = processor.process_data()
        processor.save_performance_metrics()
        print("Data processing completed successfully!")
        return results

    except Exception as e:
        print(f"Error in main execution: {e}")
        return None


if __name__ == "__main__":
    main()
