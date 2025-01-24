import json
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dask.dataframe as dd
import dask
import numpy as np


@dataclass
class ProcessingResults:
    """Store results of data processing operations."""

    df: dd.DataFrame
    execution_time: float
    additional_info: Optional[Any] = None


def time_operation(operation_name: str):
    """Decorator to measure operation execution time."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            self.performance_metrics[f"{operation_name}_time_seconds"] = execution_time

            return ProcessingResults(
                df=result[0] if isinstance(result, tuple) else result,
                execution_time=execution_time,
                additional_info=(
                    result[1] if isinstance(result, tuple) and len(result) > 1 else None
                ),
            )

        return wrapper

    return decorator


class DataProcessor:
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
        "category1": "Int64",
        "category2": "Int64",
        "category3": "Int64",
        "code": str,
        "flag": "Int64",
        "value1": "Int64",
        "value2": "Int64",
    }

    def __init__(self, file_path: str, npartitions: int = 4):
        """Initialize processor with file path and number of partitions."""
        self.file_path = Path(file_path)
        self.performance_metrics: Dict[str, float] = {}
        self.npartitions = npartitions

    @time_operation("loading")
    def load_data(self) -> dd.DataFrame:
        """Load CSV data into Dask DataFrame with proper data types."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = dd.read_csv(
            str(self.file_path), 
            header=None, 
            names=self.COLUMN_NAMES, 
            dtype=self.DTYPE_MAP,
            blocksize=25_000_000  # 25 MB per partition
        )

        # Compute memory size and row count
        memory_size = dd.compute(df.memory_usage(deep=True).sum())[0] / 1024**3
        row_count = len(df)

        self.performance_metrics["memory_size_gb"] = memory_size
        self.performance_metrics["row_count"] = row_count

        return df

    @time_operation("cleaning")
    def clean_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Clean DataFrame by filling missing values."""
        return df.fillna(0)

    @time_operation("aggregation")
    def aggregate_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Group and aggregate data."""
        group_cols = ["year_month", "category1", "category2"]
        return (
            df.groupby(group_cols)
            .agg({
                "value2": ["mean", "median", "max"]
            })
            .reset_index()
        )

    @time_operation("sorting")
    def sort_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Sort DataFrame by value2 column."""
        return df.sort_values(by="value2", ascending=False)

    @time_operation("filtering")
    def filter_data(self, df: dd.DataFrame) -> Tuple[dd.DataFrame, float]:
        """Filter rows where value2 is above mean."""
        mean_value2 = df["value2"].mean().compute()
        filtered_df = df[df["value2"] > mean_value2]
        avg_filtered = filtered_df["value2"].mean().compute()
        return filtered_df, avg_filtered

    @time_operation("correlation")
    def calculate_correlation(self, df: dd.DataFrame) -> dd.DataFrame:
        """Calculate correlation matrix for numeric columns."""
        numeric_cols = df.select_dtypes(include=["Int64"]).columns
        return df[numeric_cols].corr()

    def save_performance_metrics(
        self, output_path: str = "performance_metrics_dask.json"
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

            # Compute results (Dask is lazy, so we need to compute)
            df_agg_computed = df_agg.compute()
            df_sorted_computed = df_sorted.compute()
            df_filtered_computed = df_filtered.df.compute()
            correlation_computed = correlation.df.compute()

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

            return {
                "aggregated": df_agg_computed,
                "sorted": df_sorted_computed,
                "filtered": df_filtered_computed,
                "correlation": correlation_computed,
            }

        except Exception as e:
            print(f"Error during data processing: {e}")
            raise


def main():
    """Main execution function."""
    csv_path = "/Users/krystianswiecicki/Downloads/custom_1988_2020.csv"

    try:
        # Use distributed scheduler for better performance
        with dask.config.set(scheduler='processes'):
            processor = DataProcessor(csv_path)
            results = processor.process_data()
            processor.save_performance_metrics()
            print("Data processing completed successfully!")
            return results

    except Exception as e:
        print(f"Error in main execution: {e}")
        return None


if __name__ == "__main__":
    main()