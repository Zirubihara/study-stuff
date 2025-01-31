import json
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dask.dataframe as dd


@dataclass
class ProcessingResults:
    """Store results of data processing operations (Dask-based)."""

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


class DaskDataProcessor:
    """
    Handle data processing operations on CSV files using Dask,
    mirroring the same steps as the pandas-based DataProcessor.
    """

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
        "category1": "int64",
        "category2": "int64",
        "category3": "int64",
        "code": str,
        "flag": "int64",
        "value1": "int64",
        "value2": "int64",
    }

    def __init__(self, file_path: str):
        """Initialize processor with file path."""
        self.file_path = Path(file_path)
        self.performance_metrics: Dict[str, float] = {}

    @time_operation("loading")
    def load_data(self) -> dd.DataFrame:
        """
        Load CSV data into a Dask DataFrame with proper data types.
        Forces a computation to measure row count and memory usage.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Load CSV with Dask
        df = dd.read_csv(
            str(self.file_path),
            header=None,
            names=self.COLUMN_NAMES,
            dtype=self.DTYPE_MAP,
        )

        # Force reading of data to compute row count
        row_count = df.shape[0].compute()

        # --- Workaround for memory usage issue ---
        # Step 1: Convert partition-wise memory usage to a concrete pandas Series
        mem_usage_series = df.map_partitions(
            lambda pdf: pdf.memory_usage(deep=True).sum()
        )
        mem_usage_vals = mem_usage_series.compute()  # now it's a normal pandas Series
        # Step 2: sum the values in that pandas Series
        memory_usage = mem_usage_vals.sum() / 1024**3

        # Store in performance metrics
        self.performance_metrics["memory_size_gb"] = memory_usage
        self.performance_metrics["row_count"] = row_count

        # Return the Dask DataFrame
        return df

    @time_operation("cleaning")
    def clean_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Clean DataFrame by filling missing values.
        Return a persisted DataFrame to force computation time measurement.
        """
        df_clean = df.fillna(0)
        # Persist in memory so that the fillna() step is computed here
        df_clean = df_clean.persist()
        # Trigger partial computation to measure timing
        df_clean.head()
        return df_clean

    @time_operation("aggregation")
    def aggregate_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Group and aggregate data. We measure the time by calling compute() at the end.
        """
        group_cols = ["year_month", "category1", "category2"]
        # Perform groupby and aggregation
        df_agg = df.groupby(group_cols).agg({"value2": ["mean", "median", "max"]})

        # The resulting columns have a MultiIndex; rename them for clarity
        df_agg.columns = ["value2_mean", "value2_median", "value2_max"]
        # Convert back to a regular index (similar to reset_index in pandas)
        df_agg = df_agg.reset_index()

        # Force the aggregation to compute for timing
        df_agg = df_agg.persist()
        df_agg.head()  # Trigger a real computation
        return df_agg

    @time_operation("sorting")
    def sort_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Sort DataFrame by the value2 column in descending order.
        WARNING: A global sort can be very expensive in Dask for large data.
        """
        df_sorted = df.sort_values("value2", ascending=False)
        df_sorted = df_sorted.persist()
        df_sorted.head()
        return df_sorted

    @time_operation("filtering")
    def filter_data(self, df: dd.DataFrame) -> Tuple[dd.DataFrame, float]:
        """
        Filter rows where value2 is above the mean.
        Return the filtered DataFrame and the average of 'value2' in that filtered set.
        """
        # Compute mean first
        mean_value2 = df["value2"].mean().compute()
        filtered_df = df[df["value2"] > mean_value2]

        # Persist after filtering
        filtered_df = filtered_df.persist()
        # Compute average on the filtered df
        avg_filtered = filtered_df["value2"].mean().compute()

        return filtered_df, avg_filtered

    @time_operation("correlation")
    def calculate_correlation(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.
        By default, Dask's corr() only works with numeric columns.
        """
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        corr_df = df[list(numeric_cols)].corr()

        # Force computation so we can time it
        corr_result = corr_df.compute()  # This will be a pandas DataFrame
        # Convert it back into a Dask DataFrame just to maintain the return type
        corr_result_dd = dd.from_pandas(corr_result, npartitions=1)
        return corr_result_dd

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
        """Execute complete data processing pipeline with Dask."""
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
                t
                for key, t in self.performance_metrics.items()
                if key.endswith("_time_seconds")
            )

            return {
                "aggregated": df_agg.df,
                "sorted": df_sorted.df,
                "filtered": df_filtered.df,
                "correlation": correlation.df,
            }

        except Exception as e:
            print(f"Error during data processing: {e}")
            raise


def main():
    """Main execution function using DaskDataProcessor."""
    csv_path = "/Users/krystianswiecicki/Downloads/custom_1988_2020.csv"

    try:
        processor = DaskDataProcessor(csv_path)
        results = processor.process_data()
        processor.save_performance_metrics()
        print("Data processing (Dask) completed successfully!")
        return results

    except Exception as e:
        print(f"Error in main execution: {e}")
        return None


if __name__ == "__main__":
    main()
