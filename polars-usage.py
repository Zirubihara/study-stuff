import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps

import polars as pl


@dataclass
class ProcessingResults:
    """Store results of data processing operations."""
    df: pl.DataFrame
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
                additional_info=result[1] if isinstance(result, tuple) and len(result) > 1 else None
            )
        return wrapper
    return decorator


class PolarsDataProcessor:
    """Handle data processing operations on CSV files using Polars."""

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

    SCHEMA = {
        "year_month": pl.Utf8,  # Preserve leading zeros
        "category1": pl.Int64,
        "category2": pl.Int64,
        "category3": pl.Int64,
        "code": pl.Utf8,  # Preserve leading zeros
        "flag": pl.Int64,
        "value1": pl.Int64,
        "value2": pl.Int64,
    }

    def __init__(self, file_path: str):
        """Initialize processor with file path."""
        self.file_path = Path(file_path)
        self.performance_metrics: Dict[str, float] = {}

    def _get_size(self, df: pl.DataFrame) -> float:
        """Calculate DataFrame size in gigabytes."""
        return df.estimated_size() / (1024**3)

    @time_operation("loading")
    def load_data(self) -> pl.DataFrame:
        """Load CSV data into Polars DataFrame with proper schema."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = pl.read_csv(
            self.file_path,
            has_header=False,
            new_columns=self.COLUMN_NAMES,
            schema_overrides=self.SCHEMA,
        )

        self.performance_metrics["memory_size_gb"] = self._get_size(df)
        self.performance_metrics["row_count"] = df.height

        return df

    @time_operation("cleaning")
    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean DataFrame by filling null values with zeros."""
        return df.fill_null(0)

    @time_operation("aggregation")
    def aggregate_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Group and aggregate data using Polars expressions."""
        group_cols = ["year_month", "category1", "category2"]
        return df.group_by(group_cols).agg(
            [
                pl.col("value2").mean().alias("value2_mean"),
                pl.col("value2").median().alias("value2_median"),
                pl.col("value2").max().alias("value2_max"),
            ]
        )

    @time_operation("sorting")
    def sort_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Sort DataFrame by value2 column in descending order."""
        return df.sort("value2", descending=True)

    @time_operation("filtering")
    def filter_data(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, float]:
        """Filter rows where value2 is above mean."""
        mean_value2 = df["value2"].mean()
        filtered_df = df.filter(pl.col("value2") > mean_value2)
        avg_filtered = filtered_df["value2"].mean()
        return filtered_df, avg_filtered

    @time_operation("correlation")
    def calculate_correlation(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate correlation matrix for numeric columns."""
        numeric_cols = df.select(pl.col(pl.Int64)).columns
        return df.select(numeric_cols).corr()

    def save_performance_metrics(self, output_path: str = "performance_metrics_polars.json"):
        """Save performance metrics to JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(self.performance_metrics, f, indent=4)
            print(f"Performance metrics saved to: {output_path}")
        except Exception as e:
            print(f"Error saving performance metrics: {e}")

    def process_data(self) -> Dict[str, pl.DataFrame]:
        """Execute complete data processing pipeline."""
        try:
            print("Loading data...")
            df_result = self.load_data()
            
            print("Cleaning data...")
            df_clean = self.clean_data(df_result.df)
            
            print("Aggregating data...")
            df_agg = self.aggregate_data(df_clean.df)
            
            print("Sorting data...")
            df_sorted = self.sort_data(df_clean.df)
            
            print("Filtering data...")
            df_filtered = self.filter_data(df_clean.df)
            
            print("Calculating correlations...")
            correlation = self.calculate_correlation(df_clean.df)

            # Store filtered average
            self.performance_metrics["average_filtered_value"] = df_filtered.additional_info

            # Calculate total processing time
            self.performance_metrics["total_operation_time_seconds"] = sum(
                time for key, time in self.performance_metrics.items()
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
    """Main execution function."""
    csv_path = "/Users/krystianswiecicki/Downloads/custom_1988_2020.csv"

    try:
        processor = PolarsDataProcessor(csv_path)
        results = processor.process_data()
        processor.save_performance_metrics()
        print("Data processing completed successfully!")
        return results

    except Exception as e:
        print(f"Error in main execution: {e}")
        return None


if __name__ == "__main__":
    main()