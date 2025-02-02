import json
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import gc
import os

import dask.dataframe as dd
import pandas as pd
import polars as pl


@dataclass
class ProcessingResults:
    """Store results of data processing operations."""

    df: Union[pd.DataFrame, pl.DataFrame, dd.DataFrame]
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
                additional_info=result[1] if isinstance(result, tuple) else None,
            )

        return wrapper

    return decorator


def cleanup_memory():
    """Clean up memory between different implementations."""
    gc.collect()
    if hasattr(os, "sync"):
        os.sync()


class BaseDataProcessor:
    """Base class for data processing operations."""

    COLUMN_NAMES = [
        "year_month",
        "category1",
        "category2",
        "category3",
        "code",
        "flag",
        "value1",
        "value2",
    ]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.performance_metrics: Dict[str, float] = {}

    def save_performance_metrics(self, output_path: str):
        try:
            with open(output_path, "w") as f:
                json.dump(self.performance_metrics, f, indent=4)
            print(f"Performance metrics saved to: {output_path}")
        except Exception as e:
            print(f"Error saving performance metrics: {e}")


class PandasDataProcessor(BaseDataProcessor):
    """Memory-optimized Pandas implementation."""

    DTYPE_MAP = {
        "year_month": str,
        "category1": "int32",
        "category2": "int32",
        "category3": "int32",
        "code": str,
        "flag": "int32",
        "value1": "int32",
        "value2": "int32",
    }

    @time_operation("loading")
    def load_data(self) -> pd.DataFrame:
        chunks = []
        for chunk in pd.read_csv(
            self.file_path,
            header=None,
            names=self.COLUMN_NAMES,
            dtype=self.DTYPE_MAP,
            chunksize=100000,
        ):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

        self.performance_metrics["memory_size_gb"] = df.memory_usage(
            deep=True
        ).sum() / (1024**3)
        self.performance_metrics["row_count"] = len(df)
        return df

    @time_operation("cleaning")
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(0)

    @time_operation("aggregation")
    def aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["year_month", "category1", "category2"]
        return (
            df.groupby(group_cols)
            .agg(
                value2_mean=("value2", "mean"),
                value2_median=("value2", "median"),
                value2_max=("value2", "max"),
            )
            .reset_index()
        )

    @time_operation("sorting")
    def sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values("value2", ascending=False)

    @time_operation("filtering")
    def filter_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        mean_value2 = df["value2"].mean()
        filtered_df = df[df["value2"] > mean_value2]
        avg_filtered = filtered_df["value2"].mean()
        return filtered_df, avg_filtered

    @time_operation("correlation")
    def calculate_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=["int32", "int64"]).columns
        return df[numeric_cols].corr()


class PolarsDataProcessor(BaseDataProcessor):
    """Memory-optimized Polars implementation."""

    SCHEMA = {
        "year_month": pl.Utf8,
        "category1": pl.Int32,
        "category2": pl.Int32,
        "category3": pl.Int32,
        "code": pl.Utf8,
        "flag": pl.Int32,
        "value1": pl.Int32,
        "value2": pl.Int32,
    }

    @time_operation("loading")
    def load_data(self) -> pl.DataFrame:
        df = pl.scan_csv(
            self.file_path,
            has_header=False,
            new_columns=self.COLUMN_NAMES,
            schema=self.SCHEMA,
        ).collect()

        self.performance_metrics["memory_size_gb"] = df.estimated_size() / (1024**3)
        self.performance_metrics["row_count"] = df.height
        return df

    @time_operation("cleaning")
    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.fill_null(0)

    @time_operation("aggregation")
    def aggregate_data(self, df: pl.DataFrame) -> pl.DataFrame:
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
        return df.sort("value2", descending=True)

    @time_operation("filtering")
    def filter_data(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, float]:
        mean_value2 = df["value2"].mean()
        filtered_df = df.filter(pl.col("value2") > mean_value2)
        avg_filtered = filtered_df["value2"].mean()
        return filtered_df, avg_filtered

    @time_operation("correlation")
    def calculate_correlation(self, df: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [
            col for col, dtype in df.schema.items() if dtype in [pl.Int32, pl.Int64]
        ]
        return df.select(numeric_cols).corr()


class DaskDataProcessor(BaseDataProcessor):
    """Memory-optimized Dask implementation."""

    DTYPE_MAP = {
        "year_month": str,
        "category1": "int32",
        "category2": "int32",
        "category3": "int32",
        "code": str,
        "flag": "int32",
        "value1": "int32",
        "value2": "int32",
    }

    @time_operation("loading")
    def load_data(self) -> dd.DataFrame:
        partition_size = "128MB"

        df = dd.read_csv(
            str(self.file_path),
            header=None,
            names=self.COLUMN_NAMES,
            dtype=self.DTYPE_MAP,
            blocksize=partition_size,
        )

        self.performance_metrics["row_count"] = int(df.shape[0].compute())

        sample = df.get_partition(0).compute()
        avg_row_size = sample.memory_usage(deep=True).sum() / len(sample)
        total_size = avg_row_size * self.performance_metrics["row_count"]
        self.performance_metrics["memory_size_gb"] = total_size / (1024**3)

        return df

    @time_operation("cleaning")
    def clean_data(self, df: dd.DataFrame) -> dd.DataFrame:
        return df.fillna(0)

    @time_operation("aggregation")
    def aggregate_data(self, df: dd.DataFrame) -> dd.DataFrame:
        group_cols = ["year_month", "category1", "category2"]
        df = df.repartition(npartitions=max(1, df.npartitions // 2))

        agg_df = (
            df.groupby(group_cols)
            .agg({"value2": ["mean", "median", "max"]})
            .reset_index()
        )
        agg_df.columns = [*group_cols, "value2_mean", "value2_median", "value2_max"]
        return agg_df

    @time_operation("sorting")
    def sort_data(self, df: dd.DataFrame) -> dd.DataFrame:
        df = df.repartition(npartitions=max(1, df.npartitions // 4))
        return df.sort_values("value2", ascending=False)

    @time_operation("filtering")
    def filter_data(self, df: dd.DataFrame) -> Tuple[dd.DataFrame, float]:
        mean_value2 = df["value2"].mean().compute()
        filtered_df = df[df["value2"] > mean_value2]
        avg_filtered = filtered_df["value2"].mean().compute()
        return filtered_df, avg_filtered

    @time_operation("correlation")
    def calculate_correlation(self, df: dd.DataFrame) -> dd.DataFrame:
        numeric_cols = [col for col in df.columns if df[col].dtype == "int32"]
        df = df[numeric_cols].repartition(npartitions=max(1, df.npartitions // 4))
        return df.corr().compute()


def process_implementation(processor, name: str):
    """Process a single implementation with proper cleanup."""
    try:
        print(f"\nRunning {name} implementation...")
        cleanup_memory()
        print(f"Memory cleaned before {name} implementation")

        df_result = processor.load_data()
        df_clean = processor.clean_data(df_result.df)
        df_agg = processor.aggregate_data(df_clean.df)
        df_sorted = processor.sort_data(df_clean.df)
        df_filtered = processor.filter_data(df_clean.df)
        correlation = processor.calculate_correlation(df_clean.df)

        processor.performance_metrics["average_filtered_value"] = (
            df_filtered.additional_info
        )
        processor.performance_metrics["total_operation_time_seconds"] = sum(
            time
            for key, time in processor.performance_metrics.items()
            if key.endswith("_time_seconds")
        )

        processor.save_performance_metrics(f"performance_metrics_{name}.json")

        # Cleanup
        del df_result, df_clean, df_agg, df_sorted, df_filtered, correlation
        cleanup_memory()
        print(f"{name.capitalize()} processing completed successfully!")
        print(f"Memory cleaned after {name} implementation")

        return {"metrics": processor.performance_metrics.copy()}

    except Exception as e:
        print(f"Error in {name} implementation: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        cleanup_memory()


def run_comparison(file_path: str, skip_dask: bool = False):
    """Run comparison between different implementations."""
    processors = {
        "pandas": PandasDataProcessor(file_path),
        "polars": PolarsDataProcessor(file_path),
    }

    if not skip_dask:
        processors["dask"] = DaskDataProcessor(file_path)

    results = {}
    for name, processor in processors.items():
        result = process_implementation(processor, name)
        if result:
            results[name] = result
        cleanup_memory()

    return results


if __name__ == "__main__":
    CSV_PATH = "/Users/krystianswiecicki/Downloads/custom_1988_2020.csv"

    # Run Pandas and Polars first
    print("Running Pandas and Polars implementations...")
    results = run_comparison(CSV_PATH, skip_dask=True)
    cleanup_memory()

    # Run Dask separately
    print("\nRunning Dask implementation separately...")
    dask_processor = DaskDataProcessor(CSV_PATH)
    dask_result = process_implementation(dask_processor, "dask")
    if dask_result:
        results["dask"] = dask_result

    # Final cleanup
    cleanup_memory()
