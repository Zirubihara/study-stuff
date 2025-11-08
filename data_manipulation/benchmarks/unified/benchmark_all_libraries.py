import gc
import json
import os
import statistics
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.compute as pc  # for compute functions
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

# Try to import the CSV module from pyarrow; if not available, fallback to dataset API.
try:
    import pyarrow.csv as pacsv
except ImportError:
    pacsv = None


@dataclass
class ProcessingResults:
    """Store results of data processing operations."""

    df: Union[pd.DataFrame, pl.DataFrame, dd.DataFrame, pa.Table]
    execution_time: float
    memory_usage_mb: float
    additional_info: Optional[Any] = None


@dataclass
class BenchmarkStats:
    """Store statistical results from multiple runs."""

    mean: float
    std_dev: float
    min_val: float
    max_val: float
    runs: List[float]


def time_operation(operation_name: str):
    """Decorator to measure operation execution time and memory usage."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Memory before operation
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time

            # Memory after operation
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before

            # Store single run results
            if not hasattr(self, "run_results"):
                self.run_results = {}
            if operation_name not in self.run_results:
                self.run_results[operation_name] = {"times": [], "memory": []}

            self.run_results[operation_name]["times"].append(execution_time)
            self.run_results[operation_name]["memory"].append(memory_used)

            return ProcessingResults(
                df=result[0] if isinstance(result, tuple) else result,
                execution_time=execution_time,
                memory_usage_mb=memory_used,
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
        self.run_results: Dict[str, Dict[str, List[float]]] = {}

    def calculate_statistics(self, values: List[float]) -> BenchmarkStats:
        """Calculate statistical measures from multiple runs."""
        if len(values) < 2:
            return BenchmarkStats(
                mean=values[0] if values else 0.0,
                std_dev=0.0,
                min_val=values[0] if values else 0.0,
                max_val=values[0] if values else 0.0,
                runs=values,
            )

        return BenchmarkStats(
            mean=statistics.mean(values),
            std_dev=statistics.stdev(values),
            min_val=min(values),
            max_val=max(values),
            runs=values,
        )

    def finalize_performance_metrics(self):
        """Calculate final performance metrics with statistical analysis."""
        for operation, data in self.run_results.items():
            time_stats = self.calculate_statistics(data["times"])
            memory_stats = self.calculate_statistics(data["memory"])

            self.performance_metrics[f"{operation}_time_mean"] = time_stats.mean
            self.performance_metrics[f"{operation}_time_std"] = time_stats.std_dev
            self.performance_metrics[f"{operation}_time_min"] = time_stats.min_val
            self.performance_metrics[f"{operation}_time_max"] = time_stats.max_val
            self.performance_metrics[f"{operation}_time_runs"] = time_stats.runs

            self.performance_metrics[f"{operation}_memory_mean"] = memory_stats.mean
            self.performance_metrics[f"{operation}_memory_std"] = memory_stats.std_dev
            self.performance_metrics[f"{operation}_memory_min"] = memory_stats.min_val
            self.performance_metrics[f"{operation}_memory_max"] = memory_stats.max_val
            self.performance_metrics[f"{operation}_memory_runs"] = memory_stats.runs

    def save_performance_metrics(self, output_path: str):
        try:
            self.finalize_performance_metrics()
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


class PyArrowDataProcessor(BaseDataProcessor):
    """Memory-optimized PyArrow implementation."""

    SCHEMA = pa.schema(
        [
            ("year_month", pa.string()),
            ("category1", pa.int32()),
            ("category2", pa.int32()),
            ("category3", pa.int32()),
            ("code", pa.string()),
            ("flag", pa.int64()),
            ("value1", pa.int32()),
            ("value2", pa.int32()),
        ]
    )

    @time_operation("loading")
    def load_data(self) -> pa.Table:
        if pacsv is not None:
            read_options = pacsv.ReadOptions(
                column_names=self.COLUMN_NAMES,
                autogenerate_column_names=False,
            )
            convert_options = pacsv.ConvertOptions(
                column_types={field.name: field.type for field in self.SCHEMA}
            )
            table = pacsv.read_csv(
                str(self.file_path),
                read_options=read_options,
                convert_options=convert_options,
            )
        else:
            ds = pa.dataset.dataset(
                str(self.file_path),
                format="csv",
                schema=self.SCHEMA,
            )
            table = ds.to_table()

        self.performance_metrics["memory_size_gb"] = table.nbytes / (1024**3)
        self.performance_metrics["row_count"] = table.num_rows
        return table

    @time_operation("cleaning")
    def clean_data(self, table: pa.Table) -> pa.Table:
        arrays = {}
        for col in table.column_names:
            array = table.column(col)
            if pa.types.is_integer(array.type):
                filled = pc.fill_null(array, 0)
            elif pa.types.is_floating(array.type):
                filled = pc.fill_null(array, 0.0)
            elif pa.types.is_string(array.type):
                filled = pc.fill_null(array, "")
            else:
                filled = array
            arrays[col] = filled
        return pa.Table.from_pydict(arrays)

    @time_operation("aggregation")
    def aggregate_data(self, table: pa.Table) -> pa.Table:
        group_cols = ["year_month", "category1", "category2"]
        if hasattr(pc, "hash_aggregate"):
            try:
                result = pc.hash_aggregate(
                    table,
                    group_keys=group_cols,
                    aggregates=[
                        ("value2", "mean"),
                        ("value2", "median"),
                        ("value2", "max"),
                    ],
                )
                return result
            except Exception as e:
                raise e
        else:
            print(
                "pyarrow.compute.hash_aggregate not available, falling back to Pandas aggregation"
            )
            df = table.to_pandas()
            agg_df = (
                df.groupby(group_cols)["value2"]
                .agg(["mean", "median", "max"])
                .reset_index()
            )
            agg_df = agg_df.rename(
                columns={
                    "mean": "value2_mean",
                    "median": "value2_median",
                    "max": "value2_max",
                }
            )
            return pa.Table.from_pandas(agg_df)

    @time_operation("sorting")
    def sort_data(self, table: pa.Table) -> pa.Table:
        sorted_indices = pc.sort_indices(table, sort_keys=[("value2", "descending")])
        return table.take(sorted_indices)

    @time_operation("filtering")
    def filter_data(self, table: pa.Table) -> Tuple[pa.Table, float]:
        value2_array = table.column("value2")
        mean_value2 = pc.mean(value2_array).as_py()
        condition = pc.greater(value2_array, mean_value2)
        filtered_table = table.filter(condition)
        avg_filtered = pc.mean(filtered_table.column("value2")).as_py()
        return filtered_table, avg_filtered

    @time_operation("correlation")
    def calculate_correlation(self, table: pa.Table) -> pd.DataFrame:
        import numpy as np

        numeric_cols = [
            field.name
            for field in table.schema
            if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
        ]
        data = {}
        for col in numeric_cols:
            arr = table.column(col).to_pandas().to_numpy(dtype=float)
            data[col] = arr
        if not data:
            return pd.DataFrame()
        arr_stack = np.vstack(list(data.values()))
        corr_matrix = np.corrcoef(arr_stack)
        return pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)


class SparkDataProcessor(BaseDataProcessor):
    """Memory-optimized PySpark implementation."""

    # Define the schema using PySpark types
    SCHEMA = StructType(
        [
            StructField("year_month", StringType(), True),
            StructField("category1", IntegerType(), True),
            StructField("category2", IntegerType(), True),
            StructField("category3", IntegerType(), True),
            StructField("code", StringType(), True),
            StructField("flag", LongType(), True),
            StructField("value1", IntegerType(), True),
            StructField("value2", IntegerType(), True),
        ]
    )

    def __init__(self, file_path: str):
        super().__init__(file_path)
        try:
            self.spark = (
                SparkSession.builder.appName("DataProcessing")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .config("spark.driver.host", "localhost")
                .config("spark.ui.enabled", "false")
                .master("local[*]")
                .getOrCreate()
            )
        except Exception as e:
            print(f"Failed to initialize Spark: {e}")
            raise

    @time_operation("loading")
    def load_data(self):
        df = self.spark.read.csv(str(self.file_path), schema=self.SCHEMA, header=False)

        # Calculate metrics
        row_count = df.count()
        self.performance_metrics["row_count"] = row_count

        # Estimate memory size (rough approximation)
        sample = df.limit(1000).toPandas()
        avg_row_size = sample.memory_usage(deep=True).sum() / len(sample)
        total_size = avg_row_size * row_count
        self.performance_metrics["memory_size_gb"] = total_size / (1024**3)

        return df

    @time_operation("cleaning")
    def clean_data(self, df):
        return df.fillna(0)

    @time_operation("aggregation")
    def aggregate_data(self, df):
        group_cols = ["year_month", "category1", "category2"]
        return df.groupBy(group_cols).agg(
            F.mean("value2").alias("value2_mean"),
            F.expr("percentile_approx(value2, 0.5)").alias("value2_median"),
            F.max("value2").alias("value2_max"),
        )

    @time_operation("sorting")
    def sort_data(self, df):
        return df.orderBy(F.col("value2").desc())

    @time_operation("filtering")
    def filter_data(self, df):
        # Calculate mean value
        mean_value = df.select(F.mean("value2")).collect()[0][0]

        # Filter and calculate new mean
        filtered_df = df.filter(F.col("value2") > mean_value)
        avg_filtered = filtered_df.select(F.mean("value2")).collect()[0][0]

        return filtered_df, float(avg_filtered)

    @time_operation("correlation")
    def calculate_correlation(self, df):
        # Get numeric columns
        numeric_cols = [
            field.name
            for field in df.schema.fields
            if isinstance(field.dataType, (IntegerType, LongType))
        ]

        # Calculate correlation matrix
        correlation_matrix = []
        for col1 in numeric_cols:
            row = []
            for col2 in numeric_cols:
                corr = df.stat.corr(col1, col2)
                row.append(corr)
            correlation_matrix.append(row)

        return pd.DataFrame(
            correlation_matrix, index=numeric_cols, columns=numeric_cols
        )

    def __del__(self):
        """Clean up Spark session on object destruction."""
        if hasattr(self, "spark"):
            self.spark.stop()


def run_single_iteration(processor, name: str, iteration: int):
    """Run a single iteration of all operations."""
    try:
        print(f"  Iteration {iteration + 1}...")

        df_result = processor.load_data()
        df_clean = processor.clean_data(df_result.df)
        df_agg = processor.aggregate_data(df_clean.df)
        df_sorted = processor.sort_data(df_clean.df)
        df_filtered = processor.filter_data(df_clean.df)
        correlation = processor.calculate_correlation(df_clean.df)

        # Store average filtered value (only from last iteration)
        if iteration == 0:  # Store from first iteration
            processor.performance_metrics["average_filtered_value"] = (
                df_filtered.additional_info
            )

        # Cleanup
        del df_result, df_clean, df_agg, df_sorted, df_filtered, correlation
        cleanup_memory()

        return True

    except Exception as e:
        print(f"    Error in iteration {iteration + 1}: {e}")
        return False


def process_implementation(
    processor, name: str, num_runs: int = 5, warmup_runs: int = 1
):
    """Process implementation with multiple runs for statistical analysis."""
    try:
        print(
            f"\nRunning {name} implementation with {num_runs} runs (+ {warmup_runs} warmup)..."
        )
        cleanup_memory()
        print(f"Memory cleaned before {name} implementation")

        # Warm-up runs (not counted in statistics)
        print(f"  Running {warmup_runs} warm-up runs...")
        for i in range(warmup_runs):
            print(f"    Warm-up {i + 1}...")
            # Create temporary processor for warmup
            warmup_processor = type(processor)(processor.file_path)
            run_single_iteration(warmup_processor, name, 0)
            del warmup_processor
            cleanup_memory()

        # Actual measured runs
        print(f"  Running {num_runs} measured runs...")
        successful_runs = 0
        for i in range(num_runs):
            if run_single_iteration(processor, name, i):
                successful_runs += 1
            else:
                print(f"    Iteration {i + 1} failed, continuing...")

        if successful_runs == 0:
            print(f"All runs failed for {name}")
            return None

        # Calculate total operation time statistics
        if hasattr(processor, "run_results"):
            all_operation_times = []
            for operation, data in processor.run_results.items():
                if "times" in data:
                    all_operation_times.extend(data["times"])

            if all_operation_times:
                processor.performance_metrics["total_operation_time_mean"] = (
                    sum(all_operation_times)
                    / len(all_operation_times)
                    * len(processor.run_results)
                )

        processor.save_performance_metrics(f"performance_metrics_{name}.json")

        print(
            f"{name.capitalize()} processing completed successfully! ({successful_runs}/{num_runs} runs)"
        )
        print(f"Memory cleaned after {name} implementation")

        return {"metrics": processor.performance_metrics.copy()}

    except Exception as e:
        print(f"Error in {name} implementation: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        cleanup_memory()


def run_comparison(file_path: str, skip_dask: bool = False, skip_spark: bool = False):
    """Run comparison between different implementations."""
    processors = {
        "pandas": PandasDataProcessor(file_path),
        "polars": PolarsDataProcessor(file_path),
        "pyarrow": PyArrowDataProcessor(file_path),
    }

    if not skip_dask:
        processors["dask"] = DaskDataProcessor(file_path)

    if not skip_spark:
        try:
            processors["spark"] = SparkDataProcessor(file_path)
        except Exception as e:
            print(f"Skipping Spark due to initialization error: {e}")
            skip_spark = True

    results = {}
    for name, processor in processors.items():
        result = process_implementation(processor, name)
        if result:
            results[name] = result
        cleanup_memory()

    return results


def generate_dataset(num_rows: int, output_path: str):
    """Generate dataset with specified number of rows using the existing generator."""
    print(f"Generating dataset with {num_rows:,} rows...")

    import csv
    import random
    from pathlib import Path

    # Define realistic data generation parameters
    years = list(range(1988, 2021))  # 33 years
    months = list(range(1, 13))  # 12 months
    categories1 = list(range(1, 21))  # 20 categories
    categories2 = list(range(1, 51))  # 50 categories
    categories3 = list(range(1, 11))  # 10 categories

    # Pre-generate some commonly used codes for realism
    common_codes = [f"{i:04d}" for i in range(1, 1000)]

    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        for i in range(num_rows):
            year = random.choice(years)
            month = random.choice(months)
            year_month = f"{year}{month:02d}"

            category1 = random.choice(categories1)
            category2 = random.choice(categories2)
            category3 = random.choice(categories3)

            code = random.choice(common_codes)
            flag = random.randint(0, 1)
            value1 = random.randint(1000, 999999)
            value2 = random.randint(100, 99999)

            writer.writerow(
                [
                    year_month,
                    category1,
                    category2,
                    category3,
                    code,
                    flag,
                    value1,
                    value2,
                ]
            )

            if (i + 1) % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"  Generated {i + 1:,} rows in {elapsed:.1f}s...")

    elapsed = time.time() - start_time
    file_size = Path(output_path).stat().st_size / 1024 / 1024  # MB
    print(f"Dataset generated: {output_path}")
    print(f"  Rows: {num_rows:,}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Time: {elapsed:.1f}s")


def run_comprehensive_benchmark():
    """Run comprehensive benchmark with multiple dataset sizes and runs."""
    # Dataset sizes for comprehensive benchmarking (massive sizes for true Spark advantages)
    dataset_configs = [
        {"rows": 10000000, "name": "10M", "file": "../../../data/benchmark_10m.csv"},
    ]

    # Configuration
    num_runs = 5
    warmup_runs = 1

    print("=" * 80)
    print("COMPREHENSIVE DATA PROCESSING BENCHMARK")
    print("=" * 80)
    print(f"Configurations:")
    print(f"  - Measured runs per test: {num_runs}")
    print(f"  - Warm-up runs per test: {warmup_runs}")
    print(f"  - Dataset sizes: {[cfg['name'] for cfg in dataset_configs]}")
    print(f"  - Libraries: Pandas, Polars, PyArrow, Dask, Spark")
    print("=" * 80)

    all_results = {}

    for config in dataset_configs:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING WITH {config['name']} ROWS DATASET")
        print(f"{'=' * 60}")

        # Generate dataset if it doesn't exist
        if not Path(config["file"]).exists():
            generate_dataset(config["rows"], config["file"])
        else:
            print(f"Using existing dataset: {config['file']}")

        # Run benchmarks for this dataset size
        dataset_results = {}

        # Test each library
        libraries = ["pandas", "polars", "pyarrow", "dask", "spark"]

        for lib_name in libraries:
            try:
                if lib_name == "pandas":
                    processor = PandasDataProcessor(config["file"])
                elif lib_name == "polars":
                    processor = PolarsDataProcessor(config["file"])
                elif lib_name == "pyarrow":
                    processor = PyArrowDataProcessor(config["file"])
                elif lib_name == "dask":
                    processor = DaskDataProcessor(config["file"])
                elif lib_name == "spark":
                    processor = SparkDataProcessor(config["file"])

                result = process_implementation(
                    processor, f"{lib_name}_{config['name']}", num_runs, warmup_runs
                )
                if result:
                    dataset_results[lib_name] = result

                # Extra cleanup for memory-intensive operations
                del processor
                cleanup_memory()

            except Exception as e:
                print(f"Skipping {lib_name} for {config['name']} dataset: {e}")

        all_results[config["name"]] = dataset_results

        print(f"\nCompleted benchmarking for {config['name']} dataset")
        print("-" * 60)

    # Print summary
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETED")
    print(f"{'=' * 80}")
    print("Results saved to performance_metrics_<library>_<size>.json files")
    print("Each file contains:")
    print("  - Mean, std dev, min, max for execution times")
    print("  - Mean, std dev, min, max for memory usage")
    print("  - Individual run data for statistical analysis")

    return all_results


if __name__ == "__main__":
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
