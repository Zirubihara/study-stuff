import json
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType


@dataclass
class ProcessingResults:
    """Store results of data processing operations."""

    df: DataFrame
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


class SparkDataProcessor:
    """Handle data processing operations on CSV files using PySpark."""

    # Define schema for the CSV data
    SCHEMA = StructType(
        [
            StructField("year_month", StringType(), True),
            StructField("category1", IntegerType(), True),
            StructField("category2", IntegerType(), True),
            StructField("category3", IntegerType(), True),
            StructField("code", StringType(), True),
            StructField("flag", IntegerType(), True),
            StructField("value1", IntegerType(), True),
            StructField("value2", IntegerType(), True),
        ]
    )

    def __init__(self, file_path: str):
        """Initialize processor with file path and create Spark session."""
        self.file_path = Path(file_path)
        self.performance_metrics: Dict[str, float] = {}

        # Initialize Spark session
        self.spark = (
            SparkSession.builder.appName("DataProcessor")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .getOrCreate()
        )

    def __del__(self):
        """Clean up Spark session."""
        if hasattr(self, "spark"):
            self.spark.stop()

    @time_operation("loading")
    def load_data(self) -> DataFrame:
        """Load CSV data into Spark DataFrame with proper schema."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = self.spark.read.csv(str(self.file_path), schema=self.SCHEMA, header=False)

        # Cache the DataFrame for better performance in subsequent operations
        df = df.cache()

        # Get row count for metrics
        row_count = df.count()
        self.performance_metrics["row_count"] = row_count

        return df

    @time_operation("cleaning")
    def clean_data(self, df: DataFrame) -> DataFrame:
        """Clean DataFrame by filling missing values."""
        return df.na.fill(0)

    @time_operation("aggregation")
    def aggregate_data(self, df: DataFrame) -> DataFrame:
        """Group and aggregate data."""
        return df.groupBy("year_month", "category1", "category2").agg(
            F.mean("value2").alias("value2_mean"),
            F.expr("percentile_approx(value2, 0.5)").alias("value2_median"),
            F.max("value2").alias("value2_max"),
        )

    @time_operation("sorting")
    def sort_data(self, df: DataFrame) -> DataFrame:
        """Sort DataFrame by value2 column."""
        return df.orderBy(F.col("value2").desc())

    @time_operation("filtering")
    def filter_data(self, df: DataFrame) -> Tuple[DataFrame, float]:
        """Filter rows where value2 is above mean."""
        # Calculate mean
        mean_value2 = df.select(F.mean("value2")).first()[0]

        # Filter and calculate new mean
        filtered_df = df.filter(F.col("value2") > mean_value2)
        avg_filtered = filtered_df.select(F.mean("value2")).first()[0]

        return filtered_df, float(avg_filtered)

    @time_operation("correlation")
    def calculate_correlation(self, df: DataFrame) -> DataFrame:
        """Calculate correlation matrix for numeric columns."""
        numeric_cols = [
            field.name
            for field in df.schema.fields
            if isinstance(field.dataType, IntegerType)
        ]

        # Convert to vector for correlation
        vector_col = "features"
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol=vector_col)
        vector_df = assembler.transform(df)

        # Calculate correlation matrix
        correlation_matrix = Correlation.corr(vector_df, vector_col)

        # Convert correlation matrix to DataFrame
        correlation_array = correlation_matrix.collect()[0][0].toArray()
        correlation_df = self.spark.createDataFrame(
            correlation_array.tolist(), numeric_cols
        )

        return correlation_df

    def save_performance_metrics(
        self, output_path: str = "performance_metrics_spark.json"
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

            return {
                "aggregated": df_agg.df,
                "sorted": df_sorted.df,
                "filtered": df_filtered.df,
                "correlation": correlation.df,
            }

        except Exception as e:
            print(f"Error during data processing: {e}")
            raise
        finally:
            # Cleanup cached DataFrames
            if hasattr(self, "spark"):
                self.spark.catalog.clearCache()


def main():
    """Main execution function."""
    # Dataset options
    small_dataset = "sample_data.csv"      # 50K rows
    large_dataset = "large_data.csv"       # 1M rows
    
    # Choose dataset to use
    csv_path = large_dataset  # Change to small_dataset for smaller test

    try:
        processor = SparkDataProcessor(csv_path)
        results = processor.process_data()
        processor.save_performance_metrics()
        print("Data processing completed successfully!")
        return results

    except Exception as e:
        print(f"Error in main execution: {e}")
        return None


if __name__ == "__main__":
    main()
