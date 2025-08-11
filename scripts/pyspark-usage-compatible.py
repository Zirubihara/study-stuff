import json
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os


@dataclass
class ProcessingResults:
    """Store results of data processing operations."""

    df: Any
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
    """Handle data processing operations on CSV files using PySpark with Java compatibility handling."""

    def __init__(self, file_path: str):
        """Initialize processor with file path and handle Java compatibility."""
        self.file_path = Path(file_path)
        self.performance_metrics: Dict[str, float] = {}
        self.spark = None
        
        # Try to initialize Spark with Java 8 compatible settings
        try:
            # Set Java 8 compatible Spark configuration
            os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.sql.adaptive.enabled=false --conf spark.serializer=org.apache.spark.serializer.KryoSerializer pyspark-shell'
            
            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F
            from pyspark.sql.types import IntegerType, StringType, StructField, StructType
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.stat import Correlation
            
            self.spark = SparkSession.builder \
                .appName("DataProcessingBenchmark") \
                .config("spark.sql.adaptive.enabled", "false") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .getOrCreate()
                
            # Import functions after Spark is initialized
            self.F = F
            self.VectorAssembler = VectorAssembler
            self.Correlation = Correlation
            
            # Define schema
            self.SCHEMA = StructType([
                StructField("year_month", StringType(), True),
                StructField("category1", IntegerType(), True), 
                StructField("category2", IntegerType(), True),
                StructField("category3", IntegerType(), True),
                StructField("code", StringType(), True),
                StructField("flag", IntegerType(), True),
                StructField("value1", IntegerType(), True),
                StructField("value2", IntegerType(), True),
            ])
            
            print("SUCCESS: Spark initialized successfully with Java 8 compatibility mode")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize Spark: {str(e)}")
            print("This is likely due to Java version incompatibility.")
            print("Current system has Java 8, but PySpark 4.x requires Java 11+")
            self.spark = None
            raise

    @time_operation("loading")
    def load_data(self):
        """Load CSV data into Spark DataFrame with proper schema."""
        if not self.spark:
            raise RuntimeError("Spark not initialized")
            
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = self.spark.read.csv(str(self.file_path), schema=self.SCHEMA, header=False)
        
        # Calculate metrics
        row_count = df.count()
        self.performance_metrics["row_count"] = row_count
        
        # Estimate memory usage (rough calculation)
        self.performance_metrics["memory_size_gb"] = row_count * 8 * len(self.SCHEMA.fields) / (1024**3)
        
        return df

    @time_operation("cleaning")
    def clean_data(self, df):
        """Clean DataFrame by filling null values with zeros."""
        if not self.spark:
            raise RuntimeError("Spark not initialized")
            
        return df.fillna(0)

    @time_operation("aggregation") 
    def aggregate_data(self, df):
        """Group and aggregate data using Spark SQL functions."""
        if not self.spark:
            raise RuntimeError("Spark not initialized")
            
        return df.groupBy("year_month", "category1", "category2").agg(
            self.F.mean("value2").alias("value2_mean"),
            self.F.expr("percentile_approx(value2, 0.5)").alias("value2_median"),
            self.F.max("value2").alias("value2_max")
        )

    @time_operation("sorting")
    def sort_data(self, df):
        """Sort DataFrame by value2 column in descending order."""
        if not self.spark:
            raise RuntimeError("Spark not initialized")
            
        return df.orderBy(self.F.desc("value2"))

    @time_operation("filtering")
    def filter_data(self, df) -> Tuple[Any, float]:
        """Filter rows where value2 is above mean."""
        if not self.spark:
            raise RuntimeError("Spark not initialized")
            
        mean_value2 = df.select(self.F.mean("value2")).collect()[0][0]
        filtered_df = df.filter(self.F.col("value2") > mean_value2)
        avg_filtered = filtered_df.select(self.F.mean("value2")).collect()[0][0]
        return filtered_df, avg_filtered

    @time_operation("correlation")
    def calculate_correlation(self, df):
        """Calculate correlation matrix for numeric columns."""
        if not self.spark:
            raise RuntimeError("Spark not initialized")
            
        try:
            # Select numeric columns
            numeric_cols = ["category1", "category2", "category3", "flag", "value1", "value2"]
            assembler = self.VectorAssembler(inputCols=numeric_cols, outputCol="features")
            df_vector = assembler.transform(df)
            
            # Calculate correlation matrix
            correlation_matrix = self.Correlation.corr(df_vector, "features").head()
            return df_vector.select("features")
        except Exception as e:
            print(f"Correlation calculation failed: {e}")
            return df  # Return original df if correlation fails

    def save_performance_metrics(
        self, output_path: str = "../results/performance_metrics_spark.json"
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
        if not self.spark:
            raise RuntimeError("Spark not initialized - cannot process data")
            
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
            if self.spark:
                self.spark.stop()


def main():
    """Main execution function."""
    # Dataset options
    small_dataset = "../data/benchmark_1m.csv"  # 1M rows
    medium_dataset = "../data/benchmark_5m.csv"  # 5M rows  
    large_dataset = "../data/benchmark_10m.csv"  # 10M rows
    massive_dataset = "../data/benchmark_50m.csv"  # 50M rows (~1GB)

    # Choose dataset to use (Spark performs best with massive datasets)
    csv_path = massive_dataset  # Spark excels with 50M+ rows for true distributed processing

    try:
        print("STARTING: PySpark benchmark with Java 8 compatibility mode...")
        processor = SparkDataProcessor(csv_path)
        results = processor.process_data()
        processor.save_performance_metrics()
        print("SUCCESS: PySpark data processing completed successfully!")
        return results

    except Exception as e:
        print(f"ERROR: Error in PySpark execution: {e}")
        return None


if __name__ == "__main__":
    main()