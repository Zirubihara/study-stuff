"""
Test PySpark with Python 3.10
"""
import os
import sys

# Set Python executable paths for PySpark workers
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Import PySpark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

print("=" * 60)
print("PySpark Test with Python 3.10")
print("=" * 60)
print(f"Python version: {sys.version}")

try:
    # Create Spark session
    spark = SparkSession.builder \
        .appName("PySpark_Python310_Test") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.python.worker.reuse", "true") \
        .getOrCreate()

    print(f"Spark Version: {spark.version}")
    print(f"Spark Master: {spark.sparkContext.master}")

    # Create simple test dataframe
    data = [(1, "Alice", 100), (2, "Bob", 200), (3, "Charlie", 150)]
    columns = ["id", "name", "value"]

    df = spark.createDataFrame(data, columns)

    print("\nTest DataFrame:")
    df.show()

    print("\nRow count:", df.count())

    print("\nAggregation Test:")
    result = df.agg(
        F.sum("value").alias("total"),
        F.avg("value").alias("average"),
        F.max("value").alias("max_value")
    )
    result.show()

    spark.stop()

    print("\n" + "=" * 60)
    print("SUCCESS: PySpark is working correctly!")
    print("=" * 60)

except Exception as e:
    print(f"\nERROR: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
