"""
Test PySpark with correct Java version.
"""
import os
import sys

# Set Java 17 path before importing PySpark
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'

# Set Python executable path for PySpark workers
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Now import PySpark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

print("=" * 60)
print("PySpark Java Version Test")
print("=" * 60)
print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")

try:
    # Create Spark session
    spark = SparkSession.builder \
        .appName("PySpark_Test") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    print(f"Spark Version: {spark.version}")
    print(f"Spark Master: {spark.sparkContext.master}")

    # Create simple test dataframe
    data = [(1, "Alice", 100), (2, "Bob", 200), (3, "Charlie", 150)]
    columns = ["id", "name", "value"]

    df = spark.createDataFrame(data, columns)

    print("\nTest DataFrame:")
    df.show()

    print("\nAggregation Test:")
    result = df.agg(
        F.sum("value").alias("total"),
        F.avg("value").alias("average"),
        F.max("value").alias("max_value")
    )
    result.show()

    spark.stop()

    print("\n" + "=" * 60)
    print("SUCCESS: PySpark is working correctly with Java 17!")
    print("=" * 60)

except Exception as e:
    print(f"\nERROR: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
