"""
Minimal PySpark test - just session creation.
"""
import os
import sys

# Set Java 17 and Python paths
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession

print("Creating Spark session...")
spark = SparkSession.builder \
    .appName("Minimal_Test") \
    .config("spark.driver.memory", "1g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

print(f"SUCCESS: Spark session created!")
print(f"  Spark version: {spark.version}")
print(f"  Java Home: {os.environ.get('JAVA_HOME')}")
print(f"  Python: {sys.executable}")

# Try creating a simple RDD (not DataFrame)
rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
result = rdd.sum()
print(f"SUCCESS: RDD test passed! Sum: {result}")

spark.stop()
print("SUCCESS: Test completed!")
