"""
Minimal PySpark test with proper Java configuration
"""
import os
import sys

# Set JAVA_HOME to Java 17 location
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'

# Set Python executable paths
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")
print(f"Python: {sys.version}")

from pyspark.sql import SparkSession

print("\nCreating Spark session...")
spark = SparkSession.builder \
    .appName("Simple_Test") \
    .master("local[1]") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.instances", "1") \
    .getOrCreate()

print(f"Spark version: {spark.version}")

# Simple test
data = [(1, "test")]
df = spark.createDataFrame(data, ["id", "name"])
print(f"Row count: {df.count()}")

spark.stop()
print("\nSUCCESS!")
