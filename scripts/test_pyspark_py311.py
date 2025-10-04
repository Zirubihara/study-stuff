"""
Test PySpark 4.0 with Python 3.11 and Java 17.
"""
import os
import sys

# MUST set Java home BEFORE importing PySpark
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Find and set SPARK_HOME
import pyspark
os.environ['SPARK_HOME'] = os.path.dirname(os.path.dirname(pyspark.__file__))

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"JAVA_HOME: {os.environ['JAVA_HOME']}")

from pyspark.sql import SparkSession

print("\nCreating Spark session...")
spark = SparkSession.builder \
    .appName("PySpark_4_Test_Py311") \
    .config("spark.driver.memory", "1g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

print(f"SUCCESS: Spark session created!")
print(f"  Spark version: {spark.version}")

# Try creating a simple RDD
print("\nTesting RDD operations...")
rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
result = rdd.sum()
print(f"SUCCESS: RDD test passed! Sum = {result}")

# Try creating a DataFrame
print("\nTesting DataFrame operations...")
data = [(1, "Alice", 100), (2, "Bob", 200), (3, "Charlie", 150)]
columns = ["id", "name", "value"]
df = spark.createDataFrame(data, columns)
count = df.count()
print(f"SUCCESS: DataFrame created with {count} rows")

spark.stop()
print("\n" + "="*60)
print("ALL TESTS PASSED! PySpark 4.0 is working with Python 3.11!")
print("="*60)
