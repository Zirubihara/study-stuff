# Data Processing Performance Comparison Project

## Project Overview
This project benchmarks different Python data processing libraries (Pandas, Polars, PyArrow, Dask, and PySpark) for common data operations on large CSV datasets.

## Project Structure
```
study-stuff/
├── all.py                              # Unified implementation with all processors
├── pandas-usage.py                     # Pandas-specific implementation
├── polars-usage.py                     # Polars-specific implementation  
├── pyarrow-usage.py                    # PyArrow-specific implementation
├── dask-usage.py                       # Dask-specific implementation
├── pyspark-usage.py                    # PySpark-specific implementation
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── performance_metrics_*.json         # Performance results for each library
└── venv/                             # Virtual environment
```

## Data Schema
The CSV files processed have these columns:
- `year_month` (str): YYYYMM format
- `category1`, `category2`, `category3` (int): Integer categories
- `code` (str): String code with leading zeros
- `flag` (int): Integer flag
- `value1`, `value2` (int): Integer values

## Operations Benchmarked
1. **Loading**: CSV reading with proper data types
2. **Cleaning**: Null value handling (fill with 0)
3. **Aggregation**: Group by year_month, category1, category2 with mean/median/max of value2
4. **Sorting**: Sort by value2 column descending
5. **Filtering**: Filter rows where value2 > mean(value2)
6. **Correlation**: Calculate correlation matrix for numeric columns

## Performance Metrics Tracked
- Execution time per operation (seconds)
- Memory usage estimation (GB)
- Row count and data size
- Average filtered value
- Total operation time

## Commands to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run individual implementations
python pandas-usage.py
python polars-usage.py
python pyarrow-usage.py
python dask-usage.py
python pyspark-usage.py

# Run all implementations (unified)
python all.py
```

## Test Commands
No specific test framework is configured. To test:
1. Ensure a CSV file exists at the specified path in each script
2. Run the scripts and check for successful completion
3. Verify performance metrics JSON files are generated
4. Compare timing results across implementations

## Notes
- Update the `CSV_PATH` variable in each script to point to your data file
- Dask and Spark implementations may require additional configuration for large datasets
- Performance varies significantly based on data size and system resources
- Each implementation optimizes for different use cases (memory vs speed vs scalability)