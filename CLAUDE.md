# Data Processing Performance Comparison Project

## Project Overview
This project benchmarks different Python data processing libraries (Pandas, Polars, PyArrow, Dask, and PySpark) for common data operations on large CSV datasets.

## Project Structure
```
study-stuff/
├── scripts/                           # All Python scripts
│   ├── all.py                         # Unified implementation with all processors
│   ├── pandas-usage.py                # Pandas-specific implementation
│   ├── polars-usage.py                # Polars-specific implementation  
│   ├── pyarrow-usage.py               # PyArrow-specific implementation
│   ├── dask-usage.py                  # Dask-specific implementation
│   ├── pyspark-usage.py               # PySpark-specific implementation
│   ├── generate_sample_data.py        # Generate test data files
│   ├── generate_large_data.py         # Generate large datasets
│   ├── create_simple_charts.py        # Generate simple charts
│   └── visualize_results.py           # Generate detailed visualizations
├── data/                              # CSV data files
│   ├── sample_data.csv                # 50K rows test data
│   ├── large_data.csv                 # 1M rows test data
│   ├── benchmark_100k.csv             # 100K rows benchmark data
│   └── benchmark_500k.csv             # 500K rows benchmark data
├── results/                           # Performance metrics JSON files
│   └── performance_metrics_*.json     # Performance results for each library
├── charts/                            # Generated charts and visualizations
│   ├── *.png                          # Chart files
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
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

# Generate data files (from scripts directory)
cd scripts
python generate_sample_data.py    # Creates 50K row sample
python generate_large_data.py     # Creates 1M row dataset

# Run individual implementations (from scripts directory)
python pandas-usage.py
python polars-usage.py
python pyarrow-usage.py
python dask-usage.py
python pyspark-usage.py

# Run all implementations (unified)
python all.py

# Generate visualizations
python create_simple_charts.py    # Simple charts for thesis
python visualize_results.py       # Detailed analysis charts
```

## Test Commands
No specific test framework is configured. To test:
1. Generate data files first using the data generation scripts
2. Run the scripts from the scripts/ directory and check for successful completion
3. Verify performance metrics JSON files are generated in results/ directory
4. Check that visualization charts are created in charts/ directory
5. Compare timing results across implementations

## Notes
- All file paths have been updated to use the organized directory structure
- Scripts should be run from the scripts/ directory
- Data files are automatically saved to ../data/ directory
- Results are saved to ../results/ directory
- Charts are saved to ../charts/ directory
- Dask and Spark implementations may require additional configuration for large datasets
- Performance varies significantly based on data size and system resources
- Each implementation optimizes for different use cases (memory vs speed vs scalability)