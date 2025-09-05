# Data Processing Performance Comparison Project

## Project Overview
This project benchmarks different Python data processing libraries (Pandas, Polars, PyArrow, Dask, and PySpark) for common data operations on large CSV datasets.

## Project Structure
```
study-stuff/
├── scripts/                           # All Python scripts (30+ files)
│   ├── all.py                         # Unified implementation with all processors
│   ├── pandas-usage.py                # Pandas-specific implementation
│   ├── polars-usage.py                # Polars-specific implementation  
│   ├── pyarrow-usage.py               # PyArrow-specific implementation
│   ├── dask-usage.py                  # Dask-specific implementation
│   ├── pyspark-usage.py               # PySpark-specific implementation
│   ├── pyspark-usage-compatible.py    # PySpark implementation (compatible version)
│   ├── run_specialized_benchmarks.py  # Main specialized benchmark runner
│   ├── simple_specialized_benchmarks.py # Simple specialized benchmarks
│   ├── generate_sample_data.py        # Generate basic test data (50K rows)
│   ├── generate_large_data.py         # Generate standard datasets
│   ├── generate_big_datasets.py       # Generate 1M, 5M, 10M datasets
│   ├── generate_50m_dataset.py        # Generate 50M row dataset
│   ├── generate_massive_data.py       # Generate ultra-massive datasets
│   ├── generate_specialized_datasets.py # Generate technology-optimized datasets
│   ├── run_benchmarks.py              # Basic benchmark runner
│   ├── run_1m_only.py                 # Run 1M row benchmarks only
│   ├── run_1m_10m_benchmark.py        # Run 1M and 10M benchmarks
│   ├── run_5m_benchmark.py            # Run 5M row benchmarks
│   ├── run_50m_benchmark.py           # Run 50M row benchmarks
│   ├── run_100m_benchmark.py          # Run 100M row benchmarks
│   ├── create_simple_charts.py        # Generate simple charts
│   ├── visualize_results.py           # Generate detailed visualizations
│   └── check_progress.py              # Monitor benchmark progress
├── data/                              # CSV data files (~3.6GB total)
│   ├── sample_data.csv                # 50K rows test data (~1.5MB)
│   ├── benchmark_5m.csv               # 5M rows (medium, ~150MB)
│   ├── benchmark_10m.csv              # 10M rows (large, ~300MB)
│   ├── benchmark_100m.csv             # 100M rows (ultra-massive, ~3GB)
│   └── specialized/                   # Technology-optimized datasets
├── results/                           # Performance metrics JSON files (51 files)
│   ├── performance_metrics_*.json     # Standard benchmark results
│   ├── *_specialized_metrics.json     # Technology-specific results
│   └── specialized_strength_comparison.json # Cross-technology comparison
├── charts/                            # Generated charts and visualizations (10+ PNG files)
│   ├── *_comparison.png               # Performance comparison charts
│   ├── *_analysis.png                 # Scalability and operation analysis
│   └── *_rankings.png                 # Technology ranking visualizations
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
python generate_large_data.py     # Creates standard datasets
python generate_big_datasets.py   # Creates 1M, 5M, 10M row datasets
python generate_50m_dataset.py    # Creates 50M row dataset (~1.5GB)
python generate_massive_data.py   # Creates 100M+ row datasets
python generate_specialized_datasets.py # Creates technology-optimized datasets

# Run individual implementations (from scripts directory)
python pandas-usage.py
python polars-usage.py
python pyarrow-usage.py
python dask-usage.py
python pyspark-usage.py

# Run all implementations (unified)
python all.py

# Run specialized benchmarks
python run_specialized_benchmarks.py  # Main specialized benchmark runner
python simple_specialized_benchmarks.py # Simple specialized benchmarks

# Run specific dataset size benchmarks
python run_1m_only.py             # 1M rows only
python run_1m_10m_benchmark.py     # 1M and 10M rows
python run_5m_benchmark.py         # 5M rows
python run_50m_benchmark.py        # 50M rows
python run_100m_benchmark.py       # 100M rows

# Generate visualizations
python create_simple_charts.py    # Simple charts for thesis
python visualize_results.py       # Detailed analysis charts

# Monitor progress
python check_progress.py          # Check benchmark progress
```

## Test Commands
No specific test framework is configured. To test:
1. Generate data files first using the data generation scripts
2. Run the scripts from the scripts/ directory and check for successful completion
3. Verify performance metrics JSON files are generated in results/ directory
4. Check that visualization charts are created in charts/ directory
5. Compare timing results across implementations

## Available Dataset Sizes for Performance Testing
- **50K rows (~1.5MB)**: Quick testing and development (sample_data.csv)
- **5M rows (~150MB)**: Medium dataset for balanced performance comparison (benchmark_5m.csv)
- **10M rows (~300MB)**: Large dataset where Spark shows its advantages (benchmark_10m.csv)
- **100M rows (~3GB)**: Ultra-massive dataset for serious big data processing (benchmark_100m.csv)
- **Specialized datasets**: Technology-optimized datasets in data/specialized/
- **Recommendation**: Use 100M row dataset for Spark to see true big data performance benefits

## Notes
- All file paths have been updated to use the organized directory structure
- Scripts should be run from the scripts/ directory
- Data files are automatically saved to ../data/ directory
- Results are saved to ../results/ directory
- Charts are saved to ../charts/ directory
- **Spark performs best with massive datasets (100M+ rows, 3GB+)** - smaller datasets may not show Spark's distributed processing advantages
- Current project has 40+ performance result files and 10+ visualization charts
- Specialized datasets available in data/specialized/ directory for technology-specific benchmarking
- Dask and Spark implementations may require additional configuration for very large datasets
- Performance varies significantly based on data size and system resources
- Each implementation optimizes for different use cases (memory vs speed vs scalability)
- Use run_specialized_benchmarks.py for comprehensive technology-specific testing
- Multiple benchmark runners available for different dataset sizes and use cases