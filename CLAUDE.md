# Data Processing Performance Comparison Project

## Project Overview
This project benchmarks different Python data processing libraries (Pandas, Polars, PyArrow, Dask, and PySpark) for common data operations on large CSV datasets. The benchmarks use **real-world data** from the custom_1988_2020.csv dataset (113.6M rows, 4.23GB), providing meaningful performance comparisons based on actual production data patterns.

## Project Structure
```
study-stuff/
├── scripts/                                    # Organized Python scripts (24 files)
│   ├── data_generation/                       # Data generation scripts
│   │   ├── extract_real_world_datasets.py     # Extract real-world dataset subsets from custom_1988_2020.csv
│   │   ├── generate_test_data_50k.py          # Generate basic test data (50K rows)
│   │   ├── generate_standard_datasets.py      # Generate standard datasets (synthetic)
│   │   ├── generate_datasets_1m_5m_10m.py     # Generate 1M, 5M, 10M datasets (synthetic)
│   │   ├── generate_dataset_50million.py      # Generate 50M row dataset (synthetic)
│   │   ├── generate_datasets_100m_plus.py     # Generate ultra-massive datasets (synthetic)
│   │   └── generate_optimized_datasets.py     # Generate technology-optimized datasets (synthetic)
│   ├── benchmarks/                           # Benchmark implementations and runners
│   │   ├── implementations/                   # Individual library implementations
│   │   │   ├── benchmark_pandas_implementation.py # Pandas-specific implementation
│   │   │   ├── benchmark_polars_implementation.py # Polars-specific implementation  
│   │   │   ├── benchmark_pyarrow_implementation.py # PyArrow-specific implementation
│   │   │   ├── benchmark_dask_implementation.py   # Dask-specific implementation
│   │   │   ├── benchmark_pyspark_implementation.py # PySpark-specific implementation
│   │   │   └── benchmark_pyspark_optimized.py     # PySpark implementation (optimized version)
│   │   ├── runners/                          # Benchmark runners and orchestration
│   │   │   ├── benchmark_runner_simple.py       # Simple benchmark runner
│   │   │   ├── benchmark_runner_comprehensive.py # Main comprehensive benchmark runner
│   │   │   └── benchmark_library_comparison.py   # Library comparison benchmarks
│   │   ├── dataset_specific/                 # Dataset size-specific benchmarks
│   │   │   ├── benchmark_1m_simple.py         # Run 1M row benchmarks (simple/fast)
│   │   │   ├── benchmark_5m_simple.py         # Run 5M row benchmarks (simple/fast)
│   │   │   ├── benchmark_10m_simple.py        # Run 10M row benchmarks (simple/fast)
│   │   │   ├── benchmark_50m_simple.py        # Run 50M row benchmarks (simple/fast)
│   │   │   ├── benchmark_100m_simple.py       # Run 100M row benchmarks (simple/fast)
│   │   │   └── archived/                      # Older benchmark variants (non-simple versions)
│   │   └── unified/                          # All-libraries unified benchmarks
│   │       └── benchmark_all_libraries.py     # Unified implementation with all processors
│   ├── visualization/                         # Chart and visualization creation
│   │   ├── create_presentation_charts.py      # Generate simple charts
│   │   └── create_detailed_visualizations.py  # Generate detailed visualizations
│   └── monitoring/                           # Monitoring and utilities
│       └── monitor_benchmark_progress.py      # Monitor benchmark progress
├── data/                              # CSV data files (~6GB total, real-world data)
│   ├── benchmark_1m.csv               # 1M rows from real data (~38MB)
│   ├── benchmark_5m.csv               # 5M rows from real data (~190MB)
│   ├── benchmark_10m.csv              # 10M rows from real data (~379MB)
│   ├── benchmark_50m.csv              # 50M rows from real data (~1.86GB)
│   ├── benchmark_100m.csv             # 100M rows from real data (~3.73GB)
│   ├── sample_data.csv                # 50K rows test data (~1.5MB, synthetic)
│   └── specialized/                   # Technology-optimized datasets (synthetic)
├── custom_1988_2020.csv               # Original real-world dataset (113.6M rows, 4.23GB)
├── results/                           # Performance metrics JSON files (51 files)
│   ├── performance_metrics_*.json     # Standard benchmark results
│   ├── *_specialized_metrics.json     # Technology-specific results
│   └── specialized_strength_comparison.json # Cross-technology comparison
├── charts/                            # Generated charts and visualizations (10 PNG files)
│   ├── *_comparison.png               # Performance comparison charts
│   ├── *_analysis.png                 # Scalability and operation analysis
│   └── *_rankings.png                 # Technology ranking visualizations
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── venv_py310/                        # Virtual environment (Python 3.10, with PySpark support)
└── venv_py311_pyspark/               # Virtual environment (Python 3.11, PySpark testing)
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

# Extract real-world data subsets (RECOMMENDED - uses actual production data)
cd scripts/data_generation
python extract_real_world_datasets.py      # Extracts 1M, 5M, 10M, 50M, 100M from custom_1988_2020.csv

# OR generate synthetic data files (alternative approach)
python generate_test_data_50k.py           # Creates 50K row sample (synthetic)
python generate_standard_datasets.py       # Creates standard datasets (synthetic)
python generate_datasets_1m_5m_10m.py      # Creates 1M, 5M, 10M row datasets (synthetic)
python generate_dataset_50million.py       # Creates 50M row dataset (synthetic, ~1.5GB)
python generate_datasets_100m_plus.py      # Creates 100M+ row datasets (synthetic)
python generate_optimized_datasets.py      # Creates technology-optimized datasets (synthetic)

# Run individual implementations (from benchmarks/implementations subdirectory)
cd ../benchmarks/implementations
python benchmark_pandas_implementation.py
python benchmark_polars_implementation.py
python benchmark_pyarrow_implementation.py
python benchmark_dask_implementation.py
python benchmark_pyspark_implementation.py

# Run all implementations (unified from benchmarks/unified subdirectory)
cd ../unified
python benchmark_all_libraries.py

# Run comprehensive benchmarks (from benchmarks/runners subdirectory)
cd ../runners
python benchmark_runner_comprehensive.py   # Main comprehensive benchmark runner
python benchmark_library_comparison.py     # Library comparison benchmarks

# Run specific dataset size benchmarks (from benchmarks/dataset_specific subdirectory)
cd ../dataset_specific
python benchmark_1m_simple.py              # 1M rows only
python benchmark_5m_simple.py              # 5M rows
python benchmark_10m_simple.py             # 10M rows
python benchmark_50m_simple.py             # 50M rows
python benchmark_100m_simple.py            # 100M rows

# Generate visualizations (from visualization subdirectory)
cd ../../visualization
python create_presentation_charts.py       # Simple charts for thesis
python create_detailed_visualizations.py   # Detailed analysis charts

# Monitor progress (from monitoring subdirectory)
cd ../monitoring
python monitor_benchmark_progress.py       # Check benchmark progress
```

## Test Commands
No specific test framework is configured. To test:
1. Generate data files first using the data generation scripts
2. Run the scripts from the scripts/ directory and check for successful completion
3. Verify performance metrics JSON files are generated in results/ directory
4. Check that visualization charts are created in charts/ directory
5. Compare timing results across implementations

## Available Dataset Sizes for Performance Testing (Real-World Data)
All datasets are extracted from the original custom_1988_2020.csv file containing real production data from 1988-2020:
- **1M rows (~38MB)**: Quick testing with real data patterns (benchmark_1m.csv)
- **5M rows (~190MB)**: Medium dataset for balanced performance comparison (benchmark_5m.csv)
- **10M rows (~379MB)**: Large dataset where Spark shows its advantages (benchmark_10m.csv)
- **50M rows (~1.86GB)**: Very large dataset for serious performance testing (benchmark_50m.csv)
- **100M rows (~3.73GB)**: Ultra-massive dataset for big data processing benchmarks (benchmark_100m.csv)
- **113.6M rows (~4.23GB)**: Complete original dataset (custom_1988_2020.csv)
- **50K rows (~1.5MB)**: Quick synthetic test data (sample_data.csv)
- **Specialized datasets**: Technology-optimized synthetic datasets in data/specialized/
- **Recommendation**: Use real-world datasets (1M-100M) for thesis to demonstrate performance on actual production data

## Notes
- **Using Real-World Data**: All benchmark datasets (1M-100M rows) are extracted from actual production data (custom_1988_2020.csv), providing realistic performance metrics for thesis research
- Scripts are now organized into logical subfolders for better project management
- Data generation scripts are in scripts/data_generation/
  - **extract_real_world_datasets.py** is the primary script for creating benchmark datasets from real data
  - Other scripts generate synthetic data for comparison or specialized testing
- Benchmark scripts are organized in scripts/benchmarks/ with subfolders:
  - implementations/ (individual library benchmarks)
  - runners/ (benchmark orchestration)
  - dataset_specific/ (size-specific benchmarks)
  - unified/ (all-libraries benchmarks)
- Visualization scripts are in scripts/visualization/
- Monitoring scripts are in scripts/monitoring/
- Data files are automatically saved to ../../data/ directory (from subfolders)
- Results are saved to ../../results/ directory (from subfolders)
- Charts are saved to ../../charts/ directory (from subfolders)
- **Spark performs best with massive datasets (100M+ rows, 3GB+)** - smaller datasets may not show Spark's distributed processing advantages
- Current project has 51 performance result files and 10 visualization charts
- Specialized datasets available in data/specialized/ directory for technology-specific benchmarking
- Dask and Spark implementations may require additional configuration for very large datasets
- Performance varies significantly based on data size and system resources
- Each implementation optimizes for different use cases (memory vs speed vs scalability)
- Use benchmark_runner_comprehensive.py for comprehensive technology-specific testing
- Multiple benchmark runners available for different dataset sizes and use cases
- **Real-world data provides more credible thesis results** than synthetic data due to actual patterns and distributions