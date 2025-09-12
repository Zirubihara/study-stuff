# Data Processing Performance Comparison

This project provides comprehensive implementations and benchmarks of data processing operations using different Python libraries (Pandas, Polars, PyArrow, Dask, and PySpark). It includes both standard benchmarks on common datasets and specialized benchmarks designed to showcase each technology's unique strengths.

## Features

### Standard Benchmarks
- Data loading and size estimation
- Null value handling
- Group-by operations and aggregations
- Sorting and filtering
- Correlation matrix calculation
- Performance metrics tracking across multiple dataset sizes

### Specialized Benchmarks
- **Pandas**: Complex data wrangling with strings, datetime operations, and missing data handling
- **PyArrow**: Massive numerical analytics with vectorized operations and columnar processing
- **Polars**: Query optimization, time series analysis, and memory-efficient operations
- **Spark**: Distributed processing on massive datasets (50M+ rows)
- **Dask**: Out-of-core processing and parallel computing

## Requirements

- Python 3.8+
- Required packages listed in requirements.txt
- Minimum 8GB RAM recommended for large datasets
- Java 8+ for PySpark functionality

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/data-processing-comparison.git
cd data-processing-comparison
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
study-stuff/
├── scripts/                                    # Organized Python scripts (24 files)
│   ├── data_generation/                       # Data generation scripts
│   │   ├── generate_test_data_50k.py          # Generate basic test data (50K rows)
│   │   ├── generate_standard_datasets.py      # Generate standard datasets
│   │   ├── generate_datasets_1m_5m_10m.py     # Generate 1M, 5M, 10M datasets
│   │   ├── generate_dataset_50million.py      # Generate 50M row dataset
│   │   ├── generate_datasets_100m_plus.py     # Generate ultra-massive datasets
│   │   └── generate_optimized_datasets.py     # Generate technology-optimized datasets
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
│   │   │   ├── benchmark_1million_only.py     # Run 1M row benchmarks only
│   │   │   ├── benchmark_1m_10m_comparison.py # Run 1M and 10M benchmarks
│   │   │   ├── benchmark_5million_dataset.py  # Run 5M row benchmarks
│   │   │   ├── benchmark_50million_dataset.py # Run 50M row benchmarks
│   │   │   └── benchmark_100million_dataset.py # Run 100M row benchmarks
│   │   └── unified/                          # All-libraries unified benchmarks
│   │       └── benchmark_all_libraries.py     # Unified implementation with all processors
│   ├── visualization/                         # Chart and visualization creation
│   │   ├── create_presentation_charts.py      # Generate simple charts
│   │   └── create_detailed_visualizations.py  # Generate detailed visualizations
│   └── monitoring/                           # Monitoring and utilities
│       └── monitor_benchmark_progress.py      # Monitor benchmark progress
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
├── charts/                            # Generated charts and visualizations (10 PNG files)
│   ├── *_comparison.png               # Performance comparison charts
│   ├── *_analysis.png                 # Scalability and operation analysis
│   └── *_rankings.png                 # Technology ranking visualizations
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── CLAUDE.md                          # Project context for Claude Code
└── venv/                             # Virtual environment
```

## Usage

### Data Generation

First, generate the datasets (run from the scripts/data_generation directory):

```bash
cd scripts/data_generation

# Generate basic test data
python generate_test_data_50k.py

# Generate standard benchmark datasets
python generate_standard_datasets.py      # Creates basic datasets
python generate_datasets_1m_5m_10m.py     # Creates 1M, 5M, 10M row datasets
python generate_dataset_50million.py      # Creates 50M row dataset (~1.5GB)
python generate_datasets_100m_plus.py     # Creates 100M+ row datasets

# Generate specialized datasets optimized for each technology
python generate_optimized_datasets.py
```

### Standard Benchmarks

Run individual implementations (from scripts/benchmarks/implementations directory):

```bash
cd ../benchmarks/implementations

# Individual implementations
python benchmark_pandas_implementation.py
python benchmark_polars_implementation.py
python benchmark_pyarrow_implementation.py
python benchmark_dask_implementation.py
python benchmark_pyspark_implementation.py

# All implementations together (from unified directory)
cd ../unified
python benchmark_all_libraries.py
```

### Specialized Benchmarks

Run technology-specific strength showcases:

```bash
# Run comprehensive benchmarks (from runners directory)
cd ../runners
python benchmark_runner_comprehensive.py   # Main comprehensive benchmark runner
python benchmark_library_comparison.py     # Library comparison benchmarks

# Run specific dataset size benchmarks (from dataset_specific directory)
cd ../dataset_specific
python benchmark_1million_only.py          # 1M rows only
python benchmark_1m_10m_comparison.py      # 1M and 10M rows
python benchmark_5million_dataset.py       # 5M rows
python benchmark_50million_dataset.py      # 50M rows
python benchmark_100million_dataset.py     # 100M rows
```

#### Specialized Dataset Details

Each technology is tested on data optimized for its specific strengths:

**Pandas Showcase (500K rows)** - Complex Employee Records
- **Data**: Names, emails, phones, hire dates, salaries, skills, departments
- **Tests**: String parsing, datetime operations, missing data handling, pivot tables
- **Why Pandas excels**: Rich API for mixed data types, extensive string methods

**PyArrow Showcase (5M rows)** - Pure Numerical Transactions  
- **Data**: Transaction IDs, amounts, quantities, scores, timestamps
- **Tests**: Vectorized filtering, mathematical computations, fast aggregations
- **Why PyArrow excels**: Columnar format, SIMD vectorization, numerical optimization

**Polars Showcase (2M rows)** - IoT Time Series Data
- **Data**: Device sensors, measurements, locations, quality metrics, timestamps
- **Tests**: Query optimization, rolling windows, time series operations, categorical processing
- **Why Polars excels**: Lazy evaluation, query optimizer, memory efficiency

**Spark Showcase (20M rows)** - Retail Sales Data
- **Data**: Customer demographics, transactions, geography, product categories
- **Tests**: Massive dataset processing, distributed aggregations, enterprise analytics
- **Why Spark excels**: Distributed processing, fault tolerance, massive scale

**Dask Showcase (10M rows)** - Scientific Experiment Data
- **Data**: Lab measurements, experiments, samples, control groups, batch processing
- **Tests**: Out-of-core processing, parallel computations, larger-than-memory operations
- **Why Dask excels**: Scales Pandas operations, handles memory constraints

### Visualization

Generate charts and analysis (from visualization directory):

```bash
cd ../../visualization

# Simple charts for presentations
python create_presentation_charts.py

# Detailed analysis and comparisons
python create_detailed_visualizations.py
```

### Using as Library

```python
# Import individual processors from benchmark_all_libraries.py
from scripts.benchmarks.unified.benchmark_all_libraries import PandasDataProcessor, PolarsDataProcessor, PyArrowDataProcessor

# Example usage
processor = PandasDataProcessor("path/to/your/data.csv")
results = processor.process_data()
processor.save_performance_metrics("custom_output.json")
```

## Performance Metrics

Each implementation saves performance metrics to a JSON file, including:

- Loading time
- Memory usage
- Processing times for different operations
- Row counts
- Aggregation results

## Expected Data Format

The input CSV file should have the following columns:

- year_month (YYYYMM format)
- category1 (integer)
- category2 (integer)
- category3 (integer)
- code (string with leading zeros)
- flag (integer)
- value1 (integer)
- value2 (integer)

## Performance Comparison

### Technology Strengths

Each technology excels in different scenarios:

- **Pandas**: 
  - Best for: Complex data wrangling, string operations, datetime handling, missing data
  - Optimal datasets: Employee records, mixed data types, complex transformations
  - Sweet spot: 1M-10M rows with diverse operations

- **PyArrow**: 
  - Best for: Pure numerical analytics, vectorized operations, columnar processing
  - Optimal datasets: Financial transactions, sensor data, numerical computations
  - Sweet spot: 10M+ rows of numerical data

- **Polars**: 
  - Best for: Query optimization, time series analysis, memory efficiency
  - Optimal datasets: IoT data, time series, complex analytical queries
  - Sweet spot: 5M-50M rows with complex aggregations

- **Dask**: 
  - Best for: Out-of-core processing, parallel computing, larger-than-memory datasets
  - Optimal datasets: Distributed computations, memory-constrained environments
  - Sweet spot: 50M+ rows, memory limitations

- **PySpark**: 
  - Best for: Massive distributed processing, cluster computing
  - Optimal datasets: Big data processing, enterprise-scale analytics
  - Sweet spot: 100M+ rows, distributed environments

### Dataset Sizes

The project includes multiple dataset sizes for comprehensive testing:

- **50K rows (~1.5MB)**: Quick testing and development
- **5M rows (~150MB)**: Medium dataset for balanced testing
- **10M rows (~300MB)**: Large dataset where differences emerge
- **100M rows (~3GB)**: Ultra-massive for serious big data testing

**Note**: Current available datasets are optimized for the actual system capabilities and storage constraints.

### Results Files

Standard benchmarks generate:
- `performance_metrics_[technology].json` - Standard operation timings
- `performance_metrics_[technology]_[size].json` - Size-specific results

Specialized benchmarks generate:
- `specialized_strength_comparison.json` - Technology-optimized comparisons
- `[technology]_specialized_metrics.json` - Individual specialized results

## Key Insights

### When to Use Each Technology

1. **Start with Pandas** if you have:
   - Mixed data types (strings, dates, numbers)
   - Complex data cleaning requirements
   - Need for extensive string operations
   - Datasets under 10M rows

2. **Choose PyArrow** if you have:
   - Pure numerical data
   - Need for maximum memory efficiency
   - Large columnar datasets
   - Integration with Apache ecosystem

3. **Pick Polars** if you have:
   - Complex analytical queries
   - Time series data
   - Need for query optimization
   - Memory constraints with medium-large data

4. **Use Dask** if you have:
   - Larger-than-memory datasets
   - Existing Pandas code to scale
   - Need for parallel processing
   - Limited memory resources

5. **Choose Spark** if you have:
   - Truly massive datasets (100M+ rows)
   - Distributed computing infrastructure
   - Enterprise-scale requirements
   - Need for fault tolerance

### Performance Recommendations

- **For datasets < 1M rows**: Pandas is often fastest due to lower overhead
- **For datasets 1M-10M rows**: Polars typically excels with query optimization
- **For datasets 10M-50M rows**: PyArrow shines with columnar efficiency
- **For datasets > 50M rows**: Spark and Dask become advantageous
- **For specialized operations**: Use the technology-specific specialized benchmarks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and benchmarks for new features
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Pandas, Polars, PyArrow, Dask, and PySpark development teams
- The Python data science community

## Contact

For questions and feedback, please open an issue in the GitHub repository.
