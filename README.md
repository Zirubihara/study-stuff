# Data Processing Performance Comparison

This project provides implementations of data processing operations using different Python libraries (Pandas, Polars, PyArrow, Dask, and PySpark) with performance benchmarking. Each implementation includes loading, cleaning, aggregating, sorting, filtering, and correlation analysis capabilities.

## Features

- Data loading and size estimation
- Null value handling
- Group-by operations and aggregations
- Sorting and filtering
- Correlation matrix calculation
- Performance metrics tracking

## Requirements

- Python 3.8+
- Required packages listed in requirements.txt

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
├── requirements.txt
├── README.md
├── CLAUDE.md                          # Project context for Claude Code
├── all.py                             # Unified implementation with all processors
├── pandas-usage.py                    # Pandas-specific implementation
├── polars-usage.py                    # Polars-specific implementation
├── pyarrow-usage.py                   # PyArrow-specific implementation
├── dask-usage.py                      # Dask-specific implementation
├── pyspark-usage.py                   # PySpark-specific implementation
├── performance_metrics_*.json         # Performance results for each library
└── venv/                             # Virtual environment
```

## Usage

### Running Individual Implementations

Each implementation can be run as a standalone script:

```bash
# Pandas implementation
python pandas-usage.py

# Polars implementation  
python polars-usage.py

# PyArrow implementation
python pyarrow-usage.py

# Dask implementation
python dask-usage.py

# PySpark implementation
python pyspark-usage.py
```

### Running All Implementations

Use the unified script to run all implementations:

```bash
python all.py
```

### Using as Library

```python
# Import individual processors from all.py
from all import PandasDataProcessor, PolarsDataProcessor, PyArrowDataProcessor

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

Performance varies based on data size and operations:

- **Pandas**: Good for smaller datasets and complex operations. Uses chunked reading for memory efficiency.
- **Polars**: Excellent for large datasets and parallel processing. Fastest for most operations with lazy evaluation.
- **PyArrow**: Best for memory efficiency and integration with big data tools. Columnar storage format.
- **Dask**: Suitable for distributed data processing and handling large datasets. Out-of-core processing.
- **PySpark**: Scalable and optimized for big data processing in distributed environments.

### Key Files Generated

After running the benchmarks, you'll find these performance metric files:
- `performance_metrics_pandas.json`
- `performance_metrics_polars.json` 
- `performance_metrics_pyarrow.json`
- `performance_metrics_dask.json`
- `performance_metrics_spark.json`

Each file contains timing data for loading, cleaning, aggregation, sorting, filtering, and correlation operations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Pandas, Polars, PyArrow, Dask, and PySpark development teams

## Contact

For questions and feedback, please open an issue in the GitHub repository.
