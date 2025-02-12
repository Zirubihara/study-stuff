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
data-processing-comparison/
├── requirements.txt
├── README.md
├── pandas_processor.py
├── polars_processor.py
├── arrow_processor.py
├── dask_processor.py
└── spark_processor.py
```

## Usage

### Pandas Implementation

```python
from pandas_processor import PandasDataProcessor

processor = PandasDataProcessor("path/to/your/data.csv")
results = processor.process_data()
processor.save_performance_metrics()
```

### Polars Implementation

```python
from polars_processor import PolarsDataProcessor

processor = PolarsDataProcessor("path/to/your/data.csv")
results = processor.process_data()
processor.save_performance_metrics()
```

### PyArrow Implementation

```python
from arrow_processor import PyArrowDataProcessor

processor = PyArrowDataProcessor("path/to/your/data.csv")
results = processor.process_data()
processor.save_performance_metrics()
```

### Dask Implementation

```python
from dask_processor import DaskDataProcessor

processor = DaskDataProcessor("path/to/your/data.csv")
results = processor.process_data()
processor.save_performance_metrics()
```

### PySpark Implementation

```python
from spark_processor import SparkDataProcessor

processor = SparkDataProcessor("path/to/your/data.csv")
results = processor.process_data()
processor.save_performance_metrics()
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

- **Pandas**: Good for smaller datasets and complex operations
- **Polars**: Excellent for large datasets and parallel processing
- **PyArrow**: Best for memory efficiency and integration with big data tools
- **Dask**: Suitable for distributed data processing and handling large datasets
- **PySpark**: Scalable and optimized for big data processing in distributed environments

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
