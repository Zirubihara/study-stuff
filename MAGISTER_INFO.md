# Magister - Benchmark Information

## Overview
This document describes the benchmarking system used for comparing data processing libraries.

## Project Structure

```
study-stuff/
├── scripts/benchmarks/
│   ├── dataset_specific/          # ACTIVE: Benchmark scripts for each dataset size
│   │   ├── benchmark_1m_simple.py
│   │   ├── benchmark_5m_simple.py
│   │   ├── benchmark_10m_simple.py
│   │   ├── benchmark_50m_simple.py
│   │   └── benchmark_100m_simple.py
│   ├── unified/                   # ACTIVE: Unified implementation for all libraries
│   │   └── benchmark_all_libraries.py
│   ├── results/                   # Output directory for benchmark results
│   ├── implementations_ARCHIVED/  # Old: Separate implementation files
│   └── runners_ARCHIVED/          # Old: Previous runner scripts
├── data/                          # Input datasets (CSV files)
│   ├── benchmark_1m.csv
│   ├── benchmark_5m.csv
│   ├── benchmark_10m.csv
│   ├── benchmark_50m.csv
│   └── benchmark_100m.csv
├── results/                       # Benchmark output (JSON files)
├── data_visualization/            # Visualization scripts
└── thesis_analysis/               # Analysis documents
```

## Benchmark Scripts

### Active Benchmark Scripts
All benchmark scripts are located in: `scripts/benchmarks/dataset_specific/`

**Active scripts for different dataset sizes:**
- `benchmark_1m_simple.py` - 1M rows dataset
- `benchmark_5m_simple.py` - 5M rows dataset
- `benchmark_10m_simple.py` - 10M rows dataset
- `benchmark_50m_simple.py` - 50M rows dataset
- `benchmark_100m_simple.py` - 100M rows dataset

**All scripts:**
- Use the **same implementation** (unified approach)
- Import from `benchmark_all_libraries.py`
- Generate results in `results/` directory
- Have identical structure, differ only in dataset size

### Implementation Source
All library-specific implementations are imported from:
```
scripts/benchmarks/unified/benchmark_all_libraries.py
```

This file contains:
- `PandasDataProcessor` - pandas implementation
- `PolarsDataProcessor` - Polars implementation  
- `PyArrowDataProcessor` - PyArrow implementation
- `DaskDataProcessor` - Dask implementation
- `SparkDataProcessor` - PySpark implementation
- Helper functions: `cleanup_memory()`, `process_implementation()`

### Archived Files (Not in Use)
The following folders contain old/alternative implementations and are marked with `_ARCHIVED`:
- `scripts/benchmarks/implementations_ARCHIVED/` - Old separate implementation files
- `scripts/benchmarks/runners_ARCHIVED/` - Old runner scripts
- `scripts/benchmarks/dataset_specific/artifacts_ARCHIVED/` - Old artifacts
- Files with `_ARCHIVED` suffix in `dataset_specific/` - Old versions

## Libraries Tested
1. **pandas** - Traditional DataFrame library
2. **Polars** - Fast DataFrame library with lazy evaluation
3. **PyArrow** - Apache Arrow Python bindings
4. **Dask** - Distributed computing library
5. **PySpark** - Apache Spark Python API

## Input Data

### Dataset Locations
All datasets are in CSV format and stored in the `data/` directory:

| Dataset | File | Rows |
|---------|------|------|
| 1M | `data/benchmark_1m.csv` | 1,000,000 |
| 5M | `data/benchmark_5m.csv` | 5,000,000 |
| 10M | `data/benchmark_10m.csv` | 10,000,000 |
| 50M | `data/benchmark_50m.csv` | 50,000,000 |
| 100M | `data/benchmark_100m.csv` | 100,000,000 |

### Dataset Structure
All datasets have the same structure with 8 columns:
- `year_month` - Date in YYYYMM format
- `category1` - Integer category (1-20)
- `category2` - Integer category (1-50)
- `category3` - Integer category (1-10)
- `code` - String code (0000-0999)
- `flag` - Binary flag (0 or 1)
- `value1` - Integer value (1000-999999)
- `value2` - Integer value (100-99999)

## Output Data

### Output Directory
```
results/
```

### Output Files
The benchmark generates JSON files with performance metrics in the format:
```
results/performance_metrics_<library>_<size>.json
```

**Example files:**
- `results/performance_metrics_pandas_5M.json`
- `results/performance_metrics_polars_10M.json`
- `results/performance_metrics_pyarrow_50M.json`
- etc.

**Libraries:** pandas, polars, pyarrow, dask, spark  
**Sizes:** 1M, 5M, 10M, 50M, 100M

### Metrics Collected
Each JSON file contains:
- **Execution times** for each operation (loading, filtering, aggregation, sorting, etc.)
- **Memory usage** (mean, min, max in GB)
- **Total operation time** (seconds)
- **Statistical measures** (mean, std, min, max for multiple runs)

### Console Output
The script also prints:
- Configuration information
- Progress for each library
- Success/failure status
- Summary table with execution time and memory usage

## Benchmark Operations

The benchmark tests the following operations:
1. **Data Loading** - Reading CSV file
2. **Filtering** - Filtering rows by conditions
3. **Aggregation** - Group by and aggregate operations
4. **Sorting** - Sorting data by columns
5. **Cleaning** - Data cleaning operations
6. **Correlation** - Computing correlations between columns

## Execution Configuration

### Default Settings (all benchmark scripts)
- **Runs:** 1
- **Warmup runs:** 0
- **Java Home:** `C:\Program Files\Java\jdk-17` (for PySpark)

## Running the Benchmark

### Command Format:
```bash
cd scripts/benchmarks/dataset_specific
python benchmark_<size>_simple.py
```

### Examples:
```bash
# Run 1M dataset benchmark
python benchmark_1m_simple.py

# Run 5M dataset benchmark
python benchmark_5m_simple.py

# Run 10M dataset benchmark
python benchmark_10m_simple.py

# Run 50M dataset benchmark
python benchmark_50m_simple.py

# Run 100M dataset benchmark
python benchmark_100m_simple.py
```

### Requirements
- **Java 17** (for PySpark)
- All required Python packages (see `requirements.txt`)
- Sufficient memory for the dataset size:
  - 1M/5M: ~4GB RAM
  - 10M/50M: ~8GB RAM
  - 100M: ~16GB RAM

## Related Files

### Visualization Scripts
- `data_visualization/comparative_visualization_thesis.py` - Creates comparison charts
- `data_visualization/generate_all_visualizations.py` - Generates all visualizations

### Results Analysis
- `thesis_analysis/` - Contains analysis documents for different dataset sizes
- `data_visualization/THESIS_COMPARISON_CHARTS/` - Generated visualization charts

### Data Files
All datasets are stored in `data/` directory:
- `data/benchmark_1m.csv` - 1M rows
- `data/benchmark_5m.csv` - 5M rows
- `data/benchmark_10m.csv` - 10M rows
- `data/benchmark_50m.csv` - 50M rows
- `data/benchmark_100m.csv` - 100M rows

### Results Files
All benchmark results are stored in `results/` directory:
- Format: `performance_metrics_<library>_<size>.json`
- Example: `performance_metrics_pandas_5M.json`

## Notes

- The benchmark uses the **unified implementation** approach where all library processors are in one file
- Results are stored as **JSON files** for easy parsing and visualization
- The script handles **memory cleanup** between library runs
- **PySpark requires Java 17** to be installed and configured




