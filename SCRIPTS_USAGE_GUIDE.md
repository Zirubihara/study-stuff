# Data Manipulation Usage Guide

This guide explains when to use each script in the `data_manipulation/` folder for the data processing performance comparison project.

## Analysis Summary
**All Scripts Renamed**: All 24 scripts now have meaningful, descriptive names
**Temporary File Removed**: `temp_pandas_1m.py` has been deleted

---

## 📊 Core Implementation Scripts

### Individual Library Implementations
- **`benchmark_pandas_implementation.py`** - Pure Pandas implementation with all operations
- **`benchmark_polars_implementation.py`** - Pure Polars implementation optimized for speed
- **`benchmark_pyarrow_implementation.py`** - PyArrow implementation for columnar operations
- **`benchmark_dask_implementation.py`** - Dask implementation for distributed computing
- **`benchmark_pyspark_implementation.py`** - Main PySpark implementation
- **`benchmark_pyspark_compatible.py`** - Compatible PySpark version with fallbacks

### Unified Implementation
- **`benchmark_all_libraries.py`** - **🌟 RECOMMENDED**: Runs all libraries in one script with comprehensive benchmarking

---

## 📈 Data Generation Scripts

### Basic Data Generation
- **`generate_test_data_50k.py`** - Creates 50K row test dataset (~1.5MB)
- **`generate_standard_datasets.py`** - Creates standard benchmarking datasets

### Size-Specific Data Generation
- **`generate_datasets_1m_5m_10m.py`** - Creates 1M, 5M, 10M row datasets
- **`generate_dataset_50million.py`** - Creates 50M row dataset (~1.5GB)
- **`generate_datasets_100m_plus.py`** - Creates 100M+ row ultra-massive datasets

### Specialized Data
- **`generate_optimized_datasets.py`** - Creates technology-optimized datasets for strength testing

---

## 🚀 Benchmark Runners

### Main Runners
- **`benchmark_runner_specialized.py`** - **🌟 RECOMMENDED**: Comprehensive technology-specific benchmarking
- **`benchmark_strength_comparison.py`** - Simple strength-focused benchmarks
- **`benchmark_runner_basic.py`** - Basic benchmark runner (legacy)

### Dataset Size-Specific Runners
- **`benchmark_1million_only.py`** - Benchmarks with 1M rows only
- **`benchmark_1m_10m_comparison.py`** - Benchmarks with 1M and 10M rows
- **`benchmark_5million_dataset.py`** - Benchmarks with 5M rows
- **`benchmark_50million_dataset.py`** - Benchmarks with 50M rows  
- **`benchmark_100million_dataset.py`** - Benchmarks with 100M rows (best for Spark)

---

## 📊 Visualization & Analysis Scripts

- **`create_presentation_charts.py`** - Generates simple comparison charts for presentations
- **`create_detailed_visualizations.py`** - Creates detailed analysis visualizations
- **`monitor_benchmark_progress.py`** - Monitors benchmark progress and status

---

## 💡 Usage Recommendations

### For Quick Testing
```bash
python generate_test_data_50k.py       # Generate test data
python benchmark_all_libraries.py      # Run all implementations
```

### For Comprehensive Analysis  
```bash
python generate_datasets_1m_5m_10m.py    # Generate all dataset sizes
python benchmark_runner_specialized.py   # Run technology-specific tests
python create_detailed_visualizations.py # Create detailed charts
```

### For Specific Library Testing
```bash
python generate_standard_datasets.py           # Generate standard datasets
python benchmark_pandas_implementation.py      # Test Pandas only
python benchmark_polars_implementation.py      # Test Polars only
```

### For Big Data Testing (Spark Advantages)
```bash
python generate_datasets_100m_plus.py    # Generate 100M+ row datasets  
python benchmark_100million_dataset.py   # Run with massive datasets
```

### For Technology Strengths Analysis
```bash
python generate_optimized_datasets.py    # Generate optimized datasets
python benchmark_strength_comparison.py  # Run strength-focused tests
```

---

## 🎯 Quick Start Commands

1. **Complete Setup**: `python generate_datasets_1m_5m_10m.py && python benchmark_all_libraries.py`
2. **Fast Demo**: `python generate_test_data_50k.py && python benchmark_all_libraries.py`
3. **Specialized Analysis**: `python generate_optimized_datasets.py && python benchmark_runner_specialized.py`
4. **Big Data Demo**: `python generate_datasets_100m_plus.py && python benchmark_100million_dataset.py`
5. **Create Charts**: `python create_presentation_charts.py && python create_detailed_visualizations.py`

---

## 📝 Notes

- **Run from data_manipulation/ directory**: All scripts should be executed from within the `data_manipulation/` folder
- **Dataset recommendations**: Use 100M+ rows to see Spark's true advantages
- **Memory requirements**: Massive datasets (100M rows) require 8GB+ RAM
- **Results location**: Performance metrics saved to `../results/`, charts to `../charts/`
- **Artifacts folder**: Contains generated temporary files during execution