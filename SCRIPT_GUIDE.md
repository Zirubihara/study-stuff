# Script Usage Guide

This guide explains when to use each script in the `scripts/` folder for the data processing performance comparison project.

## Analysis Summary
**Necessary Scripts**: 25 out of 26 scripts are needed
**Redundant/Temporary**: 1 file (`temp_pandas_1m.py`) can be removed

---

## 📊 Core Implementation Scripts

### Individual Library Implementations
- **`pandas-usage.py`** - Pure Pandas implementation with all operations
- **`polars-usage.py`** - Pure Polars implementation optimized for speed
- **`pyarrow-usage.py`** - PyArrow implementation for columnar operations
- **`dask-usage.py`** - Dask implementation for distributed computing
- **`pyspark-usage.py`** - Main PySpark implementation
- **`pyspark-usage-compatible.py`** - Compatible PySpark version with fallbacks

### Unified Implementation
- **`all.py`** - **🌟 RECOMMENDED**: Runs all libraries in one script with comprehensive benchmarking

---

## 📈 Data Generation Scripts

### Basic Data Generation
- **`generate_sample_data.py`** - Creates 50K row test dataset (~1.5MB)
- **`generate_large_data.py`** - Creates standard benchmarking datasets

### Size-Specific Data Generation
- **`generate_big_datasets.py`** - Creates 1M, 5M, 10M row datasets
- **`generate_50m_dataset.py`** - Creates 50M row dataset (~1.5GB)
- **`generate_massive_data.py`** - Creates 100M+ row ultra-massive datasets

### Specialized Data
- **`generate_specialized_datasets.py`** - Creates technology-optimized datasets for strength testing

---

## 🚀 Benchmark Runners

### Main Runners
- **`run_specialized_benchmarks.py`** - **🌟 RECOMMENDED**: Comprehensive technology-specific benchmarking
- **`simple_specialized_benchmarks.py`** - Simple strength-focused benchmarks
- **`run_benchmarks.py`** - Basic benchmark runner (legacy)

### Dataset Size-Specific Runners
- **`run_1m_only.py`** - Benchmarks with 1M rows only
- **`run_1m_10m_benchmark.py`** - Benchmarks with 1M and 10M rows
- **`run_5m_benchmark.py`** - Benchmarks with 5M rows
- **`run_50m_benchmark.py`** - Benchmarks with 50M rows  
- **`run_100m_benchmark.py`** - Benchmarks with 100M rows (best for Spark)

---

## 📊 Visualization & Analysis Scripts

- **`create_simple_charts.py`** - Generates simple comparison charts for presentations
- **`visualize_results.py`** - Creates detailed analysis visualizations
- **`check_progress.py`** - Monitors benchmark progress and status

---

## 🗑️ Files to Remove

- **`temp_pandas_1m.py`** - Temporary file, can be safely deleted

---

## 💡 Usage Recommendations

### For Quick Testing
```bash
python generate_sample_data.py  # Generate test data
python all.py                   # Run all implementations
```

### For Comprehensive Analysis  
```bash
python generate_big_datasets.py          # Generate all dataset sizes
python run_specialized_benchmarks.py     # Run technology-specific tests
python visualize_results.py              # Create detailed charts
```

### For Specific Library Testing
```bash
python generate_large_data.py     # Generate standard datasets
python pandas-usage.py            # Test Pandas only
python polars-usage.py            # Test Polars only
```

### For Big Data Testing (Spark Advantages)
```bash
python generate_massive_data.py   # Generate 100M+ row datasets  
python run_100m_benchmark.py      # Run with massive datasets
```

### For Technology Strengths Analysis
```bash
python generate_specialized_datasets.py   # Generate optimized datasets
python simple_specialized_benchmarks.py   # Run strength-focused tests
```

---

## 🎯 Quick Start Commands

1. **Complete Setup**: `python generate_big_datasets.py && python all.py`
2. **Fast Demo**: `python generate_sample_data.py && python all.py`
3. **Specialized Analysis**: `python generate_specialized_datasets.py && python run_specialized_benchmarks.py`
4. **Big Data Demo**: `python generate_massive_data.py && python run_100m_benchmark.py`
5. **Create Charts**: `python create_simple_charts.py && python visualize_results.py`

---

## 📝 Notes

- **Run from scripts/ directory**: All scripts should be executed from within the `scripts/` folder
- **Dataset recommendations**: Use 100M+ rows to see Spark's true advantages
- **Memory requirements**: Massive datasets (100M rows) require 8GB+ RAM
- **Results location**: Performance metrics saved to `../results/`, charts to `../charts/`
- **Artifacts folder**: Contains generated temporary files during execution