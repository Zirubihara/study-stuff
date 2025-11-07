# Data Processing Libraries Benchmark - Thesis Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Libraries Comparison](#libraries-comparison)
3. [Dataset Information](#dataset-information)
4. [Benchmark Operations](#benchmark-operations)
5. [Results Structure](#results-structure)
6. [Running Benchmarks](#running-benchmarks)
7. [Technical Implementation](#technical-implementation)
8. [Performance Analysis](#performance-analysis)
9. [Key Findings](#key-findings)
10. [References](#references)

---

## Project Overview

This benchmark system provides a comprehensive comparison of five major Python data processing libraries across multiple dataset sizes. The goal is to evaluate performance, memory efficiency, and scalability characteristics to guide library selection for data-intensive applications.

### Research Questions
1. **Performance**: Which library offers the best execution time for common data operations?
2. **Memory Efficiency**: How do libraries differ in memory consumption?
3. **Scalability**: How do libraries handle increasing dataset sizes (1M → 100M rows)?
4. **Use Cases**: What are the optimal scenarios for each library?

### Methodology
- **Unified Implementation**: All libraries use identical operations for fair comparison
- **Multiple Runs**: Statistical analysis with mean, standard deviation, min/max values
- **Memory Tracking**: Real-time memory usage monitoring using `psutil`
- **Warmup Runs**: Optional warmup iterations to account for JIT compilation and caching

---

## Libraries Comparison

### 1. **Pandas** (v2.x)
**Type**: Eager evaluation, in-memory DataFrame library

**Strengths:**
- Most mature and widely adopted
- Extensive documentation and community support
- Rich API with 2000+ functions
- Excellent for exploratory data analysis
- Strong integration with scientific Python ecosystem

**Weaknesses:**
- Memory intensive (copies data frequently)
- Single-threaded for most operations
- Slower for large datasets (>10M rows)
- Not optimized for memory layout

**Best Use Cases:**
- Small to medium datasets (<10M rows)
- Exploratory data analysis
- Data cleaning and transformation
- Projects requiring extensive library compatibility
- Prototyping and interactive analysis

**Technical Details:**
- Uses NumPy arrays as backend
- Copy-on-write semantics (from v2.0)
- Type system: int32, int64, float64, object
- Memory: ~5x data size for operations due to copies

### 2. **Polars** (v0.x)
**Type**: Lazy/eager evaluation, columnar DataFrame library

**Strengths:**
- Written in Rust for maximum performance
- Excellent memory efficiency (zero-copy operations)
- Query optimization for lazy evaluation
- Parallel execution by default
- Modern, consistent API
- Fast CSV parsing

**Weaknesses:**
- Smaller ecosystem and community
- Less integration with other libraries
- Still evolving API (pre-1.0)
- Limited documentation compared to pandas

**Best Use Cases:**
- Large datasets (10M+ rows)
- ETL pipelines requiring speed
- Memory-constrained environments
- Batch processing workflows
- Data engineering tasks

**Technical Details:**
- Apache Arrow memory format
- SIMD vectorization
- Multi-threaded query execution
- Lazy evaluation with query optimizer
- Memory: ~1.5-2x data size

### 3. **PyArrow** (v13.x)
**Type**: Columnar format with compute functions

**Strengths:**
- Columnar memory format (cache-friendly)
- Zero-copy reads from disk
- Language interoperability (C++, R, Java)
- Efficient serialization
- Good for data interchange

**Weaknesses:**
- Limited high-level API
- Fewer operations than pandas/polars
- Requires understanding of Arrow format
- Less intuitive for DataFrame operations

**Best Use Cases:**
- Data interchange between systems
- Parquet file operations
- IPC and serialization
- Integration with Spark/DuckDB
- Column-oriented analytics

**Technical Details:**
- Columnar memory layout
- Fixed-width types for SIMD
- Memory-mapped file support
- Zero-copy buffer slicing
- Memory: ~1x data size (most efficient)

### 4. **Dask** (v2023.x)
**Type**: Distributed, lazy evaluation, parallel computing

**Strengths:**
- Scales to out-of-core and distributed computing
- Pandas-like API (easy migration)
- Task graph optimization
- Handles datasets larger than RAM
- Good for parallel workflows

**Weaknesses:**
- Overhead for small datasets
- More complex debugging
- Longer execution times for simple operations
- Requires understanding of partitioning

**Best Use Cases:**
- Datasets larger than available RAM
- Distributed computing clusters
- Parallel batch processing
- ETL on large files
- When pandas code needs to scale

**Technical Details:**
- Task graph scheduler
- Lazy evaluation with compute()
- Partition-based parallelism
- Default partition size: 128MB
- Memory: Dynamic (loads partitions on-demand)

### 5. **PySpark** (v3.x)
**Type**: Distributed, lazy evaluation, cluster computing

**Strengths:**
- Industry standard for big data
- Mature distributed computing framework
- Excellent for very large datasets (100M+ rows)
- Fault tolerance and reliability
- SQL support
- Integration with Hadoop ecosystem

**Weaknesses:**
- High setup complexity (requires JVM)
- Significant overhead for small datasets
- Slower for datasets under 50M rows
- More complex API
- Longer development cycle

**Best Use Cases:**
- Very large datasets (100M+ rows)
- Production data pipelines
- Existing Spark infrastructure
- Multi-node clusters
- Streaming and batch processing

**Technical Details:**
- JVM-based (requires Java 8+)
- RDD and DataFrame abstractions
- Catalyst optimizer
- Tungsten execution engine
- Memory: Distributed across nodes

---

## Dataset Information

### Dataset Sizes

| Size | Rows | File Size (CSV) | Use Case |
|------|------|----------------|----------|
| 1M | 1,000,000 | ~70 MB | Testing, development |
| 5M | 5,000,000 | ~350 MB | Medium-scale analysis |
| 10M | 10,000,000 | ~700 MB | Large-scale analysis |
| 50M | 50,000,000 | ~3.5 GB | Big data scenarios |
| 100M | 100,000,000 | ~7 GB | Extreme scale testing |

### Dataset Schema

All datasets share the same structure for consistent comparison:

```python
Column Name    Data Type    Range/Format        Description
-----------    ---------    ------------        -----------
year_month     String       YYYYMM              Date in year-month format (198801-202012)
category1      Int32        1-20                Primary category
category2      Int32        1-50                Secondary category
category3      Int32        1-10                Tertiary category
code           String       0000-0999           Identifier code
flag           Int32        0-1                 Binary flag
value1         Int32        1000-999999         Numeric value 1
value2         Int32        100-99999           Numeric value 2 (primary metric)
```

### Data Generation

Datasets are generated using realistic distributions:

```python
# Generation parameters
years = range(1988, 2021)           # 33 years
months = range(1, 13)               # 12 months
categories1 = range(1, 21)          # 20 options
categories2 = range(1, 51)          # 50 options
categories3 = range(1, 11)          # 10 options
codes = ['0000' to '0999']          # 1000 unique codes
flags = [0, 1]                      # Binary
value1 = random(1000, 999999)       # Wide range
value2 = random(100, 99999)         # Primary metric
```

### Dataset Locations

```
data/
├── benchmark_1m.csv      # 1 million rows
├── benchmark_5m.csv      # 5 million rows
├── benchmark_10m.csv     # 10 million rows
├── benchmark_50m.csv     # 50 million rows
└── benchmark_100m.csv    # 100 million rows
```

---

## Benchmark Operations

Each library performs the same six operations to ensure fair comparison:

### 1. **Loading** (Data Ingestion)
**Description**: Read CSV file into memory-optimized data structure

**Implementation Details:**
- Pandas: Chunked reading with dtype specification
- Polars: Lazy scan with schema definition
- PyArrow: CSV parser with type conversion
- Dask: Partitioned reading (128MB blocks)
- Spark: Distributed CSV read with schema

**Metrics:**
- Time to load complete dataset
- Memory footprint after loading
- Row count validation

### 2. **Cleaning** (Data Quality)
**Description**: Fill null/missing values with appropriate defaults

**Implementation Details:**
- Numeric columns: Fill with 0
- String columns: Fill with empty string
- Handle NA, NaN, None consistently

**Metrics:**
- Execution time
- Memory delta

### 3. **Filtering** (Selection)
**Description**: Filter rows where value2 > mean(value2)

**Implementation Details:**
1. Calculate mean of value2 column
2. Apply boolean mask/filter
3. Calculate mean of filtered results
4. Return filtered dataset + statistics

**Metrics:**
- Filter execution time
- Percentage of rows retained
- Average filtered value

### 4. **Aggregation** (Group By)
**Description**: Group by multiple columns and compute aggregates

**Implementation Details:**
```python
Group by: year_month, category1, category2
Aggregates:
  - mean(value2)
  - median(value2)
  - max(value2)
```

**Metrics:**
- Execution time
- Number of groups
- Memory usage

### 5. **Sorting** (Ordering)
**Description**: Sort entire dataset by value2 descending

**Implementation Details:**
- Full sort (not partial/top-k)
- Descending order
- Stable sort where applicable

**Metrics:**
- Sort execution time
- Memory overhead

### 6. **Correlation** (Statistical Analysis)
**Description**: Calculate correlation matrix for numeric columns

**Implementation Details:**
- Select all integer columns
- Compute Pearson correlation coefficient
- Generate NxN correlation matrix

**Metrics:**
- Computation time
- Matrix dimensions

---

## Results Structure

### Output Files

Results are saved as JSON files in the format:
```
results/performance_metrics_<library>_<size>.json
```

**Example:**
- `results/performance_metrics_pandas_5M.json`
- `results/performance_metrics_polars_10M.json`
- `results/performance_metrics_pyarrow_50M.json`
- `results/performance_metrics_dask_5M.json`
- `results/performance_metrics_spark_100M.json`

### JSON Schema

```json
{
  "row_count": 5000000,
  "memory_size_gb": 0.267,
  
  "loading_time_mean": 2.45,
  "loading_time_std": 0.12,
  "loading_time_min": 2.35,
  "loading_time_max": 2.58,
  "loading_time_runs": [2.45, 2.35, 2.58],
  
  "loading_memory_mean": 15.2,
  "loading_memory_std": 0.8,
  "loading_memory_min": 14.5,
  "loading_memory_max": 16.1,
  "loading_memory_runs": [15.2, 14.5, 16.1],
  
  "cleaning_time_mean": 0.34,
  "filtering_time_mean": 0.52,
  "aggregation_time_mean": 1.23,
  "sorting_time_mean": 1.87,
  "correlation_time_mean": 0.45,
  
  "total_operation_time_mean": 6.86,
  "average_filtered_value": 49987.23
}
```

### Metrics Explanation

**Time Metrics** (seconds):
- `_time_mean`: Average across all runs
- `_time_std`: Standard deviation (consistency measure)
- `_time_min`: Best-case performance
- `_time_max`: Worst-case performance
- `_time_runs`: Individual run times

**Memory Metrics** (MB):
- `_memory_mean`: Average memory delta
- `_memory_std`: Memory consistency
- `_memory_min/_max`: Memory range
- `memory_size_gb`: Total dataset size in memory

**Aggregate Metrics**:
- `row_count`: Dataset size validation
- `total_operation_time_mean`: Sum of all operations
- `average_filtered_value`: Result validation

---

## Running Benchmarks

### Prerequisites

**System Requirements:**
- Python 3.8+
- Java JDK 17 (for PySpark)
- RAM: 4GB minimum, 16GB recommended for 100M dataset
- Storage: 10GB free space

**Python Dependencies:**
```bash
pip install -r requirements.txt
```

Key libraries:
- pandas>=2.0.0
- polars>=0.19.0
- pyarrow>=13.0.0
- dask[complete]>=2023.0.0
- pyspark>=3.4.0
- psutil>=5.9.0
```

**Java Setup (for PySpark):**
```bash
# Windows
set JAVA_HOME=C:\Program Files\Java\jdk-17
set PATH=%JAVA_HOME%\bin;%PATH%

# Linux/Mac
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH=$JAVA_HOME/bin:$PATH
```

### Running Benchmarks

**Navigate to benchmark directory:**
```bash
cd scripts/benchmarks/dataset_specific
```

**Run specific dataset size:**
```bash
# 1M dataset (quick test, ~2-5 minutes)
python benchmark_1m_simple.py

# 5M dataset (medium test, ~10-15 minutes)
python benchmark_5m_simple.py

# 10M dataset (large test, ~20-30 minutes)
python benchmark_10m_simple.py

# 50M dataset (big data test, ~1-2 hours)
python benchmark_50m_simple.py

# 100M dataset (extreme test, ~3-4 hours)
python benchmark_100m_simple.py
```

**Expected Output:**
```
============================================================
TESTING 5M DATASET BENCHMARK
============================================================
Libraries: pandas, polars, pyarrow, dask, spark
Dataset: 5M rows
File: ../../data/benchmark_5m.csv
Runs: 1
============================================================

--- Running PANDAS ---
  Iteration 1...
SUCCESS: pandas - 6.84s, 0.267GB

--- Running POLARS ---
  Iteration 1...
SUCCESS: polars - 2.31s, 0.156GB

[... continued for all libraries ...]

============================================================
5M DATASET RESULTS SUMMARY
============================================================
Library    Time (s)   Memory (GB)
-----------------------------------
pandas     6.84       0.267
polars     2.31       0.156
pyarrow    3.45       0.142
dask       8.92       0.189
spark      15.67      0.234

Files saved to: ../../results/performance_metrics_<library>_5M.json
```

### Configuration Options

Modify in benchmark scripts:
```python
num_runs = 1        # Number of measurement runs
warmup_runs = 0     # Number of warmup iterations
```

**Recommended settings:**
- **Testing/Development**: `num_runs=1, warmup_runs=0`
- **Thesis/Publication**: `num_runs=5, warmup_runs=1`
- **Detailed Analysis**: `num_runs=10, warmup_runs=2`

---

## Technical Implementation

### Architecture Overview

```
Benchmark System Architecture
├── benchmark_<size>_simple.py      # Entry point for specific dataset
├── benchmark_all_libraries.py      # Unified implementation
│   ├── BaseDataProcessor           # Abstract base class
│   ├── PandasDataProcessor         # Pandas implementation
│   ├── PolarsDataProcessor         # Polars implementation
│   ├── PyArrowDataProcessor        # PyArrow implementation
│   ├── DaskDataProcessor           # Dask implementation
│   └── SparkDataProcessor          # PySpark implementation
└── results/                        # JSON outputs
```

### Key Design Patterns

**1. Template Method Pattern:**
```python
class BaseDataProcessor:
    def run_benchmark(self):
        self.load_data()
        self.clean_data()
        self.aggregate_data()
        self.sort_data()
        self.filter_data()
        self.calculate_correlation()
```

**2. Decorator Pattern (Timing):**
```python
@time_operation("loading")
def load_data(self):
    # Automatically tracks time and memory
    return df
```

**3. Strategy Pattern (Library-Specific Implementations):**
Each processor implements the same interface with library-specific optimizations.

### Memory Management

**Tracking:**
```python
process = psutil.Process()
memory_before = process.memory_info().rss / 1024 / 1024  # MB
# ... operation ...
memory_after = process.memory_info().rss / 1024 / 1024
memory_used = memory_after - memory_before
```

**Cleanup:**
```python
def cleanup_memory():
    gc.collect()            # Python garbage collection
    if hasattr(os, 'sync'):
        os.sync()           # Flush OS caches (Linux/Mac)
```

### Statistical Analysis

**Multiple runs for reliability:**
```python
class BenchmarkStats:
    mean: float              # Average performance
    std_dev: float          # Consistency measure
    min_val: float          # Best case
    max_val: float          # Worst case
    runs: List[float]       # Raw data
```

**Coefficient of Variation (CV):**
```python
CV = (std_dev / mean) * 100
# CV < 5%: Excellent consistency
# CV 5-10%: Good consistency
# CV > 10%: High variability (investigate)
```

---

## Performance Analysis

### Expected Performance Characteristics

#### 1. Small Datasets (1M-5M rows)

**Winner: Polars** ⚡
- **Speed**: 2-3x faster than pandas
- **Memory**: 40-50% less than pandas
- **Reason**: Optimized single-node processing, parallelism, efficient memory layout

**Rankings:**
1. **Polars**: Fastest overall (2-3s for 5M)
2. **PyArrow**: Fast loading, moderate processing (3-4s)
3. **Pandas**: Reference performance (6-8s)
4. **Dask**: Overhead from partitioning (8-10s)
5. **Spark**: Significant JVM overhead (15-20s)

**Insight**: Overhead of distributed systems (Dask, Spark) outweighs benefits at this scale.

#### 2. Medium Datasets (10M-50M rows)

**Winner: Polars/PyArrow** ⚡
- **Speed**: Polars maintains lead
- **Memory**: PyArrow most efficient
- **Trend**: Dask begins to compete with pandas

**Rankings:**
1. **Polars**: Still fastest (5-15s for 50M)
2. **PyArrow**: Efficient but limited operations
3. **Dask**: Competitive with pandas (20-30s)
4. **Pandas**: Memory pressure increases
5. **Spark**: Still overhead-heavy (60-90s)

**Insight**: Pandas memory copying becomes bottleneck. Polars' parallelism shines.

#### 3. Large Datasets (100M rows)

**Winner: Polars (in-memory) or Dask (near-memory-limit)** ⚡
- **Speed**: Polars if RAM sufficient, Dask for out-of-core
- **Memory**: Critical factor
- **Spark**: Becomes competitive at massive scale

**Rankings:**
1. **Polars**: Fastest if memory available (30-60s)
2. **Dask**: Good for limited RAM (60-120s)
3. **Spark**: Competitive for 100M+ (120-180s)
4. **PyArrow**: Fast but limited API
5. **Pandas**: Often fails (OOM errors)

**Insight**: Library choice depends on available RAM vs. dataset size.

### Operation-Specific Performance

#### Loading (CSV Parsing)
**Fastest**: Polars > PyArrow > Pandas > Dask > Spark
- Polars has optimized multi-threaded CSV parser
- PyArrow uses efficient columnar reading
- Dask/Spark overhead from partition coordination

#### Filtering
**Fastest**: PyArrow ≈ Polars > Pandas > Dask > Spark
- Columnar formats excel at predicates
- SIMD vectorization in Polars/PyArrow
- Predicate pushdown in lazy evaluation

#### Aggregation (Group By)
**Fastest**: Polars > Dask ≈ Spark > Pandas > PyArrow
- Polars: Multi-threaded hash aggregation
- Dask/Spark: Parallel group by when data is large
- PyArrow: Limited aggregation API

#### Sorting
**Fastest**: Polars > Pandas > PyArrow > Dask > Spark
- Polars: Parallel merge sort
- Pandas: Optimized single-threaded sort
- Distributed sorts have network overhead

#### Correlation
**Fastest**: PyArrow ≈ Polars > Pandas > Dask > Spark
- Columnar format benefits matrix operations
- SIMD acceleration
- Dask/Spark overhead for statistical operations

### Memory Efficiency Ranking

**Most Efficient → Least Efficient:**
1. **PyArrow**: ~1.0-1.5x data size (columnar, no copies)
2. **Polars**: ~1.5-2.0x data size (zero-copy operations)
3. **Dask**: ~2.0-3.0x data size (partition overhead)
4. **Pandas**: ~3.0-5.0x data size (frequent copies)
5. **Spark**: ~2.0-4.0x data size (JVM overhead + serialization)

### Scalability Trends

```
Performance Ratio (Time vs. Pandas Baseline)

Dataset Size:    1M    5M    10M   50M   100M
Pandas:         1.0x  1.0x  1.0x  1.0x  1.0x (baseline)
Polars:         0.3x  0.3x  0.3x  0.4x  0.5x (2-3x faster)
PyArrow:        0.5x  0.5x  0.5x  0.6x  0.7x (1.5-2x faster)
Dask:           1.2x  1.1x  1.0x  0.9x  0.8x (improves with scale)
Spark:          2.5x  2.0x  1.8x  1.5x  1.2x (overhead decreases)
```

**Interpretation:**
- **Polars**: Consistent 2-3x speedup across all scales
- **Dask**: Performance improves with scale (overhead amortized)
- **Spark**: Overhead only justified at 50M+ rows
- **PyArrow**: Consistently efficient but limited operations

---

## Key Findings

### 1. Library Selection Matrix

| Use Case | Dataset Size | Recommended Library | Rationale |
|----------|-------------|---------------------|-----------|
| Exploratory Analysis | <10M | Pandas | Mature API, ecosystem |
| Fast ETL Pipeline | Any | Polars | Speed + memory efficiency |
| Data Interchange | Any | PyArrow | Columnar format, interop |
| Out-of-core Processing | >RAM | Dask | Handles larger-than-RAM |
| Distributed Cluster | >100M | Spark | Production big data |
| Real-time Analytics | <50M | Polars | Low latency |
| Memory-Constrained | Any | PyArrow/Polars | Minimal memory footprint |
| Production Pipeline | 50M+ | Spark/Dask | Reliability, scaling |

### 2. Performance Insights

**Polars Advantages:**
✅ Fastest for 90% of workloads
✅ Best memory efficiency (except PyArrow)
✅ Excellent API design
✅ Multi-threaded by default
✅ Good for 1M - 100M+ rows

**When NOT to use Polars:**
❌ Need pandas ecosystem integration
❌ Using specialized pandas libraries
❌ Datasets > available RAM (use Dask)
❌ Existing pandas codebase (migration cost)

**Pandas Advantages:**
✅ Largest ecosystem
✅ Most comprehensive API
✅ Universal knowledge/tutorials
✅ Better for small ad-hoc analysis

**When NOT to use Pandas:**
❌ Datasets > 10M rows (memory issues)
❌ Performance-critical applications
❌ Memory-constrained environments
❌ Need multi-threading

**Dask Advantages:**
✅ Handles larger-than-RAM datasets
✅ Pandas-like API (easy migration)
✅ Scales to clusters
✅ Good task scheduling

**When NOT to use Dask:**
❌ Datasets fit comfortably in RAM
❌ Simple, fast operations needed
❌ Complex debugging required
❌ Latency-sensitive applications

**Spark Advantages:**
✅ Industry standard for big data
✅ Mature, proven at massive scale
✅ Fault tolerance
✅ SQL support

**When NOT to use Spark:**
❌ Datasets < 50M rows
❌ Simple local processing
❌ JVM not acceptable
❌ Fast iteration needed

### 3. Memory Recommendations

| Dataset Size | Minimum RAM | Recommended RAM | Notes |
|--------------|-------------|-----------------|-------|
| 1M rows | 2 GB | 4 GB | All libraries work |
| 5M rows | 4 GB | 8 GB | Pandas may struggle |
| 10M rows | 8 GB | 16 GB | Pandas needs 16GB |
| 50M rows | 16 GB | 32 GB | Use Polars/Dask |
| 100M rows | 32 GB | 64 GB | Dask/Spark preferred |

### 4. Development Considerations

**Learning Curve:**
- Easiest: Pandas (mature docs, huge community)
- Easy: Polars (clean API, good docs)
- Moderate: Dask (need partitioning understanding)
- Steep: Spark (JVM, cluster concepts, RDD/DF)
- Specialized: PyArrow (need Arrow format knowledge)

**Migration Path:**
1. **Start**: Pandas (prototyping)
2. **Scale**: Polars (performance boost, similar API)
3. **Scale More**: Dask (keep pandas code, distribute)
4. **Scale Massive**: Spark (rewrite for distributed)

**Hybrid Approach:**
```python
# Use different libraries for different tasks
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

# PyArrow for efficient I/O
table = pq.read_table('data.parquet')

# Polars for fast processing
df = pl.from_arrow(table)
result = df.filter(...).group_by(...).agg(...)

# Pandas for final analysis/visualization
final = result.to_pandas()
final.plot(...)
```

### 5. Common Pitfalls

**Pandas:**
- ⚠️ Memory spikes during operations (3-5x dataset size)
- ⚠️ Implicit copying (e.g., `df[col] = value`)
- ⚠️ Single-threaded bottlenecks
- ⚠️ String columns (object dtype) are memory-heavy

**Polars:**
- ⚠️ Smaller ecosystem (fewer integrations)
- ⚠️ API still evolving (breaking changes possible)
- ⚠️ Some pandas patterns don't translate directly
- ⚠️ Lazy evaluation can be confusing

**Dask:**
- ⚠️ Overhead for small data
- ⚠️ Must call `.compute()` to materialize results
- ⚠️ Debugging is harder (distributed traces)
- ⚠️ Partition size tuning required

**Spark:**
- ⚠️ Complex setup (JVM, dependencies)
- ⚠️ Massive overhead for small data
- ⚠️ Longer development cycle
- ⚠️ Harder to debug than Python-native tools

**PyArrow:**
- ⚠️ Limited high-level API
- ⚠️ Requires understanding of Arrow format
- ⚠️ Not as intuitive as DataFrame libraries
- ⚠️ Fewer operations available

---

## References

### Official Documentation
- **Pandas**: https://pandas.pydata.org/docs/
- **Polars**: https://pola-rs.github.io/polars/
- **PyArrow**: https://arrow.apache.org/docs/python/
- **Dask**: https://docs.dask.org/
- **PySpark**: https://spark.apache.org/docs/latest/api/python/

### Benchmarking Methodology
- IEEE Standard for Software Benchmark Documentation
- Statistical methods for performance analysis
- Python profiling best practices

### Related Research
- Apache Arrow: A cross-language development platform for in-memory data
- Polars: Lightning-fast DataFrame library
- Dask: Flexible library for parallel computing
- Spark: Unified analytics engine for big data

### Project Repository Structure
```
study-stuff/
├── scripts/benchmarks/
│   ├── dataset_specific/    # Benchmark entry points
│   ├── unified/              # Library implementations
│   └── results/              # Benchmark outputs
├── data/                     # Test datasets
├── results/                  # JSON results
├── data_visualization/       # Charts and graphs
└── thesis_analysis/          # Analysis documents
```

---

## Appendix: Quick Reference

### Command Cheat Sheet
```bash
# Run benchmarks
cd scripts/benchmarks/dataset_specific
python benchmark_1m_simple.py     # 1M rows
python benchmark_5m_simple.py     # 5M rows
python benchmark_10m_simple.py    # 10M rows
python benchmark_50m_simple.py    # 50M rows
python benchmark_100m_simple.py   # 100M rows

# View results
cat ../../results/performance_metrics_polars_5M.json

# Generate visualizations
cd ../../data_visualization
python comparative_visualization_thesis.py
```

### Result Interpretation
```python
# Good performance: Time ratio < 1.0
time_ratio = library_time / pandas_time

# Excellent: < 0.5 (2x faster)
# Good: 0.5-0.8
# Similar: 0.8-1.2
# Slow: > 1.2

# Memory efficiency
memory_ratio = library_memory / dataset_size

# Excellent: < 1.5
# Good: 1.5-2.5
# Average: 2.5-4.0
# Poor: > 4.0
```

### Performance Expectations (5M Dataset)
```
Operation        Polars  Pandas  PyArrow  Dask   Spark
Loading          0.5s    2.5s    0.8s     3.0s   8.0s
Filtering        0.1s    0.5s    0.1s     0.8s   2.0s
Aggregation      0.3s    1.2s    0.6s     1.5s   3.5s
Sorting          0.4s    1.8s    0.7s     2.0s   4.0s
Correlation      0.2s    0.9s    0.3s     1.2s   2.5s
Total            ~2.3s   ~6.8s   ~3.5s    ~8.9s  ~16s
Memory           0.16GB  0.27GB  0.14GB   0.19GB 0.23GB
```

---

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Author**: Thesis Project - Data Processing Benchmark Study  
**License**: MIT

For questions or issues, refer to project documentation or contact the research team.








