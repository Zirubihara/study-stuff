# 5M Dataset Benchmark Results - Interpretation

## Overview
This benchmark tested 5 Python data processing libraries (Pandas, Polars, PyArrow, Dask, PySpark) on a 5,000,000 row CSV dataset (~150MB).

## Performance Summary

| Library | Total Time (s) | Memory (GB) | Performance Rank |
|---------|---------------|-------------|------------------|
| **Polars** | **0.85** | **0.275** | 🥇 1st |
| **PyArrow** | **4.07** | **0.312** | 🥈 2nd |
| **Pandas** | **6.93** | **0.731** | 🥉 3rd |
| Dask | 15.38 | 0.349 | 4th |
| PySpark | 16.31 | N/A | 5th |

## Key Findings

### 1. **Polars Dominates at 5M Scale (8.2x faster than Pandas)**
- **Total time**: 0.85 seconds
- **Memory usage**: 0.275 GB (62% less than Pandas)
- **Strengths**: Lightning-fast loading (0.12s), near-instant cleaning (0.002s), efficient operations across the board
- **Best for**: Medium-to-large datasets where speed is critical

### 2. **PyArrow Shows Solid Performance**
- **Total time**: 4.07 seconds (1.7x faster than Pandas)
- **Memory usage**: 0.312 GB (57% less than Pandas)
- **Weakness**: Slow aggregation (1.37s) and sorting (1.30s) compared to Polars
- **Best for**: Memory-constrained environments

### 3. **Pandas Performance Degrading**
- **Total time**: 6.93 seconds
- **Memory usage**: 0.731 GB (highest among traditional libraries)
- **Major bottleneck**: Sorting takes 2.59s (37% of total time)
- **Loading time**: 2.95s (much slower than competitors)

### 4. **Dask Still Not Competitive**
- **Total time**: 15.38 seconds (2.2x slower than Pandas)
- **Memory usage**: 0.349 GB (efficient)
- **Issue**: High overhead across all operations (loading: 3.99s, cleaning: 3.11s)
- **Note**: 5M rows still too small to justify Dask's parallel processing overhead

### 5. **PySpark Barely Improved**
- **Total time**: 16.31 seconds (2.4x slower than Pandas)
- **Critical bottleneck**: Loading (8.85s) and correlation (5.80s)
- **Reason**: Distributed framework overhead still dominates at this scale
- **Improvement vs 1M**: Relative performance slightly better but still poor

## Operation-Level Analysis

### Loading Performance (5M rows)
1. **Polars**: 0.12s ⚡ (fastest, 24.6x faster than Pandas!)
2. **PyArrow**: 0.12s ⚡
3. **Pandas**: 2.95s
4. **Dask**: 3.99s
5. **PySpark**: 8.85s

**Key Insight**: Modern libraries (Polars/PyArrow) excel at I/O operations

### Cleaning Performance
1. **PyArrow**: 0.001s ⚡ (nearly instantaneous)
2. **Polars**: 0.002s ⚡
3. **PySpark**: 0.11s
4. **Pandas**: 0.36s
5. **Dask**: 3.11s (surprisingly slow)

**Key Insight**: Dask's lazy evaluation hurts simple operations

### Aggregation Performance
1. **PySpark**: 0.16s (finally shows strength!)
2. **Polars**: 0.21s
3. **Pandas**: 0.44s
4. **PyArrow**: 1.37s
5. **Dask**: 2.55s

**Key Insight**: Spark's aggregation engine optimized even at medium scale

### Sorting Performance
1. **PySpark**: 0.03s (lazy evaluation)
2. **Polars**: 0.23s
3. **Dask**: 2.24s
4. **PyArrow**: 1.30s
5. **Pandas**: 2.59s

**Key Insight**: Pandas sorting becomes a bottleneck at 5M rows

### Filtering Performance
1. **Polars**: 0.025s ⚡ (fastest)
2. **Pandas**: 0.092s
3. **PyArrow**: 0.098s
4. **Dask**: 0.35s
5. **PySpark**: 1.36s

### Correlation Performance
1. **Polars**: 0.26s (fastest)
2. **Pandas**: 0.50s
3. **PyArrow**: 1.18s
4. **Dask**: 3.14s
5. **PySpark**: 5.80s

**Key Insight**: Statistical operations favor in-memory libraries

## Performance Scaling from 1M to 5M

| Library | 1M Time | 5M Time | Scaling Factor | Theoretical (5x) |
|---------|---------|---------|----------------|------------------|
| Polars | 0.42s | 0.85s | 2.0x | ✅ Better than linear |
| PyArrow | 1.04s | 4.07s | 3.9x | ✅ Good scaling |
| Pandas | 1.51s | 6.93s | 4.6x | ✅ Good scaling |
| Dask | 3.93s | 15.38s | 3.9x | ✅ Better scaling |
| PySpark | 20.59s | 16.31s | 0.8x | ✅ Improving! |

**Key Finding**: PySpark improves with dataset size (overhead amortized), while Polars shows exceptional sub-linear scaling.

## Recommendations

### For 5M Row Datasets:
1. **Use Polars** - Clear winner for speed and efficiency (8x faster than Pandas)
2. **Use PyArrow** - If you need low memory usage and can accept slower aggregations
3. **Use Pandas** - If ecosystem compatibility is critical, but expect performance degradation
4. **Avoid Dask** - Still too much overhead; wait for 10M+ rows
5. **Avoid PySpark** - Improving but still 19x slower than Polars

### When Performance Matters Most:
- **Polars**: 0.85s total (best choice)
- **PyArrow**: 4.07s total (2nd choice, memory-efficient)
- **Pandas**: 6.93s total (acceptable for one-off analyses)

### Memory-Constrained Scenarios:
1. **Polars**: 0.275 GB (best performance + low memory)
2. **PyArrow**: 0.312 GB
3. **Dask**: 0.349 GB
4. **Pandas**: 0.731 GB (2.7x more than Polars!)

## Technical Observations

### Polars Strengths at 5M Scale:
- **Lazy evaluation** optimizes query plans
- **Parallel execution** on multi-core systems
- **Columnar memory layout** improves cache efficiency
- **Rust implementation** eliminates Python overhead

### Pandas Weaknesses Emerging:
- **Single-threaded** execution limits scaling
- **Row-based operations** less cache-friendly
- **Memory overhead** from flexible indexing
- **Sorting bottleneck** (2.59s at 5M rows)

### PyArrow Tradeoffs:
- **Excellent I/O** performance
- **Low memory** footprint
- **Limited compute** functionality (falls back to Pandas)
- **Not all operations** fully optimized

### Dask & PySpark:
- **Overhead dominates** at 5M rows
- **Spark improving** but still 19x slower than Polars
- **Dask lazy evaluation** causes unexpected delays
- **Both better suited** for 10M+ rows

## Conclusion

At 5 million rows, **Polars emerges as the dominant solution**, offering 8x faster performance than Pandas with 2.7x less memory usage. PyArrow provides a solid alternative for memory-constrained environments. Pandas shows acceptable performance but is falling behind modern alternatives. Dask and PySpark remain unsuitable for this scale, though PySpark shows improving relative performance as data size increases.

**Bottom Line**: For datasets in the 5M row range (~150MB), use Polars unless you have specific requirements for Pandas ecosystem compatibility.
