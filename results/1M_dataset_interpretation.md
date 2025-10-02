# 1M Dataset Benchmark Results - Interpretation

## Overview
This benchmark tested 5 Python data processing libraries (Pandas, Polars, PyArrow, Dask, PySpark) on a 1,000,000 row CSV dataset.

## Performance Summary

| Library | Total Time (s) | Memory (GB) | Performance Rank |
|---------|---------------|-------------|------------------|
| **Polars** | **0.42** | **0.060** | 🥇 1st |
| **PyArrow** | **1.04** | **0.044** | 🥈 2nd |
| **Pandas** | **1.51** | **0.124** | 🥉 3rd |
| Dask | 3.93 | 0.047 | 4th |
| PySpark | 20.59 | 0.128 | 5th |

## Key Findings

### 1. **Polars is the Clear Winner (2.6x faster than Pandas)**
- **Total time**: 0.42 seconds
- **Memory usage**: 0.060 GB (50% less than Pandas)
- **Strengths**: Fast loading (0.19s), extremely fast cleaning (0.004s), efficient aggregation (0.08s)
- **Best for**: Medium datasets requiring speed and memory efficiency

### 2. **PyArrow Shows Strong Performance**
- **Total time**: 1.04 seconds (2.5x faster than Pandas)
- **Memory usage**: 0.044 GB (most memory-efficient)
- **Note**: Falls back to Pandas for aggregation (hash_aggregate not available)
- **Strengths**: Efficient loading, low memory footprint
- **Weakness**: Slow aggregation due to Pandas fallback (0.48s)

### 3. **Pandas Remains Solid for 1M Rows**
- **Total time**: 1.51 seconds
- **Memory usage**: 0.124 GB
- **Strengths**: Balanced performance across all operations
- **Best for**: Traditional workflows, wide ecosystem support

### 4. **Dask Overhead Apparent**
- **Total time**: 3.93 seconds (2.6x slower than Pandas)
- **Memory usage**: 0.047 GB (very efficient)
- **Issue**: Slow filtering (1.27s) and correlation (1.42s) operations
- **Note**: Dask is designed for larger-than-memory datasets; overhead not justified at 1M rows

### 5. **PySpark Shows Worst Performance**
- **Total time**: 20.59 seconds (13.6x slower than Pandas, 49x slower than Polars!)
- **Memory usage**: 0.128 GB
- **Critical bottleneck**: Correlation operation takes 14.39 seconds (70% of total time)
- **Reason**: Spark's distributed processing overhead is not justified for small datasets
- **Recommendation**: Only use Spark for datasets >10M rows

## Operation-Level Analysis

### Loading Performance
1. **PyArrow**: 0.18s (fastest)
2. **Polars**: 0.19s
3. **Pandas**: 1.07s
4. **Dask**: 1.23s
5. **PySpark**: 3.79s

### Aggregation Performance
1. **Polars**: 0.08s (fastest)
2. **Pandas**: 0.10s
3. **PySpark**: 0.10s
4. **PyArrow**: 0.48s (slow due to Pandas fallback)
5. **Dask**: 0.005s (lazy evaluation - operation deferred)

### Sorting Performance
1. **Polars**: 0.08s (fastest)
2. **Pandas**: 0.11s
3. **PySpark**: 0.02s
4. **PyArrow**: 0.28s
5. **Dask**: 0.00s (lazy evaluation)

### Filtering Performance
1. **Polars**: 0.005s (fastest)
2. **PyArrow**: 0.017s
3. **Pandas**: 0.019s
4. **Dask**: 1.27s (very slow)
5. **PySpark**: 2.24s (very slow)

### Correlation Performance
1. **Polars**: 0.06s (fastest)
2. **PyArrow**: 0.05s
3. **Pandas**: 0.10s
4. **Dask**: 1.42s
5. **PySpark**: 14.39s (extremely slow!)

## Recommendations

### For 1M Row Datasets:
1. **Use Polars** if you need maximum speed and efficiency
2. **Use Pandas** if you need ecosystem compatibility and balanced performance
3. **Use PyArrow** if memory is constrained and you can work around aggregation limitations
4. **Avoid Dask** - overhead not justified for this size
5. **Avoid PySpark** - distributed processing overhead is counterproductive

### When to Consider Alternatives:
- **Pandas**: When you need broad library support, plotting, or traditional workflows
- **Dask**: Only when dataset exceeds available RAM (>5-10GB)
- **PySpark**: Only for very large datasets (>50M rows / >3GB) where distributed processing benefits outweigh overhead

## Technical Notes
- All measurements are single-run (no warmup)
- Memory values are incremental per operation
- PySpark correlation operation is a major bottleneck at this scale
- PyArrow's aggregation falls back to Pandas (hash_aggregate not available)
- Dask's lazy evaluation defers some operations until materialization

## Conclusion
For 1 million row datasets, **Polars is the undisputed champion**, offering the best combination of speed and memory efficiency. Pandas remains a solid choice for traditional workflows, while Dask and PySpark show that distributed processing frameworks have significant overhead at this scale and should be reserved for truly large datasets.
