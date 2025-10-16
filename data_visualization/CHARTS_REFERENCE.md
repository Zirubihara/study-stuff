# Visualization Charts Reference Guide

Complete documentation organized by chart type, showing which visualization frameworks support each chart.

---

## Table of Contents

- [Chart Catalog](#chart-catalog)
  - [Data Processing Charts](#data-processing-charts)
  - [ML/DL Framework Charts](#mldl-framework-charts)
  - [Operation-Specific Charts](#operation-specific-charts)
- [File Structure by Framework](#file-structure-by-framework)
- [Color Coding Reference](#color-coding-reference)
- [Quick Reference](#quick-reference)

---

## Chart Catalog

### Data Processing Charts

---

#### Chart: Total Execution Time Comparison

**Filename Pattern**: `dp_execution_time_10M.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare total execution time across all data processing libraries |
| **Chart Type** | Bar chart (vertical or horizontal) |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Metric** | Total operation time (seconds) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/dp_execution_time_10M.html`)<br>• Holoviews (`holoviews/output/dp_execution_time_10M.html`)<br>• Matplotlib (`matplotlib/output/dp_execution_time.png`)<br>• Plotly (`plotly/output/dp_execution_time.html`) |
| **Interaction (HTML)** | Hover tooltips, zoom, pan, download |
| **Best For** | Quick overall performance comparison |

---

#### Chart: Operation Breakdown

**Filename Pattern**: `dp_operation_breakdown_10M.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare performance across individual operations |
| **Chart Type** | Grouped bar chart |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Operations** | • Loading<br>• Cleaning<br>• Aggregation<br>• Sorting<br>• Filtering<br>• Correlation |
| **Metric** | Time per operation (seconds) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/dp_operation_breakdown_10M.html`)<br>• Holoviews (`holoviews/output/dp_operation_breakdown_10M.html`)<br>• Matplotlib (`matplotlib/output/dp_operation_breakdown.png`)<br>• Plotly (`plotly/output/dp_operation_breakdown.html`) |
| **Interaction (HTML)** | Toggle operations, hover tooltips, zoom |
| **Best For** | Identifying operation-specific strengths |

---

#### Chart: Scalability Analysis

**Filename Pattern**: `dp_scalability_analysis.{html|png}` or `dp_scalability.png`

| Field | Value |
|-------|-------|
| **Purpose** | Analyze how libraries scale with increasing dataset size |
| **Chart Type** | Multi-line chart (log-log scale) |
| **Dataset Sizes** | 5M, 10M, 50M rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Metric** | Execution time vs dataset size |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/dp_scalability_analysis.html`)<br>• Holoviews (`holoviews/output/dp_scalability_analysis.html`)<br>• Matplotlib (`matplotlib/output/dp_scalability.png`)<br>• Plotly (`plotly/output/dp_scalability_analysis.html`) |
| **Interaction (HTML)** | Legend toggle, zoom on trends |
| **Best For** | Predicting performance at scale |

---

#### Chart: Memory Usage Comparison

**Filename Pattern**: `dp_memory_usage.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare memory consumption during operations |
| **Chart Type** | Bar chart |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Metric** | Peak memory usage (GB) |
| **Visualization Frameworks** | • Matplotlib (`matplotlib/output/dp_memory_usage.png`)<br>• Plotly (`plotly/output/dp_memory_usage.html`) |
| **Interaction (HTML)** | Zoom, pan, box select |
| **Best For** | Resource planning |

---

#### Chart: Performance Rankings

**Filename Pattern**: `dp_performance_rankings.png`

| Field | Value |
|-------|-------|
| **Purpose** | Overall library ranking by composite score |
| **Chart Type** | Horizontal bar chart |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Ranking Method** | Composite score across all operations |
| **Visualization Frameworks** | • Matplotlib (`matplotlib/output/dp_performance_rankings.png`) |
| **Best For** | Executive summary |

---

#### Chart: Summary Table

**Filename Pattern**: `dp_summary_table.png`

| Field | Value |
|-------|-------|
| **Purpose** | Comprehensive numerical summary |
| **Chart Type** | Rendered table |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Statistics** | Mean, median, std dev per operation |
| **Visualization Frameworks** | • Matplotlib (`matplotlib/output/dp_summary_table.png`) |
| **Best For** | Thesis appendix, detailed reference |

---

### ML/DL Framework Charts

---

#### Chart: Training Time Comparison

**Filename Pattern**: `ml_training_time.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare ML/DL framework training duration |
| **Chart Type** | Bar chart |
| **ML Frameworks Compared** | • Scikit-learn<br>• PyTorch<br>• TensorFlow<br>• XGBoost<br>• JAX |
| **Model Type** | Anomaly detection (Isolation Forest/Autoencoders) |
| **Metric** | Training time (seconds) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/ml_training_time.html`)<br>• Holoviews (`holoviews/output/ml_training_time.html`)<br>• Matplotlib (`matplotlib/output/ml_training_time.png`)<br>• Plotly (`plotly/output/ml_training_time.html`) |
| **Interaction (HTML)** | Hover tooltips, click to isolate |
| **Best For** | Framework selection for training |

---

#### Chart: Inference Speed Comparison

**Filename Pattern**: `ml_inference_speed.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare prediction throughput |
| **Chart Type** | Bar chart |
| **ML Frameworks Compared** | • Scikit-learn<br>• PyTorch<br>• TensorFlow<br>• XGBoost<br>• JAX |
| **Metric** | Samples per second (higher is better) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/ml_inference_speed.html`)<br>• Holoviews (`holoviews/output/ml_inference_speed.html`)<br>• Matplotlib (`matplotlib/output/ml_inference_speed.png`)<br>• Plotly (`plotly/output/ml_inference_speed.html`) |
| **Interaction (HTML)** | Hover tooltips, click to isolate |
| **Best For** | Production deployment decisions |

---

#### Chart: Memory Usage Comparison

**Filename Pattern**: `ml_memory_usage.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare memory consumption during training |
| **Chart Type** | Bar chart |
| **ML Frameworks Compared** | • Scikit-learn<br>• PyTorch<br>• TensorFlow<br>• XGBoost<br>• JAX |
| **Metric** | Memory usage (GB) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/ml_memory_usage.html`)<br>• Holoviews (memory included in heatmap)<br>• Matplotlib (`matplotlib/output/ml_memory_usage.png`)<br>• Plotly (`plotly/output/ml_memory_usage.html`) |
| **Interaction (HTML)** | Hover tooltips, zoom, pan |
| **Best For** | Resource-constrained environments |

---

#### Chart: Anomaly Detection Rate

**Filename Pattern**: `ml_anomaly_rate.png`

| Field | Value |
|-------|-------|
| **Purpose** | Model effectiveness comparison |
| **Chart Type** | Bar chart |
| **ML Frameworks Compared** | • Scikit-learn<br>• PyTorch<br>• TensorFlow<br>• XGBoost<br>• JAX |
| **Metric** | Anomaly detection rate (%) |
| **Visualization Frameworks** | • Matplotlib (`matplotlib/output/ml_anomaly_rate.png`) |
| **Best For** | Model quality assessment |

---

#### Chart: Framework Comparison Matrix/Heatmap

**Filename Pattern**: `ml_comparison_heatmap.html` or `ml_comparison_matrix.png`

| Field | Value |
|-------|-------|
| **Purpose** | Multi-dimensional framework comparison |
| **Chart Type** | Heatmap or radar chart |
| **ML Frameworks Compared** | • Scikit-learn<br>• PyTorch<br>• TensorFlow<br>• XGBoost<br>• JAX |
| **Metrics** | • Training Time<br>• Inference Speed<br>• Memory Usage<br>• (Accuracy in some versions) |
| **Visualization Frameworks** | • Holoviews (`holoviews/output/ml_comparison_heatmap.html`) **[Heatmap]**<br>• Matplotlib (`matplotlib/output/ml_comparison_matrix.png`) **[Radar/Heatmap]** |
| **Color Scheme** | Red-Yellow-Green diverging (Holoviews) |
| **Interaction (HTML)** | Hover shows exact values |
| **Best For** | All-in-one trade-off analysis |
| **Note** | Unique heatmap visualization in Holoviews |

---

#### Chart: ML Summary Table

**Filename Pattern**: `ml_summary_table.png`

| Field | Value |
|-------|-------|
| **Purpose** | Complete numerical results |
| **Chart Type** | Rendered table |
| **ML Frameworks Compared** | • Scikit-learn<br>• PyTorch<br>• TensorFlow<br>• XGBoost<br>• JAX |
| **Metrics** | All performance metrics |
| **Visualization Frameworks** | • Matplotlib (`matplotlib/output/ml_summary_table.png`) |
| **Best For** | Comprehensive reference |

---

### Operation-Specific Charts

These charts provide deep-dive analysis into individual operations.

---

#### Chart: Loading Performance

**Filename Pattern**: `op_loading_10M.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare CSV loading performance |
| **Chart Type** | Bar chart |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Operation** | CSV read and data loading |
| **Metric** | Loading time (seconds) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/op_loading_10M.html`)<br>• Matplotlib (`matplotlib/output/op_loading_10M.png`) |
| **Best For** | ETL pipeline optimization |

---

#### Chart: Cleaning Performance

**Filename Pattern**: `op_cleaning_10M.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare data cleaning performance |
| **Chart Type** | Bar chart |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Operation** | Missing value handling, duplicate removal |
| **Metric** | Cleaning time (seconds) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/op_cleaning_10M.html`)<br>• Matplotlib (`matplotlib/output/op_cleaning_10M.png`) |
| **Best For** | Data preprocessing pipelines |

---

#### Chart: Aggregation Performance

**Filename Pattern**: `op_aggregation_10M.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare aggregation/groupby performance |
| **Chart Type** | Bar chart |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Operation** | GroupBy with multiple aggregations |
| **Metric** | Aggregation time (seconds) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/op_aggregation_10M.html`)<br>• Matplotlib (`matplotlib/output/op_aggregation_10M.png`) |
| **Best For** | Analytics and reporting workloads |

---

#### Chart: Sorting Performance

**Filename Pattern**: `op_sorting_10M.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare sorting performance |
| **Chart Type** | Bar chart |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Operation** | Multi-column sort |
| **Metric** | Sorting time (seconds) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/op_sorting_10M.html`)<br>• Matplotlib (`matplotlib/output/op_sorting_10M.png`) |
| **Best For** | Ordered data analysis |

---

#### Chart: Filtering Performance

**Filename Pattern**: `op_filtering_10M.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare row filtering performance |
| **Chart Type** | Bar chart |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Operation** | Conditional row filtering |
| **Metric** | Filtering time (seconds) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/op_filtering_10M.html`)<br>• Matplotlib (`matplotlib/output/op_filtering_10M.png`) |
| **Best For** | Data subsetting operations |

---

#### Chart: Correlation Performance

**Filename Pattern**: `op_correlation_10M.{html|png}`

| Field | Value |
|-------|-------|
| **Purpose** | Compare correlation matrix computation |
| **Chart Type** | Bar chart |
| **Dataset Size** | 10 Million rows |
| **Data Libraries Compared** | • Pandas<br>• Polars<br>• PyArrow<br>• Dask<br>• Spark |
| **Operation** | Pairwise correlation calculation |
| **Metric** | Computation time (seconds) |
| **Visualization Frameworks** | • Bokeh (`bokeh/output/op_correlation_10M.html`)<br>• Matplotlib (`matplotlib/output/op_correlation_10M.png`) |
| **Best For** | Statistical analysis workloads |

---

## File Structure by Framework

### Bokeh
**Source Files**: `bokeh/combined_visualization.py`, `bokeh/operation_specific_charts.py`  
**Output Directory**: `bokeh/output/`  
**Format**: Interactive HTML

**Generated Charts**:
- `dp_execution_time_10M.html`
- `dp_operation_breakdown_10M.html`
- `dp_scalability_analysis.html`
- `ml_training_time.html`
- `ml_inference_speed.html`
- `ml_memory_usage.html`
- `op_loading_10M.html`
- `op_cleaning_10M.html`
- `op_aggregation_10M.html`
- `op_sorting_10M.html`
- `op_filtering_10M.html`
- `op_correlation_10M.html`

**Total**: 12 charts

---

### Holoviews
**Source File**: `holoviews/combined_visualization.py`  
**Output Directory**: `holoviews/output/`  
**Format**: Interactive HTML

**Generated Charts**:
- `dp_execution_time_10M.html`
- `dp_operation_breakdown_10M.html`
- `dp_scalability_analysis.html`
- `ml_training_time.html`
- `ml_inference_speed.html`
- `ml_comparison_heatmap.html` *(Unique - Multi-metric heatmap)*

**Total**: 6 charts (includes 1 unique visualization)

---

### Matplotlib
**Source Files**: `matplotlib/data_processing_visualization.py`, `matplotlib/ml_frameworks_visualization.py`, `matplotlib/operation_specific_charts.py`  
**Output Directory**: `matplotlib/output/`  
**Format**: Static PNG (publication-ready)

**Generated Charts**:
- `dp_execution_time.png`
- `dp_operation_breakdown.png`
- `dp_memory_usage.png`
- `dp_scalability.png`
- `dp_performance_rankings.png`
- `dp_summary_table.png`
- `ml_training_time.png`
- `ml_inference_speed.png`
- `ml_memory_usage.png`
- `ml_anomaly_rate.png`
- `ml_comparison_matrix.png`
- `ml_summary_table.png`
- `op_loading_10M.png`
- `op_cleaning_10M.png`
- `op_aggregation_10M.png`
- `op_sorting_10M.png`
- `op_filtering_10M.png`
- `op_correlation_10M.png`

**Total**: 18 charts

---

### Plotly
**Source Files**: `plotly/data_processing_visualization.py`, `plotly/ml_frameworks_visualization.py`  
**Output Directory**: `plotly/output/`  
**Format**: Interactive HTML (highly interactive)

**Generated Charts**:
- `dp_execution_time.html`
- `dp_operation_breakdown.html`
- `dp_memory_usage.html`
- `dp_scalability_analysis.html`
- `ml_training_time.html`
- `ml_inference_speed.html`
- `ml_memory_usage.html`

**Total**: 7 charts

---

### Streamlit
**Source File**: `streamlit/dashboard.py`  
**Run Command**: `streamlit run dashboard.py`  
**Format**: Interactive web application

**Features**:
- All charts from other frameworks combined
- Dataset size selector (5M, 10M, 50M)
- Library/framework filters
- Download functionality
- Live interaction

**Data Libraries**: Pandas, Polars, PyArrow, Dask, Spark  
**ML Frameworks**: Scikit-learn, PyTorch, TensorFlow, XGBoost, JAX

---

## Color Coding Reference

### Data Processing Libraries
| Library | Color | Hex Code |
|---------|-------|----------|
| Pandas | Blue | #1f77b4 |
| Polars | Orange | #ff7f0e |
| PyArrow | Green | #2ca02c |
| Dask | Red | #d62728 |
| Spark | Purple | #9467bd |

**Consistent across**: All visualization frameworks

---

### ML/DL Frameworks
| Framework | Color | Hex Code |
|-----------|-------|----------|
| Scikit-learn | Pink | #e377c2 |
| PyTorch | Gray | #7f7f7f |
| TensorFlow | Yellow-green | #bcbd22 |
| XGBoost | Cyan | #17becf |
| JAX | Light pink | #ff9896 |

**Consistent across**: All visualization frameworks

---

## Quick Reference

### Chart Availability Matrix

| Chart Type | Bokeh | Holoviews | Matplotlib | Plotly |
|------------|:-----:|:---------:|:----------:|:------:|
| **Data Processing** |
| Execution Time | ✓ | ✓ | ✓ | ✓ |
| Operation Breakdown | ✓ | ✓ | ✓ | ✓ |
| Scalability Analysis | ✓ | ✓ | ✓ | ✓ |
| Memory Usage | — | — | ✓ | ✓ |
| Performance Rankings | — | — | ✓ | — |
| Summary Table | — | — | ✓ | — |
| **ML/DL Frameworks** |
| Training Time | ✓ | ✓ | ✓ | ✓ |
| Inference Speed | ✓ | ✓ | ✓ | ✓ |
| Memory Usage | ✓ | — | ✓ | ✓ |
| Anomaly Rate | — | — | ✓ | — |
| Comparison Heatmap | — | ✓ | ✓ | — |
| Summary Table | — | — | ✓ | — |
| **Operations** |
| Loading | ✓ | — | ✓ | — |
| Cleaning | ✓ | — | ✓ | — |
| Aggregation | ✓ | — | ✓ | — |
| Sorting | ✓ | — | ✓ | — |
| Filtering | ✓ | — | ✓ | — |
| Correlation | ✓ | — | ✓ | — |

**Legend**: ✓ = Available, — = Not available

---

### File Format Guide

| Need | Format | Framework |
|------|--------|-----------|
| Thesis/Paper | PNG | Matplotlib |
| Presentation | HTML | Plotly or Bokeh |
| Live Demo | Web App | Streamlit |
| Quick View | HTML or PNG | Any |
| Print/PDF | PNG | Matplotlib |
| Web Embed | HTML | Plotly or Bokeh |

---

### Generation Commands

**Generate All**:
```bash
cd data_visualization
python generate_all_visualizations.py
```

**Individual Frameworks**:
```bash
# Bokeh (12 charts)
python bokeh/combined_visualization.py
python bokeh/operation_specific_charts.py

# Holoviews (6 charts)
python holoviews/combined_visualization.py

# Matplotlib (18 charts)
python matplotlib/data_processing_visualization.py
python matplotlib/ml_frameworks_visualization.py
python matplotlib/operation_specific_charts.py

# Plotly (7 charts)
python plotly/data_processing_visualization.py
python plotly/ml_frameworks_visualization.py

# Streamlit (dashboard)
streamlit run streamlit/dashboard.py
```

---

### Summary Statistics

**Total Unique Chart Types**: 20+

**Total Files Generated**: 43 files
- Bokeh: 12 HTML files
- Holoviews: 6 HTML files
- Matplotlib: 18 PNG files
- Plotly: 7 HTML files

**Data Libraries Benchmarked**: 5
- Pandas, Polars, PyArrow, Dask, Spark

**ML Frameworks Benchmarked**: 5
- Scikit-learn, PyTorch, TensorFlow, XGBoost, JAX

**Operations Analyzed**: 6
- Loading, Cleaning, Aggregation, Sorting, Filtering, Correlation

---

## How to Use This Reference

### Finding a Chart

1. **By Chart Type**: See [Chart Catalog](#chart-catalog)
2. **By Framework**: See [File Structure by Framework](#file-structure-by-framework)
3. **By Availability**: See [Chart Availability Matrix](#chart-availability-matrix)
4. **Search**: Use Ctrl+F to find specific charts or libraries

### Choosing the Right Chart

**For thesis/publications**:
- Use Matplotlib PNG files (highest quality, static)
- Include summary tables for exact numbers

**For presentations**:
- Use Plotly HTML (best interactivity)
- Or use Streamlit for live demos

**For web embedding**:
- Use Plotly or Bokeh HTML files
- Holoviews for unique heatmap visualization

**For quick analysis**:
- Any `dp_execution_time` or `ml_training_time` chart
- Use interactive versions to explore data

---

## File Naming Convention

```
{prefix}_{chart_name}_{dataset_size}.{ext}

Prefixes:
- dp_    = Data Processing
- ml_    = Machine Learning
- op_    = Individual Operation

Extensions:
- .html  = Interactive (Bokeh, Holoviews, Plotly)
- .png   = Static (Matplotlib)

Examples:
- dp_execution_time_10M.html
- ml_training_time.png
- op_loading_10M.html
```
