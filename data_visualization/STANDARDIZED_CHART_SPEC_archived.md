# Standardized Chart Specification for All 5 Visualization Libraries

## Purpose
This document defines the EXACT charts that ALL 5 visualization libraries must produce with IDENTICAL file names for thesis comparison purposes.

## Chart List (12 Total Charts)

### Data Processing Charts (6 files)
All files use `10M` dataset as default for consistency.

1. **`dp_execution_time.{png|html}`**
   - **Type**: Bar chart
   - **Content**: Total execution time comparison across 5 libraries (Pandas, Polars, PyArrow, Dask, Spark)
   - **X-axis**: Library name
   - **Y-axis**: Total execution time (seconds)
   - **Title**: "Data Processing: Total Execution Time (10M Dataset)"

2. **`dp_operation_breakdown.{png|html}`**
   - **Type**: Grouped bar chart
   - **Content**: Time breakdown for 6 operations (loading, cleaning, aggregation, sorting, filtering, correlation)
   - **X-axis**: Operation name
   - **Y-axis**: Execution time (seconds)
   - **Groups**: 5 libraries
   - **Title**: "Data Processing: Operation Breakdown (10M Dataset)"

3. **`dp_memory_usage.{png|html}`**
   - **Type**: Bar chart
   - **Content**: Memory usage comparison across 5 libraries
   - **X-axis**: Library name
   - **Y-axis**: Memory usage (GB)
   - **Title**: "Data Processing: Memory Usage (10M Dataset)"

4. **`dp_scalability.{png|html}`**
   - **Type**: Line chart (log-log scale)
   - **Content**: Performance scaling across dataset sizes (5M, 10M, 50M)
   - **X-axis**: Dataset size (million rows) - log scale
   - **Y-axis**: Execution time (seconds) - log scale
   - **Lines**: 5 libraries with markers
   - **Title**: "Data Processing: Scalability Analysis"

5. **`dp_performance_rankings.{png|html}`**
   - **Type**: Multi-panel horizontal bar charts (2x3 grid)
   - **Content**: 6 panels, one for each operation showing libraries ranked by speed
   - **Each panel**: Horizontal bars sorted by time (fastest first)
   - **Title**: "Data Processing: Performance Rankings by Operation (10M)"

6. **`dp_summary_table.{png|html}`**
   - **Type**: Table/heatmap
   - **Content**: Comprehensive table with all times
   - **Rows**: 5 libraries
   - **Columns**: Total Time + 6 operation times
   - **Title**: "Data Processing: Performance Summary (10M Dataset)"

### ML/DL Framework Charts (6 files)
All files use 10M dataset for model training.

1. **`ml_training_time.{png|html}`**
   - **Type**: Bar chart
   - **Content**: Training time comparison across 5 frameworks
   - **X-axis**: Framework name (Scikit-learn, PyTorch, TensorFlow, XGBoost, JAX)
   - **Y-axis**: Training time (seconds)
   - **Title**: "ML/DL Framework: Training Time Comparison"

2. **`ml_inference_speed.{png|html}`**
   - **Type**: Bar chart
   - **Content**: Inference speed comparison across 5 frameworks
   - **X-axis**: Framework name
   - **Y-axis**: Inference speed (samples/second)
   - **Title**: "ML/DL Framework: Inference Speed Comparison"

3. **`ml_memory_usage.{png|html}`**
   - **Type**: Bar chart
   - **Content**: Memory usage comparison across 5 frameworks
   - **X-axis**: Framework name
   - **Y-axis**: Memory usage (GB)
   - **Title**: "ML/DL Framework: Memory Usage Comparison"

4. **`ml_anomaly_rate.{png|html}`**
   - **Type**: Bar chart
   - **Content**: Anomaly detection rate comparison
   - **X-axis**: Framework name
   - **Y-axis**: Anomaly rate (%)
   - **Reference line**: Expected rate (1.0%)
   - **Title**: "ML/DL Framework: Anomaly Detection Rate"

5. **`ml_comparison_matrix.{png|html}`**
   - **Type**: 2x2 grid of bar charts
   - **Content**: 4 panels showing all 4 metrics
   - **Panels**: Training Time, Inference Speed, Memory Usage, Anomaly Rate
   - **Title**: "ML/DL Framework: Comprehensive Comparison Matrix"

6. **`ml_summary_table.{png|html}`**
   - **Type**: Table
   - **Content**: Comprehensive table with all metrics
   - **Rows**: 5 frameworks
   - **Columns**: Training Time, Inference Speed, Memory Usage, Anomaly Rate
   - **Title**: "ML/DL Framework: Performance Summary"

## File Format by Library

| Library | Extension | DPI/Quality |
|---------|-----------|-------------|
| Matplotlib | `.png` | 300 DPI |
| Plotly | `.html` | Interactive |
| Bokeh | `.html` | Interactive |
| Holoviews | `.html` | Interactive |
| Streamlit | Dashboard | Real-time |

## Output Directory Structure

```
data_visualization/
├── matplotlib/output/
│   ├── dp_execution_time.png
│   ├── dp_operation_breakdown.png
│   ├── dp_memory_usage.png
│   ├── dp_scalability.png
│   ├── dp_performance_rankings.png
│   ├── dp_summary_table.png
│   ├── ml_training_time.png
│   ├── ml_inference_speed.png
│   ├── ml_memory_usage.png
│   ├── ml_anomaly_rate.png
│   ├── ml_comparison_matrix.png
│   └── ml_summary_table.png
│
├── plotly/output/
│   ├── dp_execution_time.html
│   ├── dp_operation_breakdown.html
│   ├── dp_memory_usage.html
│   ├── dp_scalability.html
│   ├── dp_performance_rankings.html
│   ├── dp_summary_table.html
│   ├── ml_training_time.html
│   ├── ml_inference_speed.html
│   ├── ml_memory_usage.html
│   ├── ml_anomaly_rate.html
│   ├── ml_comparison_matrix.html
│   └── ml_summary_table.html
│
├── bokeh/output/
│   └── (same 12 files as .html)
│
├── holoviews/output/
│   └── (same 12 files as .html)
│
└── streamlit/
    └── dashboard.py (interactive dashboard with all charts)
```

## Total Charts
- **Per Library**: 12 charts
- **Total Files**: 4 libraries × 12 charts = 48 static/interactive files + 1 dashboard
- **For Thesis**: Perfect side-by-side comparison across all 5 visualization approaches

## Data Sources
- **Data Processing**: `../../results/performance_metrics_{library}_{size}.json`
- **ML/DL Frameworks**: `../../models/results/{framework}_anomaly_detection_results.json`

## Implementation Priority
1. ✅ Define this specification
2. Update Matplotlib (6 + 6 = 12 charts)
3. Update Plotly (6 + 6 = 12 charts)
4. Update Bokeh (6 + 6 = 12 charts)
5. Update Holoviews (6 + 6 = 12 charts)
6. Update Streamlit dashboard (include all 12 visualizations)
7. Update master generation script
8. Generate all 48+ files
9. Verify consistency

## Verification Checklist
- [ ] All 5 libraries produce exactly 12 charts each
- [ ] File names match exactly across all libraries
- [ ] Chart content shows same data comparisons
- [ ] Titles are consistent
- [ ] All charts use 10M dataset as default
- [ ] Output directories are clean (no old files)
