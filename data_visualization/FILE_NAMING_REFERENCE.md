# Standardized File Naming Reference

## Quick Reference Card - All 12 Charts

Use this as a reference when comparing visualizations across all 5 libraries.

### Data Processing Charts (6 files)

| # | File Name | Chart Type | Description |
|---|-----------|------------|-------------|
| 1 | `dp_execution_time` | Bar Chart | Total execution time across 5 libraries (10M dataset) |
| 2 | `dp_operation_breakdown` | Grouped Bar Chart | Time per operation (loading, cleaning, aggregation, etc.) |
| 3 | `dp_memory_usage` | Bar Chart | Memory consumption across libraries |
| 4 | `dp_scalability` | Line Chart | Performance scaling (5M, 10M, 50M) - log-log scale |
| 5 | `dp_performance_rankings` | Multi-Panel Bars | 6 panels ranking libraries per operation |
| 6 | `dp_summary_table` | Table | Complete performance matrix |

### ML/DL Framework Charts (6 files)

| # | File Name | Chart Type | Description |
|---|-----------|------------|-------------|
| 1 | `ml_training_time` | Bar Chart | Model training time across 5 frameworks |
| 2 | `ml_inference_speed` | Bar Chart | Prediction speed (samples/second) |
| 3 | `ml_memory_usage` | Bar Chart | Memory footprint of models |
| 4 | `ml_anomaly_rate` | Bar Chart | Anomaly detection accuracy |
| 5 | `ml_comparison_matrix` | 2x2 Grid | All 4 metrics in one view |
| 6 | `ml_summary_table` | Table | Complete framework comparison |

## File Extensions by Library

| Library | Extension | Location |
|---------|-----------|----------|
| Matplotlib | `.png` | `matplotlib/output/` |
| Plotly | `.html` | `plotly/output/` |
| Bokeh | `.html` | `bokeh/output/` |
| Holoviews | `.html` | `holoviews/output/` |
| Streamlit | Dashboard | `streamlit/dashboard.py` |

## Example File Paths

```
# Same chart across all libraries
matplotlib/output/dp_execution_time.png
plotly/output/dp_execution_time.html
bokeh/output/dp_execution_time.html
holoviews/output/dp_execution_time.html

# ML framework comparison
matplotlib/output/ml_training_time.png
plotly/output/ml_training_time.html
bokeh/output/ml_training_time.html
holoviews/output/ml_training_time.html
```

## Naming Convention Rules

### Prefix:
- `dp_` = Data Processing libraries comparison
- `ml_` = ML/DL frameworks comparison

### Common Names:
- `execution_time` = Total processing time
- `operation_breakdown` = Individual operations
- `memory_usage` = RAM consumption
- `scalability` = Performance vs dataset size
- `performance_rankings` = Operation-wise rankings
- `summary_table` = Complete data table
- `training_time` = Model training duration
- `inference_speed` = Prediction throughput
- `anomaly_rate` = Detection accuracy
- `comparison_matrix` = Multi-metric grid

## For Thesis Use

### In Written Document (LaTeX/Word):
Use Matplotlib PNG files at 300 DPI:
```
\includegraphics{matplotlib/output/dp_execution_time.png}
\includegraphics{matplotlib/output/ml_training_time.png}
```

### In Digital Appendix:
Reference HTML files for interactivity:
```
See interactive chart: plotly/output/dp_execution_time.html
```

### In Presentation:
Use Streamlit dashboard or Plotly HTML files:
```
streamlit run streamlit/dashboard.py
# OR
Open plotly/output/dp_execution_time.html in browser
```

### For Library Comparison:
Show same chart from multiple libraries:
```
Figure X: Execution Time Comparison
(a) Matplotlib static version
(b) Plotly interactive version
(c) Bokeh interactive version
```

## Verification Checklist

- [ ] All file names use lowercase with underscores
- [ ] All data processing charts start with `dp_`
- [ ] All ML framework charts start with `ml_`
- [ ] File extensions match library type (.png or .html)
- [ ] Same base name across all libraries
- [ ] Total of 12 charts per library (6 + 6)
- [ ] Total of 48 files (12 × 4 libraries)

## Quick Commands

```bash
# List all DP charts from Matplotlib
ls matplotlib/output/dp_*.png

# List all ML charts from Plotly
ls plotly/output/ml_*.html

# Count total charts
find . -path "*/output/*" -type f | wc -l   # Should be 48

# Compare file names across libraries
comm <(ls matplotlib/output/ | sed 's/.png//') \
     <(ls plotly/output/ | sed 's/.html//')
```

## Notes

- All charts use **10M dataset** as the standard for consistency
- Data processing covers: Pandas, Polars, PyArrow, Dask, Spark
- ML/DL covers: Scikit-learn, PyTorch, TensorFlow, XGBoost, JAX
- Charts show same data across all libraries (perfect for comparison)
- Naming is consistent and predictable (easy to find files)
