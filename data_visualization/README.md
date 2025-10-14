# Data Visualization Suite for Thesis

Professional visualization suite comparing 5 Python libraries for thesis research on data processing and ML/DL frameworks.

## Overview

This project provides comprehensive visualizations for comparing:
1. **Data Processing Libraries**: Pandas, Polars, PyArrow, Dask, PySpark
2. **ML/DL Frameworks**: Scikit-learn, PyTorch, TensorFlow, XGBoost, JAX

Using 5 different visualization libraries:
- **Matplotlib** - Publication-quality static charts
- **Plotly** - Interactive web-based visualizations
- **Streamlit** - Full-featured dashboard application
- **Bokeh** - Interactive browser-based charts
- **Holoviews** - High-level declarative visualizations

## Clean Project Structure

```
data_visualization/
│
├── matplotlib/                          # Static charts for thesis document
│   ├── data_processing_visualization.py
│   ├── ml_frameworks_visualization.py
│   └── output/  (12 PNG files, 300 DPI)
│
├── plotly/                              # Interactive HTML visualizations
│   ├── data_processing_visualization.py
│   ├── ml_frameworks_visualization.py
│   └── output/  (10 HTML files)
│
├── bokeh/                               # Interactive charts
│   ├── combined_visualization.py
│   └── output/  (6 HTML files)
│
├── holoviews/                           # Declarative visualizations
│   ├── combined_visualization.py
│   └── output/  (6 HTML files)
│
├── streamlit/                           # Dashboard application
│   └── dashboard.py
│
├── docs/                                # Archived documentation
│
├── README.md                            # This file
├── QUICK_START.md                       # Quick reference
├── VISUALIZATION_LIBRARY_COMPARISON.md  # Detailed comparison
└── generate_all_visualizations.py       # Master script
```

## Quick Start

### Generate All Visualizations (Recommended)

```bash
cd data_visualization
python generate_all_visualizations.py
```

Creates **34 visualizations** in ~3 minutes.

### Generate Individual Libraries

```bash
# Matplotlib (for thesis document)
cd matplotlib
python data_processing_visualization.py
python ml_frameworks_visualization.py

# Plotly (interactive)
cd plotly
python data_processing_visualization.py
python ml_frameworks_visualization.py

# Bokeh
cd bokeh
python combined_visualization.py

# Holoviews
cd holoviews
python combined_visualization.py

# Streamlit Dashboard
cd streamlit
streamlit run dashboard.py
```

## Installation

Libraries already installed:
```
matplotlib==3.10.6
plotly==6.3.1
streamlit==1.50.0
bokeh==3.8.0
holoviews==1.21.0
```

## What Gets Generated

| Library | Files | Type | Best For |
|---------|-------|------|----------|
| Matplotlib | 12 PNG | Static, 300 DPI | Thesis document |
| Plotly | 10 HTML | Interactive | Online viewing |
| Bokeh | 6 HTML | Interactive | Large datasets |
| Holoviews | 6 HTML | Interactive | Quick exploration |
| Streamlit | 1 App | Interactive | Presentations |

**Total**: 34 visualizations + 1 dashboard

## Usage Recommendations

### For Written Thesis
**Use**: `matplotlib/output/` ← PNG files for LaTeX/Word

### For Thesis Defense
**Use**: `streamlit/dashboard.py` ← Interactive Q&A

### For Digital Thesis
**Use**: `plotly/output/` ← Self-contained HTML

## Key Insights from Data

### Data Processing (10M rows)
- **Fastest**: Polars (~5-8s total)
- **Most Memory Efficient**: PyArrow
- **Best for Scale**: Dask/Spark (50M+)

### ML/DL Frameworks
- **Fastest Training**: XGBoost (27s)
- **Fastest Inference**: XGBoost (1.98M samples/s)
- **Most Balanced**: Scikit-learn
- **Best for Deep Learning**: PyTorch

## Library Comparison Summary

| Feature | Matplotlib | Plotly | Streamlit | Bokeh | Holoviews |
|---------|-----------|--------|-----------|-------|-----------|
| Interactive | ❌ | ✅✅✅ | ✅✅✅ | ✅✅ | ✅✅ |
| Thesis Document | ✅✅✅ | ❌ | ❌ | ❌ | ❌ |
| Code Lines* | ~15 | ~6 | ~2 | ~12 | ~3 |
| File Size | Small | Large | N/A | Medium | Medium |

*For simple bar chart

## Documentation

- **README.md** - Main documentation (this file)
- **QUICK_START.md** - Quick commands and examples
- **VISUALIZATION_LIBRARY_COMPARISON.md** - Detailed analysis
- **docs/** - Development notes and fixes

## Scripts Overview

```
generate_all_visualizations.py  → Run all scripts (master)
matplotlib/*.py                 → 12 static charts
plotly/*.py                     → 10 interactive charts
bokeh/*.py                      → 6 interactive charts
holoviews/*.py                  → 6 interactive charts
streamlit/dashboard.py          → Interactive dashboard
```

## Troubleshooting

**Charts not generating?**
- Check that `../results/` and `../models/results/` have data
- Run from correct directory

**Streamlit port busy?**
```bash
streamlit run streamlit/dashboard.py --server.port 8502
```

**Memory issues?**
- Use smaller datasets (5M instead of 50M)
- Generate one library at a time

## Summary

✅ 34 professional visualizations
✅ 5 visualization libraries
✅ Clean, organized structure
✅ Publication-ready charts
✅ Interactive dashboards
✅ Comprehensive documentation

Perfect for your thesis! 🎓

---

**Status**: ✅ Production Ready | **Created**: 2025-10-14
