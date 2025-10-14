# Quick Start Guide

## One Command - Generate Everything

```bash
cd data_visualization
python generate_all_visualizations.py
```

**Result**: 34 visualizations in ~3 minutes

---

## Individual Libraries

### Matplotlib (Static, for Thesis Document)
```bash
cd matplotlib
python data_processing_visualization.py    # 6 PNG charts
python ml_frameworks_visualization.py       # 6 PNG charts
```
**Output**: `matplotlib/output/` (12 PNG files, 300 DPI)

### Plotly (Interactive HTML)
```bash
cd plotly
python data_processing_visualization.py    # 6 HTML charts
python ml_frameworks_visualization.py       # 4 HTML charts
```
**Output**: `plotly/output/` (10 HTML files)

### Bokeh (Interactive HTML)
```bash
cd bokeh
python combined_visualization.py           # 6 HTML charts
```
**Output**: `bokeh/output/` (6 HTML files)

### Holoviews (Interactive HTML)
```bash
cd holoviews
python combined_visualization.py           # 6 HTML charts
```
**Output**: `holoviews/output/` (6 HTML files)

### Streamlit (Dashboard)
```bash
cd streamlit
streamlit run dashboard.py                 # Opens in browser
```
**Output**: Interactive web application

---

## View Generated Charts

```bash
# Open interactive charts in browser
start plotly/output/execution_time_interactive.html
start bokeh/output/ml_memory_usage.html

# View static charts
start matplotlib/output/ml_training_time_comparison.png
```

---

## Quick Chart Locations

| What You Need | Where to Find It |
|---------------|------------------|
| **Thesis document charts** | `matplotlib/output/` (PNG files) |
| **Online interactive charts** | `plotly/output/` (HTML files) |
| **Dashboard for presentation** | `streamlit/dashboard.py` |
| **All visualizations** | Each library's `output/` folder |

---

## Recommendations

### For Written Thesis
📄 Use **Matplotlib** PNG files from `matplotlib/output/`
- High resolution (300 DPI)
- Perfect for LaTeX/Word
- Small file sizes

### For Thesis Defense
🎤 Use **Streamlit Dashboard**
```bash
streamlit run streamlit/dashboard.py
```
- Interactive exploration
- Real-time filtering
- Professional appearance

### For Digital/Online Thesis
🌐 Use **Plotly** HTML files from `plotly/output/`
- Self-contained
- No server needed
- Beautiful interactivity

---

## Folder Structure After Generation

```
data_visualization/
├── matplotlib/output/          ← 12 PNG charts
├── plotly/output/              ← 10 HTML charts
├── bokeh/output/               ← 6 HTML charts
├── holoviews/output/           ← 6 HTML charts
└── streamlit/dashboard.py      ← Interactive app
```

---

## Common Commands

```bash
# Generate all visualizations
python generate_all_visualizations.py

# Run Streamlit dashboard
streamlit run streamlit/dashboard.py

# Run from root directory
cd data_visualization
python generate_all_visualizations.py
```

---

## Troubleshooting

### Port Already in Use (Streamlit)
```bash
streamlit run streamlit/dashboard.py --server.port 8502
```

### Missing Data Files
Ensure these folders have JSON files:
- `../results/` - Data processing results
- `../models/results/` - ML/DL results

### Memory Issues
Generate one library at a time instead of all at once.

---

## Summary

**Fastest way**: Run `python generate_all_visualizations.py`

**For thesis document**: Use `matplotlib/output/` PNG files

**For presentation**: Use `streamlit/dashboard.py`

**For online viewing**: Use `plotly/output/` HTML files

That's it! 🎉
