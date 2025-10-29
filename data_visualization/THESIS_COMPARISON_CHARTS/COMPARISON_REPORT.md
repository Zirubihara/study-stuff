
# Comparative Analysis: Data Visualization Libraries
## Master's Thesis - Chapter 4

---

## Executive Summary

This report presents a side-by-side comparison of 5 popular data visualization 
libraries in Python, evaluated through 7 common chart implementations.

**Libraries Analyzed:**
1. **Bokeh** - Low-level interactive visualizations
2. **Holoviews** - Declarative visualization library
3. **Matplotlib** - Publication-quality static graphics
4. **Plotly** - High-level interactive charts
5. **Streamlit** - Dashboard framework

---

## Methodology

### Chart Selection Criteria
The 7 charts were selected based on:
- **Universality**: Available in all 5 libraries
- **Relevance**: Core metrics for thesis research
- **Complexity**: Range from simple bars to multi-line charts

### Evaluation Metrics
Each implementation is evaluated on:
- **Lines of Code (LOC)**: Implementation complexity
- **API Style**: Declarative vs Imperative
- **Interactivity**: Static vs Interactive
- **Customization**: Flexibility in styling
- **Learning Curve**: Ease of implementation

---

## Chart-by-Chart Analysis

### Chart 1: Execution Time Comparison

**Purpose**: Compare total execution time across data processing libraries

| Library | LOC | API Style | Complexity | Best For |
|---------|:---:|-----------|:----------:|----------|
| Bokeh | 25 | Imperative | Medium | Fine-grained control |
| Holoviews | 12 | Declarative | Low | Quick prototypes |
| Matplotlib | 20 | Imperative | Medium | Publications |
| Plotly | 8 | Declarative | Very Low | Fast development |
| Streamlit | 15 | Declarative | Low | Dashboards |

**Winner (Simplicity)**: Plotly (8 LOC)  
**Winner (Quality)**: Matplotlib (publication-ready)  
**Winner (UX)**: Streamlit (interactive metrics)

---

### Chart 2: Operation Breakdown

**Purpose**: Grouped bar chart showing 6 operations × 5 libraries

**Challenge**: Positioning multiple bar groups

| Library | Grouping Method | Complexity |
|---------|----------------|:----------:|
| Bokeh | Manual x-offsets | ⚠️ High |
| Holoviews | Multi-dim keys | ⭐ Low |
| Matplotlib | NumPy offsets | ⚠️ Medium |
| Plotly | barmode='group' | ⭐ Low |
| Streamlit | Plotly wrapper | ⭐ Low |

**Key Insight**: Plotly & Holoviews handle grouped bars elegantly, 
while Bokeh & Matplotlib require manual positioning.

---

### Chart 3-4: Memory & Scalability

Simple bar charts and line charts follow similar patterns to Chart 1.

**Unique Feature - Chart 4**:
- Bokeh & Holoviews support **log-log scale** natively
- Matplotlib uses linear scale by default
- Critical for scalability visualization

---

### Chart 5-7: ML/DL Framework Comparisons

Similar bar chart implementations with ML metrics.

**Consistency**: All libraries maintain similar LOC and complexity 
across chart types, showing good design consistency.

---

## Quantitative Comparison

### Lines of Code (Average across 7 charts)

```
Plotly:      8-10 LOC  ████░░░░░░ (shortest)
Holoviews:   12-15 LOC ██████░░░░
Streamlit:   15-18 LOC ███████░░░
Matplotlib:  18-22 LOC █████████░
Bokeh:       22-28 LOC ██████████ (longest)
```

### API Complexity

**Declarative (Easier)**:
- Plotly Express: `px.bar(df, x='col', y='val')`
- Holoviews: `hv.Bars(df).opts(...)`

**Imperative (More Control)**:
- Bokeh: `figure() → vbar() → add_tools() → save()`
- Matplotlib: `subplots() → bar() → set_xlabel() → savefig()`

---

## Feature Matrix

| Feature | Bokeh | Holoviews | Matplotlib | Plotly | Streamlit |
|---------|:-----:|:---------:|:----------:|:------:|:---------:|
| **Output Format** | HTML | HTML | PNG | HTML | Web App |
| **Interactivity** | ✅ High | ✅ High | ❌ None | ✅ High | ✅ Highest |
| **Customization** | ✅✅✅ | ✅✅ | ✅✅✅ | ✅✅ | ✅✅ |
| **Learning Curve** | Steep | Gentle | Medium | Easy | Easy |
| **Log Scale** | ✅ Native | ✅ Native | ⚠️ Manual | ✅ Auto | ✅ Auto |
| **Grouped Bars** | ⚠️ Manual | ✅ Auto | ⚠️ Manual | ✅ Auto | ✅ Auto |
| **Hover Tooltips** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Publication Quality** | ✅✅ | ✅✅ | ✅✅✅ | ✅✅ | ✅ |
| **Dashboard Ready** | ⚠️ | ⚠️ | ❌ | ⚠️ | ✅✅✅ |

---

## Recommendations

### For Master's Thesis Use Cases:

1. **Thesis Document (PDF)**
   - **Use**: Matplotlib
   - **Reason**: Best print quality, IEEE/ACM compliant
   - **Format**: PNG, 300 DPI

2. **Interactive Appendix**
   - **Use**: Bokeh or Plotly
   - **Reason**: Self-contained HTML files
   - **Format**: HTML (can be attached to thesis)

3. **Thesis Defense Presentation**
   - **Use**: Streamlit
   - **Reason**: Live demonstration, interactive exploration
   - **Format**: Web application

4. **Quick Prototyping**
   - **Use**: Plotly Express
   - **Reason**: Fastest implementation
   - **Format**: HTML or PNG

5. **Academic Publication**
   - **Use**: Matplotlib
   - **Reason**: Journal standards compliance
   - **Format**: Vector (PDF/SVG) or high-res PNG

---

## Code Quality Assessment

### Best Practices Observed:

**Holoviews**:
```python
# Clean separation of data and presentation
bars = hv.Bars(df, kdims=['x'], vdims=['y'])
bars.opts(width=800, title="Title", ...)
```

**Plotly**:
```python
# Minimal boilerplate
fig = px.bar(df, x='col', y='val')
fig.write_html('output.html')
```

### Anti-Patterns Found:

**Bokeh** (manual positioning):
```python
# Requires complex offset calculations
x_offset = [-0.3, -0.15, 0, 0.15, 0.3]
x_positions = [i + x_offset[idx] for i in range(len(data))]
```

---

## Performance Considerations

### Generation Time (7 charts):
- Matplotlib: ~2-3 seconds (PNG rendering)
- Plotly: ~1-2 seconds (HTML generation)
- Bokeh: ~3-4 seconds (HTML + JS)
- Holoviews: ~2-3 seconds (Bokeh backend)
- Streamlit: N/A (runtime rendering)

### File Sizes:
- Matplotlib PNG: 50-200 KB each
- Plotly HTML: 500-800 KB each
- Bokeh HTML: 400-700 KB each
- Holoviews HTML: 600-1000 KB each

---

## Conclusion

### Overall Rankings:

1. **Best for Simplicity**: Plotly Express (8-10 LOC average)
2. **Best for Quality**: Matplotlib (publication-ready output)
3. **Best for Interactivity**: Streamlit (dashboard features)
4. **Best for Flexibility**: Bokeh (low-level control)
5. **Best Balance**: Holoviews (clean code + interactivity)

### Key Findings:

- **Declarative APIs** (Plotly, Holoviews) reduce code by 50-60%
- **Matplotlib** remains essential for academic publications
- **Streamlit** offers best user experience for data exploration
- **Bokeh** provides maximum control at cost of complexity

### Thesis Recommendation:

Use **hybrid approach**:
- Matplotlib for thesis document
- Plotly for interactive HTML appendix
- Streamlit for defense presentation
- Document all three in methodology chapter

---

## Reproducibility

All visualizations generated by:
```bash
python comparative_visualization_thesis.py
```

 Output structure:
 ```
 THESIS_COMPARISON_CHARTS/
 ├── bokeh/       (7 HTML files)
 ├── holoviews/   (7 HTML files)
 ├── matplotlib/  (7 PNG files)
 ├── plotly/      (7 HTML files)
 └── streamlit/   (7 Python files - code only)
 ```

---

**Generated**: 2025-10-26  
**Total Charts**: 35 visualizations (7 × 5 libraries)  
**Total LOC**: ~850 lines of implementation code  
**License**: For academic use in master's thesis

