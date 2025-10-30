"""
═══════════════════════════════════════════════════════════════════════════════
CZYSTE PRZYKŁADY KODU DLA LISTINGÓW LATEX
═══════════════════════════════════════════════════════════════════════════════

Ten plik zawiera UPROSZCZONE wersje kodu, idealne do pokazania w pracy 
magisterskiej. Każdy przykład jest minimalny, ale funkcjonalny.

Użycie w LaTeX:
\begin{lstlisting}[language=Python, caption={...}, label={lst:...}]
# skopiuj kod stąd
\end{lstlisting}

"""

# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 1: PLOTLY - Najkrótsza Implementacja (8 LOC)
# ═══════════════════════════════════════════════════════════════════════════

def plotly_example_chart1():
    """PLOTLY: Bar Chart - Execution Time"""
    import plotly.express as px
    
    # Dane
    df = pd.DataFrame({
        'Library': ['Pandas', 'Polars', 'PyArrow', 'Dask', 'Spark'],
        'Time': [11.0, 1.51, 5.31, 22.28, 99.87]
    })
    
    # Wykres w 2 liniach!
    fig = px.bar(df, x='Library', y='Time', color='Library',
                 title='Performance Comparison')
    fig.write_html('output.html')


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 2: BOKEH - Niskopoziomowe API (25 LOC)
# ═══════════════════════════════════════════════════════════════════════════

def bokeh_example_chart1():
    """BOKEH: Bar Chart - Wymaga Więcej Kodu"""
    from bokeh.plotting import figure, output_file, save
    from bokeh.models import ColumnDataSource, HoverTool
    
    # Dane - wymaga ColumnDataSource
    source = ColumnDataSource(data=dict(
        libraries=['Pandas', 'Polars', 'PyArrow', 'Dask', 'Spark'],
        times=[11.0, 1.51, 5.31, 22.28, 99.87]
    ))
    
    # Figure creation
    p = figure(x_range=['Pandas', 'Polars', 'PyArrow', 'Dask', 'Spark'],
               title="Performance Comparison",
               width=800, height=500)
    
    # Dodaj słupki
    p.vbar(x='libraries', top='times', width=0.7, source=source)
    
    # Hover tooltip - manualna konfiguracja
    hover = HoverTool(tooltips=[("Library", "@libraries"), 
                                ("Time", "@times{0.00}s")])
    p.add_tools(hover)
    
    # Osie - manualne ustawienia
    p.xaxis.axis_label = "Library"
    p.yaxis.axis_label = "Time (seconds)"
    
    output_file('output.html')
    save(p)


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 3: MATPLOTLIB - Publikacje (20 LOC)
# ═══════════════════════════════════════════════════════════════════════════

def matplotlib_example_chart1():
    """MATPLOTLIB: Bar Chart - Wysokiej Jakości PNG"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Dane
    libraries = ['Pandas', 'Polars', 'PyArrow', 'Dask', 'Spark']
    times = [11.0, 1.51, 5.31, 22.28, 99.87]
    
    # Wykres
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(libraries))
    bars = ax.bar(x, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', 
                                     '#d62728', '#9467bd'])
    
    # Etykiety na słupkach
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # Opisz osie
    ax.set_xlabel('Library', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(libraries)
    
    plt.savefig('output.png', dpi=300, bbox_inches='tight')


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 4: STREAMLIT - Dashboard (15 LOC)
# ═══════════════════════════════════════════════════════════════════════════

def streamlit_example_chart1():
    """STREAMLIT: Interaktywny Dashboard"""
    import streamlit as st
    import plotly.express as px
    
    st.title("📊 Performance Analysis")
    
    # Dane
    df = pd.DataFrame({
        'Library': ['Pandas', 'Polars', 'PyArrow', 'Dask', 'Spark'],
        'Time': [11.0, 1.51, 5.31, 22.28, 99.87]
    })
    
    # Metryki (unique feature!)
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Fastest", "Polars", "1.51s")
    col2.metric("📊 Average", "All", "27.99s")
    col3.metric("🐌 Slowest", "Spark", "99.87s")
    
    # Wykres
    fig = px.bar(df, x='Library', y='Time', color='Library')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 5: HOLOVIEWS - Deklaratywne API (12 LOC)
# ═══════════════════════════════════════════════════════════════════════════

def holoviews_example_chart1():
    """HOLOVIEWS: Czysty, Deklaratywny Kod"""
    import holoviews as hv
    from holoviews import opts
    hv.extension('bokeh')
    
    # Dane
    df = pd.DataFrame({
        'Library': ['Pandas', 'Polars', 'PyArrow', 'Dask', 'Spark'],
        'Time': [11.0, 1.51, 5.31, 22.28, 99.87]
    })
    
    # Wykres - jedna linia!
    bars = hv.Bars(df, kdims=['Library'], vdims=['Time'])
    
    # Styling - przez .opts()
    bars.opts(opts.Bars(width=800, height=500, 
                        title="Performance Comparison",
                        color='Library', tools=['hover']))
    
    hv.save(bars, 'output.html')


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 6: GROUPED BARS - Największe Różnice!
# ═══════════════════════════════════════════════════════════════════════════

def plotly_grouped_bars():
    """PLOTLY: Grouped Bars - Automatyczne"""
    import plotly.express as px
    
    # Dane: 6 operacji × 5 bibliotek
    df = pd.DataFrame({
        'Library': ['Pandas']*6 + ['Polars']*6,
        'Operation': ['Load', 'Clean', 'Agg', 'Sort', 'Filter', 'Corr']*2,
        'Time': [6.5, 0.8, 1.2, 2.1, 0.5, 10.2,  # Pandas
                 0.4, 0.1, 0.3, 0.2, 0.1, 0.5]   # Polars
    })
    
    # Grouped bars w JEDNYM parametrze!
    fig = px.bar(df, x='Operation', y='Time', color='Library',
                 barmode='group')  # <-- MAGIA!
    fig.write_html('grouped.html')


def bokeh_grouped_bars():
    """BOKEH: Grouped Bars - Manualne Pozycjonowanie"""
    from bokeh.plotting import figure, save
    from bokeh.models import ColumnDataSource
    
    operations = ['Load', 'Clean', 'Agg', 'Sort', 'Filter', 'Corr']
    pandas_times = [6.5, 0.8, 1.2, 2.1, 0.5, 10.2]
    polars_times = [0.4, 0.1, 0.3, 0.2, 0.1, 0.5]
    
    # MANUALNE obliczanie pozycji!
    x_offset_pandas = [-0.15, 0.85, 1.85, 2.85, 3.85, 4.85]
    x_offset_polars = [0.15, 1.15, 2.15, 3.15, 4.15, 5.15]
    
    p = figure(x_range=operations, width=800, height=500)
    
    # Każda biblioteka osobno
    p.vbar(x=x_offset_pandas, top=pandas_times, width=0.25,
           color='blue', legend_label='Pandas')
    p.vbar(x=x_offset_polars, top=polars_times, width=0.25,
           color='orange', legend_label='Polars')
    
    save(p, 'grouped.html')


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 7: PORÓWNANIE WIELOLINIOWE
# ═══════════════════════════════════════════════════════════════════════════

def plotly_multiline():
    """PLOTLY: Multi-line Chart"""
    import plotly.express as px
    
    df = pd.DataFrame({
        'Size': [5, 10, 50] * 3,
        'Time': [2.1, 11.0, 87.3,    # Pandas
                 0.5, 1.51, 6.2,      # Polars
                 1.8, 5.31, 42.1],    # PyArrow
        'Library': ['Pandas']*3 + ['Polars']*3 + ['PyArrow']*3
    })
    
    # Multi-line w jednej linii!
    fig = px.line(df, x='Size', y='Time', color='Library', markers=True)
    fig.write_html('scalability.html')


def matplotlib_multiline():
    """MATPLOTLIB: Multi-line Chart"""
    import matplotlib.pyplot as plt
    
    sizes = [5, 10, 50]
    pandas = [2.1, 11.0, 87.3]
    polars = [0.5, 1.51, 6.2]
    pyarrow = [1.8, 5.31, 42.1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Każda linia osobno
    ax.plot(sizes, pandas, marker='o', label='Pandas', linewidth=2)
    ax.plot(sizes, polars, marker='o', label='Polars', linewidth=2)
    ax.plot(sizes, pyarrow, marker='o', label='PyArrow', linewidth=2)
    
    ax.set_xlabel('Dataset Size (M rows)')
    ax.set_ylabel('Time (seconds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('scalability.png', dpi=300)


# ═══════════════════════════════════════════════════════════════════════════
# PODSUMOWANIE LOC (Lines of Code)
# ═══════════════════════════════════════════════════════════════════════════

"""
PODSUMOWANIE - Chart 1 (Simple Bar):
─────────────────────────────────────
Plotly:      8 LOC   ████░░░░░░  (najkrótszy!)
Holoviews:  12 LOC   ██████░░░░
Streamlit:  15 LOC   ███████░░░
Matplotlib: 20 LOC   █████████░
Bokeh:      25 LOC   ██████████  (najdłuższy)

RÓŻNICA: Plotly = 68% mniej kodu niż Bokeh!


PODSUMOWANIE - Chart 2 (Grouped Bars):
───────────────────────────────────────
Plotly:     10 LOC (barmode='group')
Bokeh:      35 LOC (manual x-offsets!)

RÓŻNICA: 71% więcej kodu w Bokeh!
"""




