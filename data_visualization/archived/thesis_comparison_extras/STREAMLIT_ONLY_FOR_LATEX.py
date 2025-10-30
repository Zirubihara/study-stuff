"""
═══════════════════════════════════════════════════════════════════════════════
STREAMLIT - PRZYKŁADY KODU DLA LISTINGÓW LATEX
═══════════════════════════════════════════════════════════════════════════════

Ten plik zawiera TYLKO przykłady Streamlit, gotowe do użycia w pracy.

"""

# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 1: Prosty Wykres z Metrykami
# ═══════════════════════════════════════════════════════════════════════════


def streamlit_simple_chart():
    """STREAMLIT: Prosty wykres słupkowy z metrykami"""
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.title("📊 Performance Analysis")

    # Dane
    df = pd.DataFrame(
        {
            "Library": ["Pandas", "Polars", "PyArrow", "Dask", "Spark"],
            "Time": [11.0, 1.51, 5.31, 22.28, 99.87],
        }
    )

    # Metryki w kolumnach
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Fastest", "Polars", "1.51s")
    col2.metric("📊 Average", "All", "27.99s")
    col3.metric("🐌 Slowest", "Spark", "99.87s")

    # Wykres
    fig = px.bar(df, x="Library", y="Time", color="Library")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 2: Tabs (Zakładki)
# ═══════════════════════════════════════════════════════════════════════════


def streamlit_with_tabs():
    """STREAMLIT: Dashboard z zakładkami"""
    import plotly.express as px
    import streamlit as st

    st.title("📊 Multi-Chart Dashboard")

    # Zakładki
    tab1, tab2, tab3 = st.tabs(["Chart 1", "Chart 2", "Chart 3"])

    with tab1:
        st.subheader("Execution Time")
        fig1 = px.bar(df, x="Library", y="Time")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader("Memory Usage")
        fig2 = px.bar(df, x="Library", y="Memory")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Scalability")
        fig3 = px.line(df, x="Size", y="Time", color="Library")
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 3: Sidebar + Filtrowanie
# ═══════════════════════════════════════════════════════════════════════════


def streamlit_with_sidebar():
    """STREAMLIT: Interaktywne filtrowanie"""
    import plotly.express as px
    import streamlit as st

    st.title("📊 Interactive Filtering")

    # Sidebar
    st.sidebar.header("Filters")
    selected_libs = st.sidebar.multiselect(
        "Select Libraries:",
        options=["Pandas", "Polars", "PyArrow", "Dask", "Spark"],
        default=["Pandas", "Polars", "PyArrow"],
    )

    # Filtruj dane
    filtered_df = df[df["Library"].isin(selected_libs)]

    # Wykres
    fig = px.bar(filtered_df, x="Library", y="Time", color="Library")
    st.plotly_chart(fig, use_container_width=True)

    # Pokaż dane
    st.dataframe(filtered_df)


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 4: Expanders (Rozwijane sekcje)
# ═══════════════════════════════════════════════════════════════════════════


def streamlit_with_expanders():
    """STREAMLIT: Expandable sections"""
    import plotly.express as px
    import streamlit as st

    st.title("📊 Analysis Dashboard")

    # Wykres
    fig = px.bar(df, x="Library", y="Time", color="Library")
    st.plotly_chart(fig, use_container_width=True)

    # Rozwijane sekcje
    with st.expander("📝 Key Findings"):
        st.write(
            """
        **Performance Analysis:**
        - Polars is 66× faster than Spark
        - PyArrow shows good balance
        - Pandas is baseline reference
        """
        )

    with st.expander("📊 Statistical Summary"):
        st.write(df.describe())

    with st.expander("💡 Recommendations"):
        st.write(
            """
        - Use Polars for speed
        - Use Pandas for compatibility
        - Use Spark for distributed computing
        """
        )


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 5: Multi-page App
# ═══════════════════════════════════════════════════════════════════════════


def streamlit_multipage():
    """STREAMLIT: Multi-page navigation"""
    import plotly.express as px
    import streamlit as st

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page:", ["Home", "Analysis", "Comparison"])

    if page == "Home":
        st.title("🏠 Welcome")
        st.write("This is a multi-page Streamlit application")

    elif page == "Analysis":
        st.title("📊 Performance Analysis")
        fig = px.bar(df, x="Library", y="Time", color="Library")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Comparison":
        st.title("📈 Library Comparison")
        fig = px.scatter(df, x="Time", y="Memory", text="Library", size="Score")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PRZYKŁAD 6: Real-time Updates (Cache)
# ═══════════════════════════════════════════════════════════════════════════


def streamlit_with_cache():
    """STREAMLIT: Caching dla wydajności"""
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    @st.cache_data  # Cache data loading!
    def load_data():
        """Load data once, reuse on reruns"""
        df = pd.DataFrame(
            {"Library": ["Pandas", "Polars", "PyArrow"], "Time": [11.0, 1.51, 5.31]}
        )
        return df

    st.title("📊 Optimized Dashboard")

    # Load cached data
    df = load_data()

    # Dynamic chart
    chart_type = st.selectbox("Chart Type:", ["Bar", "Line", "Scatter"])

    if chart_type == "Bar":
        fig = px.bar(df, x="Library", y="Time")
    elif chart_type == "Line":
        fig = px.line(df, x="Library", y="Time")
    else:
        fig = px.scatter(df, x="Library", y="Time", size="Time")

    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# MINIMAL EXAMPLE (DO PRACY MAGISTERSKIEJ)
# ═══════════════════════════════════════════════════════════════════════════


def streamlit_minimal():
    """STREAMLIT: Absolutne minimum (10 LOC)"""
    import plotly.express as px
    import streamlit as st

    st.title("Performance Comparison")

    df = pd.DataFrame(
        {"Library": ["Pandas", "Polars", "Spark"], "Time": [11.0, 1.51, 99.87]}
    )

    fig = px.bar(df, x="Library", y="Time")
    st.plotly_chart(fig)


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON: STREAMLIT vs OTHERS
# ═══════════════════════════════════════════════════════════════════════════

"""
STREAMLIT UNIQUE FEATURES:
══════════════════════════

1. METRICS (unikalne dla Streamlit):
   col1.metric("Label", "Value", "Delta")
   
2. TABS (prostsze niż w innych):
   tab1, tab2 = st.tabs(["A", "B"])
   
3. SIDEBAR (wbudowany):
   st.sidebar.selectbox(...)
   
4. EXPANDERS (natywne):
   with st.expander("Title"):
       st.write(...)
       
5. REACTIVITY (automatyczna):
   - Każda zmiana → odświeżenie
   - Nie trzeba button.on_click()
   
6. CACHING (@st.cache_data):
   - Automatyczne cachowanie
   - Nie trzeba manualnie

LINES OF CODE:
══════════════
Streamlit: 15 LOC average
Plotly:     8 LOC (krótszy, ale bez UI)
Bokeh:     25 LOC (dłuższy)
Matplotlib: 20 LOC (statyczny)

BEST USE CASE:
══════════════
✓ Dashboards (najlepszy)
✓ Prototypy (szybkie)
✓ Demos (interaktywne)
✗ Publications (nie PNG)
✗ Embedding (wymaga serwera)
"""



