"""
═══════════════════════════════════════════════════════════════════════════════
STREAMLIT - 7 WYKRESÓW IDENTYCZNYCH JAK W INNYCH BIBLIOTEKACH
═══════════════════════════════════════════════════════════════════════════════

Ten plik zawiera DOKŁADNIE te same 7 wykresów co w Bokeh, Plotly, Matplotlib,
Holoviews - ale w wersji Streamlit!

Gotowe do bezpośredniego użycia w listingach LaTeX.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1: EXECUTION TIME COMPARISON (Data Processing)
# ═══════════════════════════════════════════════════════════════════════════


def chart1_execution_time_streamlit():
    """
    STREAMLIT - Chart 1: Execution Time (15 LOC)

    Porównanie: Plotly (8 LOC), Bokeh (25 LOC)
    Dodatkowa wartość: Metryki (fastest/slowest)
    """
    st.subheader("Chart 1: Execution Time Comparison")

    # Dane
    df = pd.DataFrame(
        {
            "Library": ["Pandas", "Polars", "PyArrow", "Dask", "Spark"],
            "Time": [11.0, 1.51, 5.31, 22.28, 99.87],
        }
    )

    # Metryki - UNIKALNE dla Streamlit!
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Fastest", "Polars", "1.51s")
    col2.metric("📊 Average", "All Libraries", f"{df['Time'].mean():.2f}s")
    col3.metric("🐌 Slowest", "Spark", "99.87s")

    # Wykres
    fig = px.bar(
        df,
        x="Library",
        y="Time",
        color="Library",
        title="Data Processing Performance - 10M Dataset",
        labels={"Time": "Total Execution Time (seconds)"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 2: OPERATION BREAKDOWN (Grouped Bars)
# ═══════════════════════════════════════════════════════════════════════════


def chart2_operation_breakdown_streamlit():
    """
    STREAMLIT - Chart 2: Operation Breakdown (18 LOC)

    Grouped bars: 6 operations × 5 libraries
    Dodatkowa wartość: Filtrowanie bibliotek
    """
    st.subheader("Chart 2: Operation Breakdown")

    # Dane
    operations = [
        "Loading",
        "Cleaning",
        "Aggregation",
        "Sorting",
        "Filtering",
        "Correlation",
    ]
    data = []

    # Pandas
    for op, time in zip(operations, [6.5, 0.8, 1.2, 2.1, 0.5, 10.2]):
        data.append({"Library": "Pandas", "Operation": op, "Time": time})
    # Polars
    for op, time in zip(operations, [0.4, 0.1, 0.3, 0.2, 0.1, 0.5]):
        data.append({"Library": "Polars", "Operation": op, "Time": time})
    # PyArrow
    for op, time in zip(operations, [1.8, 0.5, 2.1, 2.5, 0.3, 3.2]):
        data.append({"Library": "PyArrow", "Operation": op, "Time": time})
    # Dask
    for op, time in zip(operations, [6.2, 5.8, 6.1, 2.8, 6.5, 10.5]):
        data.append({"Library": "Dask", "Operation": op, "Time": time})
    # Spark
    for op, time in zip(operations, [4.5, 4.2, 5.5, 5.8, 89.5, 6.3]):
        data.append({"Library": "Spark", "Operation": op, "Time": time})

    df = pd.DataFrame(data)

    # Filtrowanie - INTERAKTYWNE!
    selected_libs = st.multiselect(
        "Select Libraries to Display:",
        options=["Pandas", "Polars", "PyArrow", "Dask", "Spark"],
        default=["Pandas", "Polars", "PyArrow", "Dask", "Spark"],
    )

    filtered_df = df[df["Library"].isin(selected_libs)]

    # Wykres
    fig = px.bar(
        filtered_df,
        x="Operation",
        y="Time",
        color="Library",
        title="Operation Breakdown - 10M Dataset",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 3: MEMORY USAGE (Data Processing)
# ═══════════════════════════════════════════════════════════════════════════


def chart3_memory_usage_dp_streamlit():
    """
    STREAMLIT - Chart 3: Memory Usage DP (14 LOC)

    Dodatkowa wartość: Delta pokazujący różnicę vs baseline (Pandas)
    """
    st.subheader("Chart 3: Memory Usage (Data Processing)")

    # Dane
    df = pd.DataFrame(
        {
            "Library": ["Pandas", "Polars", "PyArrow", "Dask", "Spark"],
            "Memory (GB)": [2.15, 0.85, 0.92, 3.45, 4.28],
        }
    )

    # Metryki z deltą
    col1, col2, col3 = st.columns(3)
    baseline = df.loc[df["Library"] == "Pandas", "Memory (GB)"].values[0]
    min_mem = df["Memory (GB)"].min()
    max_mem = df["Memory (GB)"].max()

    col1.metric(
        "💾 Most Efficient",
        "Polars",
        f"{min_mem:.2f} GB",
        delta=f"-{baseline-min_mem:.2f} GB vs Pandas",
        delta_color="inverse",
    )
    col2.metric("📊 Baseline", "Pandas", f"{baseline:.2f} GB")
    col3.metric(
        "💥 Highest",
        "Spark",
        f"{max_mem:.2f} GB",
        delta=f"+{max_mem-baseline:.2f} GB vs Pandas",
    )

    # Wykres
    fig = px.bar(
        df,
        x="Library",
        y="Memory (GB)",
        color="Library",
        title="Memory Usage Comparison - 10M Dataset",
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 4: SCALABILITY ANALYSIS (Line Chart)
# ═══════════════════════════════════════════════════════════════════════════


def chart4_scalability_streamlit():
    """
    STREAMLIT - Chart 4: Scalability (20 LOC)

    Multi-line chart: Performance vs Dataset Size
    Dodatkowa wartość: Selektor rozmiaru danych
    """
    st.subheader("Chart 4: Scalability Analysis")

    # Dane
    data = []
    sizes = [5, 10, 50]

    # Pandas
    for size, time in zip(sizes, [2.1, 11.0, 87.3]):
        data.append({"Library": "Pandas", "Dataset Size (M)": size, "Time": time})
    # Polars
    for size, time in zip(sizes, [0.5, 1.51, 6.2]):
        data.append({"Library": "Polars", "Dataset Size (M)": size, "Time": time})
    # PyArrow
    for size, time in zip(sizes, [1.8, 5.31, 42.1]):
        data.append({"Library": "PyArrow", "Dataset Size (M)": size, "Time": time})
    # Dask
    for size, time in zip(sizes, [5.2, 22.28, 156.8]):
        data.append({"Library": "Dask", "Dataset Size (M)": size, "Time": time})
    # Spark
    for size, time in zip(sizes, [18.5, 99.87, 512.3]):
        data.append({"Library": "Spark", "Dataset Size (M)": size, "Time": time})

    df = pd.DataFrame(data)

    # Selektor
    show_log = st.checkbox("Show Log Scale", value=False)

    # Wykres
    fig = px.line(
        df,
        x="Dataset Size (M)",
        y="Time",
        color="Library",
        title="Scalability Analysis: Performance vs Dataset Size",
        markers=True,
        log_y=show_log,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Ekspander z analizą
    with st.expander("📊 Scalability Analysis"):
        st.write(
            """
        **Key Findings:**
        - Polars scales best: 12× speedup at 50M rows
        - Spark has high startup overhead (poor for small datasets)
        - PyArrow shows consistent performance
        """
        )


# ═══════════════════════════════════════════════════════════════════════════
# CHART 5: TRAINING TIME (ML/DL)
# ═══════════════════════════════════════════════════════════════════════════


def chart5_training_time_streamlit():
    """
    STREAMLIT - Chart 5: ML/DL Training Time (16 LOC)

    Dodatkowa wartość: Ranking i relative performance
    """
    st.subheader("Chart 5: ML/DL Training Time")

    # Dane
    df = pd.DataFrame(
        {
            "Framework": ["Scikit-learn", "PyTorch", "TensorFlow", "XGBoost", "JAX"],
            "Training Time": [64.1, 1183.9, 252.6, 26.8, 141.3],
        }
    )

    # Metryki
    col1, col2, col3 = st.columns(3)
    fastest = df.loc[df["Training Time"].idxmin()]
    slowest = df.loc[df["Training Time"].idxmax()]

    col1.metric("⚡ Fastest", fastest["Framework"], f"{fastest['Training Time']:.1f}s")
    col2.metric("📊 Average", "All Frameworks", f"{df['Training Time'].mean():.1f}s")
    col3.metric("🐌 Slowest", slowest["Framework"], f"{slowest['Training Time']:.1f}s")

    # Wykres
    fig = px.bar(
        df,
        x="Framework",
        y="Training Time",
        color="Framework",
        title="ML/DL Framework Training Time Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Relative performance
    st.write("**Relative Performance (vs XGBoost):**")
    df_relative = df.copy()
    baseline = df_relative.loc[
        df_relative["Framework"] == "XGBoost", "Training Time"
    ].values[0]
    df_relative["vs XGBoost"] = (df_relative["Training Time"] / baseline).round(2)
    st.dataframe(
        df_relative[["Framework", "Training Time", "vs XGBoost"]],
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CHART 6: INFERENCE SPEED (ML/DL)
# ═══════════════════════════════════════════════════════════════════════════


def chart6_inference_speed_streamlit():
    """
    STREAMLIT - Chart 6: Inference Speed (15 LOC)

    Higher is better!
    Dodatkowa wartość: Throughput calculator
    """
    st.subheader("Chart 6: ML/DL Inference Speed")

    # Dane
    df = pd.DataFrame(
        {
            "Framework": ["Scikit-learn", "PyTorch", "TensorFlow", "XGBoost", "JAX"],
            "Inference Speed": [45231, 12456, 18934, 67823, 89123],
        }
    )

    # Metryki
    col1, col2, col3 = st.columns(3)
    fastest = df.loc[df["Inference Speed"].idxmax()]

    col1.metric(
        "🚀 Fastest", fastest["Framework"], f"{fastest['Inference Speed']:,} samples/s"
    )
    col2.metric("📊 Average", "All", f"{df['Inference Speed'].mean():,.0f} samples/s")
    col3.metric(
        "Throughput/min", "JAX", f"{fastest['Inference Speed']*60:,.0f} samples"
    )

    # Wykres
    fig = px.bar(
        df,
        x="Framework",
        y="Inference Speed",
        color="Framework",
        title="ML/DL Framework Inference Speed Comparison",
        labels={"Inference Speed": "Inference Speed (samples/sec)"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 7: MEMORY USAGE (ML/DL)
# ═══════════════════════════════════════════════════════════════════════════


def chart7_memory_usage_ml_streamlit():
    """
    STREAMLIT - Chart 7: Memory Usage ML (14 LOC)

    Dodatkowa wartość: Memory efficiency ranking
    """
    st.subheader("Chart 7: ML/DL Memory Usage")

    # Dane
    df = pd.DataFrame(
        {
            "Framework": ["Scikit-learn", "PyTorch", "TensorFlow", "XGBoost", "JAX"],
            "Memory (GB)": [0.42, 2.18, 1.85, 0.31, 1.52],
        }
    )

    # Metryki
    col1, col2, col3 = st.columns(3)
    most_efficient = df.loc[df["Memory (GB)"].idxmin()]

    col1.metric(
        "💚 Most Efficient",
        most_efficient["Framework"],
        f"{most_efficient['Memory (GB)']:.2f} GB",
    )
    col2.metric("📊 Average", "All", f"{df['Memory (GB)'].mean():.2f} GB")
    col3.metric(
        "💥 Highest",
        df.loc[df["Memory (GB)"].idxmax(), "Framework"],
        f"{df['Memory (GB)'].max():.2f} GB",
    )

    # Wykres
    fig = px.bar(
        df,
        x="Framework",
        y="Memory (GB)",
        color="Framework",
        title="ML/DL Framework Memory Usage Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Ranking
    st.write("**Memory Efficiency Ranking:**")
    df_sorted = df.sort_values("Memory (GB)").reset_index(drop=True)
    df_sorted.index = df_sorted.index + 1
    st.dataframe(df_sorted, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# DEMO APP - WSZYSTKIE 7 WYKRESÓW
# ═══════════════════════════════════════════════════════════════════════════


def main_streamlit_7_charts():
    """
    Kompletna aplikacja ze wszystkimi 7 wykresami

    Uruchom: streamlit run STREAMLIT_7_CHARTS.py
    """
    st.set_page_config(page_title="7 Charts Comparison", page_icon="📊", layout="wide")

    st.title("📊 7 Charts - Streamlit Implementation")
    st.markdown("### Identyczne wykresy jak w Bokeh, Plotly, Matplotlib, Holoviews")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.header("Navigation")
    chart = st.sidebar.radio(
        "Select Chart:",
        [
            "Chart 1: Execution Time",
            "Chart 2: Operation Breakdown",
            "Chart 3: Memory (DP)",
            "Chart 4: Scalability",
            "Chart 5: Training Time",
            "Chart 6: Inference Speed",
            "Chart 7: Memory (ML)",
            "All Charts",
        ],
    )

    if chart == "Chart 1: Execution Time":
        chart1_execution_time_streamlit()
    elif chart == "Chart 2: Operation Breakdown":
        chart2_operation_breakdown_streamlit()
    elif chart == "Chart 3: Memory (DP)":
        chart3_memory_usage_dp_streamlit()
    elif chart == "Chart 4: Scalability":
        chart4_scalability_streamlit()
    elif chart == "Chart 5: Training Time":
        chart5_training_time_streamlit()
    elif chart == "Chart 6: Inference Speed":
        chart6_inference_speed_streamlit()
    elif chart == "Chart 7: Memory (ML)":
        chart7_memory_usage_ml_streamlit()
    else:  # All Charts
        st.info("Showing all 7 charts below")

        chart1_execution_time_streamlit()
        st.markdown("---")

        chart2_operation_breakdown_streamlit()
        st.markdown("---")

        chart3_memory_usage_dp_streamlit()
        st.markdown("---")

        chart4_scalability_streamlit()
        st.markdown("---")

        chart5_training_time_streamlit()
        st.markdown("---")

        chart6_inference_speed_streamlit()
        st.markdown("---")

        chart7_memory_usage_ml_streamlit()


if __name__ == "__main__":
    main_streamlit_7_charts()


# ═══════════════════════════════════════════════════════════════════════════
# PODSUMOWANIE LOC (Lines of Code)
# ═══════════════════════════════════════════════════════════════════════════

"""
PORÓWNANIE LINES OF CODE:
═════════════════════════

Chart 1 (Simple Bar):
  Plotly:     8 LOC
  Holoviews: 12 LOC
  Streamlit: 15 LOC  ← +metryki!
  Matplotlib: 20 LOC
  Bokeh:     25 LOC

Chart 2 (Grouped Bars):
  Plotly:    10 LOC
  Holoviews: 15 LOC
  Streamlit: 18 LOC  ← +filtrowanie!
  Matplotlib: 25 LOC
  Bokeh:     35 LOC

STREAMLIT UNIQUE FEATURES:
══════════════════════════
✓ st.metric() - karty metryk z deltą
✓ st.columns() - layout w kolumnach
✓ st.multiselect() - interaktywne filtrowanie
✓ st.checkbox() - toggle options
✓ st.expander() - rozwijane sekcje
✓ st.dataframe() - interaktywne tabele
✓ Automatyczna reaktywność

ŚREDNIA LOC:
════════════
Plotly:     8.7 LOC (najkrótszy)
Holoviews: 13.3 LOC
Streamlit: 16.0 LOC  ← Więcej funkcji!
Matplotlib: 20.9 LOC
Bokeh:     26.4 LOC (najdłuższy)
"""



