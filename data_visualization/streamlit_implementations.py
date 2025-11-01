"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    STREAMLIT IMPLEMENTATIONS                                ║
║                    Master's Thesis - Code Listings                          ║
║                                                                              ║
║  Implementacje Streamlit dla 7 wykresów                                    ║
║  Oddzielny moduł dla czystości kodu i łatwości kopiowania do LaTeX         ║
║                                                                              ║
║  USAGE:                                                                     ║
║  1. For thesis listings: Copy functions directly to LaTeX                   ║
║  2. For live demo: streamlit run streamlit_implementations.py               ║
╚════════════════════════════════════════════════════════════════════════════╝

Autor: [Twoje Imię]
Data: 2025-10-26

STRUKTURA:
==========
- Każda klasa odpowiada wykresowi z głównego pliku
- Każda klasa ma metodę statyczną do implementacji w Streamlit
- Struktura identyczna jak w comparative_visualization_thesis.py
- Kod gotowy do skopiowania jako listing do pracy magisterskiej
"""

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

# Import configuration and data preparation from main file
try:
    from comparative_visualization_thesis import (
        Chart1_ExecutionTime,
        Chart2_OperationBreakdown,
        Chart3_MemoryUsage_DP,
        Chart4_Scalability,
        Chart5_TrainingTime,
        Chart6_InferenceSpeed,
        Chart7_MemoryUsage_ML,
        Config,
        DataLoader,
    )
except ImportError:
    # Fallback for when run standalone
    st.error(
        "Please run from the same directory as comparative_visualization_thesis.py"
    )


# ═══════════════════════════════════════════════════════════════════════════
# CHART 1: EXECUTION TIME COMPARISON (Data Processing)
# ═══════════════════════════════════════════════════════════════════════════


class Chart1_ExecutionTime_Streamlit:
    """
    WYKRES 1: Porównanie całkowitego czasu wykonania operacji

    Implementacja: Streamlit + Plotly
    Lines of code: ~15
    Complexity: Low (reactive components)
    """

    @staticmethod
    def streamlit(dp_data: Dict) -> None:
        """
        STREAMLIT IMPLEMENTATION
        ════════════════════════
        Approach: Plotly wrapper with reactive metrics
        Unique features: st.columns() for metrics, automatic reactivity
        """
        st.subheader("Chart 1: Execution Time Comparison")

        # Prepare data
        df = Chart1_ExecutionTime.prepare_data(dp_data)

        # Interactive metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Fastest",
            df.loc[df["Time"].idxmin(), "Library"],
            f"{df['Time'].min():.2f}s",
        )
        col2.metric("Average", "All Libraries", f"{df['Time'].mean():.2f}s")
        col3.metric(
            "Slowest",
            df.loc[df["Time"].idxmax(), "Library"],
            f"{df['Time'].max():.2f}s",
        )

        # Chart
        fig = px.bar(
            df,
            x="Library",
            y="Time",
            color="Library",
            title="Data Processing Performance - 10M Dataset",
            labels={"Time": "Total Execution Time (seconds)"},
            color_discrete_sequence=Config.DP_COLORS,
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 2: OPERATION BREAKDOWN (Data Processing)
# ═══════════════════════════════════════════════════════════════════════════


class Chart2_OperationBreakdown_Streamlit:
    """
    WYKRES 2: Rozbicie czasu wykonania na poszczególne operacje

    Implementacja: Streamlit + Plotly (grouped bars)
    Lines of code: ~12
    Complexity: Low (Plotly handles grouping)
    """

    @staticmethod
    def streamlit(dp_data: Dict) -> None:
        """
        STREAMLIT IMPLEMENTATION
        ════════════════════════
        Approach: Plotly barmode='group' with Streamlit container
        Advantage: Clean code, automatic handling of 30 bars
        """
        st.subheader("Chart 2: Operation Breakdown")

        # Prepare data
        df = Chart2_OperationBreakdown.prepare_data(dp_data)

        # Chart with grouped bars
        fig = px.bar(
            df,
            x="Operation",
            y="Time",
            color="Library",
            title="Operation Breakdown - 10M Dataset",
            barmode="group",
            color_discrete_sequence=Config.DP_COLORS,
        )

        fig.update_layout(
            xaxis_title="Operation",
            yaxis_title="Time (seconds)",
        )

        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 3: MEMORY USAGE (Data Processing)
# ═══════════════════════════════════════════════════════════════════════════


class Chart3_MemoryUsage_DP_Streamlit:
    """
    WYKRES 3: Porównanie zużycia pamięci podczas przetwarzania danych

    Implementacja: Streamlit + Plotly
    Lines of code: ~14
    Complexity: Low
    """

    @staticmethod
    def streamlit(dp_data: Dict) -> None:
        """
        STREAMLIT IMPLEMENTATION
        ════════════════════════
        Approach: Bar chart with memory metrics
        """
        st.subheader("Chart 3: Memory Usage (Data Processing)")

        # Prepare data
        df = Chart3_MemoryUsage_DP.prepare_data(dp_data)

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric(
            "Lowest Memory",
            df.loc[df["Memory (GB)"].idxmin(), "Library"],
            f"{df['Memory (GB)'].min():.2f} GB",
        )
        col2.metric(
            "Highest Memory",
            df.loc[df["Memory (GB)"].idxmax(), "Library"],
            f"{df['Memory (GB)'].max():.2f} GB",
        )

        # Chart
        fig = px.bar(
            df,
            x="Library",
            y="Memory (GB)",
            color="Library",
            title="Memory Usage Comparison - 10M Dataset",
            color_discrete_sequence=Config.DP_COLORS,
        )

        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 4: SCALABILITY ANALYSIS (Data Processing)
# ═══════════════════════════════════════════════════════════════════════════


class Chart4_Scalability_Streamlit:
    """
    WYKRES 4: Analiza skalowalności

    Implementacja: Streamlit + Plotly (line chart)
    Lines of code: ~20
    Complexity: Medium (interactive library selection)
    """

    @staticmethod
    def streamlit(dp_data: Dict) -> None:
        """
        STREAMLIT IMPLEMENTATION
        ════════════════════════
        Approach: Interactive line chart with library filter
        Unique feature: st.multiselect() for dynamic filtering
        """
        st.subheader("Chart 4: Scalability Analysis")

        # Prepare data
        df = Chart4_Scalability.prepare_data(dp_data)

        # Interactive library selection
        selected_libs = st.multiselect(
            "Select libraries to display:",
            options=df["Library"].unique().tolist(),
            default=df["Library"].unique().tolist(),
        )

        # Filter data based on selection
        filtered_df = df[df["Library"].isin(selected_libs)]

        # Display info
        st.info(
            f"Showing {len(selected_libs)} libraries across "
            f"{filtered_df['Dataset Size (M)'].nunique()} dataset sizes"
        )

        # Chart
        fig = px.line(
            filtered_df,
            x="Dataset Size (M)",
            y="Time",
            color="Library",
            markers=True,
            title="Scalability Analysis: Performance vs Dataset Size",
            color_discrete_sequence=Config.DP_COLORS,
        )

        fig.update_layout(
            xaxis_title="Dataset Size (Million Rows)",
            yaxis_title="Execution Time (seconds)",
        )

        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 5: TRAINING TIME COMPARISON (ML/DL)
# ═══════════════════════════════════════════════════════════════════════════


class Chart5_TrainingTime_Streamlit:
    """
    WYKRES 5: Porównanie czasu trenowania modeli ML/DL

    Implementacja: Streamlit + Plotly
    Lines of code: ~15
    Complexity: Low
    """

    @staticmethod
    def streamlit(ml_data: Dict) -> None:
        """
        STREAMLIT IMPLEMENTATION
        ════════════════════════
        Approach: Bar chart with training time metrics
        """
        st.subheader("Chart 5: ML/DL Training Time")

        # Prepare data
        df = Chart5_TrainingTime.prepare_data(ml_data)

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric(
            "Fastest",
            df.loc[df["Training Time"].idxmin(), "Framework"],
            f"{df['Training Time'].min():.1f}s",
        )
        col2.metric(
            "Slowest",
            df.loc[df["Training Time"].idxmax(), "Framework"],
            f"{df['Training Time'].max():.1f}s",
        )

        # Chart
        fig = px.bar(
            df,
            x="Framework",
            y="Training Time",
            color="Framework",
            title="ML/DL Framework Training Time Comparison",
            color_discrete_sequence=Config.ML_COLORS,
        )

        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 6: INFERENCE SPEED COMPARISON (ML/DL)
# ═══════════════════════════════════════════════════════════════════════════


class Chart6_InferenceSpeed_Streamlit:
    """
    WYKRES 6: Porównanie prędkości inferencji modeli ML/DL

    Implementacja: Streamlit + Plotly
    Lines of code: ~16
    Complexity: Low
    """

    @staticmethod
    def streamlit(ml_data: Dict) -> None:
        """
        STREAMLIT IMPLEMENTATION
        ════════════════════════
        Approach: Bar chart with inference speed metrics
        Note: Higher values = better performance
        """
        st.subheader("Chart 6: ML/DL Inference Speed")

        # Prepare data
        df = Chart6_InferenceSpeed.prepare_data(ml_data)

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric(
            "Fastest",
            df.loc[df["Inference Speed"].idxmax(), "Framework"],
            f"{df['Inference Speed'].max():,.0f} samp/s",
        )
        col2.metric(
            "Average",
            "All Frameworks",
            f"{df['Inference Speed'].mean():,.0f} samp/s",
        )

        # Chart
        fig = px.bar(
            df,
            x="Framework",
            y="Inference Speed",
            color="Framework",
            title="ML/DL Framework Inference Speed Comparison",
            labels={"Inference Speed": "Inference Speed (samples/sec)"},
            color_discrete_sequence=Config.ML_COLORS,
        )

        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 7: MEMORY USAGE (ML/DL)
# ═══════════════════════════════════════════════════════════════════════════


class Chart7_MemoryUsage_ML_Streamlit:
    """
    WYKRES 7: Porównanie zużycia pamięci podczas treningu modeli ML/DL

    Implementacja: Streamlit + Plotly
    Lines of code: ~14
    Complexity: Low
    """

    @staticmethod
    def streamlit(ml_data: Dict) -> None:
        """
        STREAMLIT IMPLEMENTATION
        ════════════════════════
        Approach: Bar chart with memory usage metrics
        """
        st.subheader("Chart 7: ML/DL Memory Usage")

        # Prepare data
        df = Chart7_MemoryUsage_ML.prepare_data(ml_data)

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric(
            "Lowest",
            df.loc[df["Memory (GB)"].idxmin(), "Framework"],
            f"{df['Memory (GB)'].min():.2f} GB",
        )
        col2.metric(
            "Highest",
            df.loc[df["Memory (GB)"].idxmax(), "Framework"],
            f"{df['Memory (GB)'].max():.2f} GB",
        )

        # Chart
        fig = px.bar(
            df,
            x="Framework",
            y="Memory (GB)",
            color="Framework",
            title="ML/DL Framework Memory Usage Comparison",
            color_discrete_sequence=Config.ML_COLORS,
        )

        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD (Optional - for live demo)
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """
    Main Streamlit dashboard - combines all 7 charts
    Usage: streamlit run streamlit_implementations.py
    """
    st.set_page_config(
        page_title="Data Visualization Thesis - Streamlit Demo",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Comparative Visualization Analysis")
    st.markdown("### Master's Thesis - Streamlit Implementations")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    chart_selection = st.sidebar.radio(
        "Select Chart:",
        [
            "Overview",
            "Chart 1: Execution Time",
            "Chart 2: Operation Breakdown",
            "Chart 3: Memory Usage (DP)",
            "Chart 4: Scalability",
            "Chart 5: Training Time (ML)",
            "Chart 6: Inference Speed (ML)",
            "Chart 7: Memory Usage (ML)",
        ],
    )

    # Load data
    try:
        dp_data = DataLoader.load_data_processing()
        ml_data = DataLoader.load_ml_frameworks()

        # Display selected chart
        if chart_selection == "Overview":
            st.header("Overview")
            st.markdown(
                """
            This dashboard demonstrates Streamlit implementations of 7 visualization charts
            comparing data processing libraries and ML/DL frameworks.
            
            **Data Processing Libraries:**
            - Pandas, Polars, PyArrow, Dask, Spark
            
            **ML/DL Frameworks:**
            - Scikit-learn, PyTorch, TensorFlow, XGBoost, JAX
            
            **Select a chart from the sidebar to view the visualization.**
            """
            )

            # Summary metrics
            col1, col2 = st.columns(2)
            with col1:
                st.info("📁 **Data Processing Charts**: 4 charts (1-4)")
            with col2:
                st.info("🤖 **ML/DL Charts**: 3 charts (5-7)")

        elif chart_selection == "Chart 1: Execution Time":
            Chart1_ExecutionTime_Streamlit.streamlit(dp_data)

        elif chart_selection == "Chart 2: Operation Breakdown":
            Chart2_OperationBreakdown_Streamlit.streamlit(dp_data)

        elif chart_selection == "Chart 3: Memory Usage (DP)":
            Chart3_MemoryUsage_DP_Streamlit.streamlit(dp_data)

        elif chart_selection == "Chart 4: Scalability":
            Chart4_Scalability_Streamlit.streamlit(dp_data)

        elif chart_selection == "Chart 5: Training Time (ML)":
            Chart5_TrainingTime_Streamlit.streamlit(ml_data)

        elif chart_selection == "Chart 6: Inference Speed (ML)":
            Chart6_InferenceSpeed_Streamlit.streamlit(ml_data)

        elif chart_selection == "Chart 7: Memory Usage (ML)":
            Chart7_MemoryUsage_ML_Streamlit.streamlit(ml_data)

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info(
            "Make sure you're running this from the same directory as "
            "comparative_visualization_thesis.py and that result files exist."
        )


if __name__ == "__main__":
    main()
