"""
╔════════════════════════════════════════════════════════════════════════════╗
║                     STREAMLIT INTERACTIVE DASHBOARD                         ║
║                   Master's Thesis - Visualization Comparison                ║
║                                                                              ║
║  Uruchom: streamlit run streamlit_dashboard_app.py                         ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Visualization Library Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colors
DP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
ML_COLORS = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff9896"]

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_data
def load_data_processing():
    """Load Data Processing benchmark results"""
    data = {}
    results_dir = Path("../../results")
    libraries = ["pandas", "polars", "pyarrow", "dask", "spark"]

    for lib in libraries:
        filename = f"performance_metrics_{lib}_10M.json"
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                data[lib] = json.load(f)

    return data


@st.cache_data
def load_ml_frameworks():
    """Load ML/DL framework results"""
    data = {}
    results_dir = Path("../../models/results")
    frameworks = ["sklearn", "pytorch", "tensorflow", "xgboost", "jax"]

    for fw in frameworks:
        filepath = results_dir / f"{fw}_anomaly_detection_results.json"
        if filepath.exists():
            with open(filepath, "r") as f:
                data[fw] = json.load(f)

    return data


# ═══════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def chart1_execution_time(dp_data):
    """Chart 1: Execution Time Comparison"""
    chart_data = []
    lib_names = {
        "pandas": "Pandas",
        "polars": "Polars",
        "pyarrow": "PyArrow",
        "dask": "Dask",
        "spark": "Spark",
    }

    for lib, data in dp_data.items():
        time = data.get("total_operation_time_mean", 0)
        chart_data.append({"Library": lib_names[lib], "Time": time})

    df = pd.DataFrame(chart_data)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        fastest_idx = df["Time"].idxmin()
        st.metric(
            "🏆 Fastest",
            df.loc[fastest_idx, "Library"],
            f"{df.loc[fastest_idx, 'Time']:.2f}s",
        )
    with col2:
        st.metric("📊 Average", "All Libraries", f"{df['Time'].mean():.2f}s")
    with col3:
        slowest_idx = df["Time"].idxmax()
        st.metric(
            "🐌 Slowest",
            df.loc[slowest_idx, "Library"],
            f"{df.loc[slowest_idx, 'Time']:.2f}s",
        )

    # Chart
    fig = px.bar(
        df,
        x="Library",
        y="Time",
        color="Library",
        title="Data Processing Performance - 10M Dataset",
        labels={"Time": "Total Execution Time (seconds)"},
        color_discrete_sequence=DP_COLORS,
    )
    fig.update_layout(showlegend=False, height=500)

    return fig


def chart2_operation_breakdown(dp_data):
    """Chart 2: Operation Breakdown"""
    operations = [
        "loading",
        "cleaning",
        "aggregation",
        "sorting",
        "filtering",
        "correlation",
    ]
    chart_data = []
    lib_names = {
        "pandas": "Pandas",
        "polars": "Polars",
        "pyarrow": "PyArrow",
        "dask": "Dask",
        "spark": "Spark",
    }

    for lib, data in dp_data.items():
        for op in operations:
            key = f"{op}_time_mean"
            time = data.get(key, 0)
            chart_data.append(
                {"Library": lib_names[lib], "Operation": op.capitalize(), "Time": time}
            )

    df = pd.DataFrame(chart_data)

    fig = px.bar(
        df,
        x="Operation",
        y="Time",
        color="Library",
        title="Operation Breakdown - 10M Dataset",
        barmode="group",
        color_discrete_sequence=DP_COLORS,
    )
    fig.update_layout(height=500)

    return fig


def chart3_memory_usage_dp(dp_data):
    """Chart 3: Memory Usage (Data Processing)"""
    chart_data = []
    lib_names = {
        "pandas": "Pandas",
        "polars": "Polars",
        "pyarrow": "PyArrow",
        "dask": "Dask",
        "spark": "Spark",
    }

    for lib, data in dp_data.items():
        load_mem = data.get("loading_memory_mean", 0)
        clean_mem = data.get("cleaning_memory_mean", 0)
        total_mem = (load_mem + clean_mem) / 1024  # MB to GB
        chart_data.append({"Library": lib_names[lib], "Memory (GB)": total_mem})

    df = pd.DataFrame(chart_data)

    fig = px.bar(
        df,
        x="Library",
        y="Memory (GB)",
        color="Library",
        title="Memory Usage Comparison - 10M Dataset",
        color_discrete_sequence=DP_COLORS,
    )
    fig.update_layout(showlegend=False, height=500)

    return fig


def chart5_training_time(ml_data):
    """Chart 5: ML/DL Training Time"""
    chart_data = []
    fw_names = {
        "sklearn": "Scikit-learn",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "xgboost": "XGBoost",
        "jax": "JAX",
    }

    for fw, data in ml_data.items():
        if fw == "sklearn":
            time = data.get("isolation_forest", {}).get("training_time", 0)
        elif fw in ["pytorch", "tensorflow", "jax"]:
            time = data.get(f"{fw}_autoencoder", {}).get("training_time", 0)
        elif fw == "xgboost":
            time = data.get("xgboost_detector", {}).get("training_time", 0)
        else:
            time = 0

        if time > 0:
            chart_data.append({"Framework": fw_names[fw], "Training Time": time})

    df = pd.DataFrame(chart_data)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        fastest_idx = df["Training Time"].idxmin()
        st.metric(
            "⚡ Fastest Training",
            df.loc[fastest_idx, "Framework"],
            f"{df.loc[fastest_idx, 'Training Time']:.1f}s",
        )
    with col2:
        st.metric("📊 Average", "All Frameworks", f"{df['Training Time'].mean():.1f}s")
    with col3:
        slowest_idx = df["Training Time"].idxmax()
        st.metric(
            "🐌 Slowest Training",
            df.loc[slowest_idx, "Framework"],
            f"{df.loc[slowest_idx, 'Training Time']:.1f}s",
        )

    fig = px.bar(
        df,
        x="Framework",
        y="Training Time",
        color="Framework",
        title="ML/DL Framework Training Time Comparison",
        color_discrete_sequence=ML_COLORS,
    )
    fig.update_layout(showlegend=False, height=500)

    return fig


def chart6_inference_speed(ml_data):
    """Chart 6: ML/DL Inference Speed"""
    chart_data = []
    fw_names = {
        "sklearn": "Scikit-learn",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "xgboost": "XGBoost",
        "jax": "JAX",
    }

    for fw, data in ml_data.items():
        if fw == "sklearn":
            speed = data.get("isolation_forest", {}).get("inference_speed", 0)
        elif fw in ["pytorch", "tensorflow", "jax"]:
            speed = data.get(f"{fw}_autoencoder", {}).get("inference_speed", 0)
        elif fw == "xgboost":
            speed = data.get("xgboost_detector", {}).get("inference_speed", 0)
        else:
            speed = 0

        if speed > 0:
            chart_data.append({"Framework": fw_names[fw], "Inference Speed": speed})

    df = pd.DataFrame(chart_data)

    fig = px.bar(
        df,
        x="Framework",
        y="Inference Speed",
        color="Framework",
        title="ML/DL Framework Inference Speed Comparison",
        labels={"Inference Speed": "Inference Speed (samples/sec)"},
        color_discrete_sequence=ML_COLORS,
    )
    fig.update_layout(showlegend=False, height=500)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════


def main():
    # Header
    st.title("📊 Data Visualization Library Comparison")
    st.markdown("### Master's Thesis - Interactive Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Analysis:",
        ["🏠 Home", "📊 Data Processing", "🤖 ML/DL Frameworks", "📈 Comparison"],
    )

    # Load data
    try:
        dp_data = load_data_processing()
        ml_data = load_ml_frameworks()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info(
            "Make sure you're running this from THESIS_COMPARISON_CHARTS/ directory"
        )
        return

    # ═══════════════════════════════════════════════════════════════════════
    # HOME PAGE
    # ═══════════════════════════════════════════════════════════════════════

    if page == "🏠 Home":
        st.header("Welcome to the Interactive Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Data Processing Analysis")
            st.write(
                """
            Compare 5 data processing libraries:
            - **Pandas** - Traditional DataFrame library
            - **Polars** - Modern, fast DataFrame library
            - **PyArrow** - Apache Arrow in Python
            - **Dask** - Distributed computing
            - **Spark** - Big data processing
            
            **Dataset:** 10 million rows
            """
            )

        with col2:
            st.subheader("🤖 ML/DL Framework Analysis")
            st.write(
                """
            Compare 5 ML/DL frameworks:
            - **Scikit-learn** - Traditional ML
            - **PyTorch** - Deep Learning
            - **TensorFlow** - Deep Learning
            - **XGBoost** - Gradient Boosting
            - **JAX** - High-performance ML
            
            **Task:** Anomaly Detection
            """
            )

        st.markdown("---")
        st.info("👈 Use the sidebar to navigate between different analyses")

        # Quick stats
        st.subheader("📈 Quick Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Libraries Compared", "5+5")
        with col2:
            st.metric("Charts Generated", "35")
        with col3:
            st.metric("Dataset Size", "10M rows")
        with col4:
            st.metric("Frameworks", "10 total")

    # ═══════════════════════════════════════════════════════════════════════
    # DATA PROCESSING PAGE
    # ═══════════════════════════════════════════════════════════════════════

    elif page == "📊 Data Processing":
        st.header("Data Processing Performance Analysis")

        tab1, tab2, tab3 = st.tabs(
            ["⚡ Execution Time", "🔧 Operation Breakdown", "💾 Memory Usage"]
        )

        with tab1:
            st.subheader("Total Execution Time Comparison")
            st.plotly_chart(chart1_execution_time(dp_data), use_container_width=True)

            with st.expander("📝 Analysis"):
                st.write(
                    """
                **Key Findings:**
                - Polars is the fastest library for 10M row operations
                - Spark has significant overhead for this dataset size
                - PyArrow shows good performance for columnar operations
                """
                )

        with tab2:
            st.subheader("Operation-by-Operation Breakdown")
            st.plotly_chart(
                chart2_operation_breakdown(dp_data), use_container_width=True
            )

            with st.expander("📝 Analysis"):
                st.write(
                    """
                **Key Findings:**
                - Loading time varies significantly between libraries
                - Cleaning operations are fastest in Polars
                - Correlation is the most expensive operation for most libraries
                """
                )

        with tab3:
            st.subheader("Memory Usage Comparison")
            st.plotly_chart(chart3_memory_usage_dp(dp_data), use_container_width=True)

            with st.expander("📝 Analysis"):
                st.write(
                    """
                **Key Findings:**
                - Polars and PyArrow are most memory-efficient
                - Spark has higher memory footprint due to distributed architecture
                - Pandas uses moderate memory for this dataset size
                """
                )

    # ═══════════════════════════════════════════════════════════════════════
    # ML/DL FRAMEWORKS PAGE
    # ═══════════════════════════════════════════════════════════════════════

    elif page == "🤖 ML/DL Frameworks":
        st.header("ML/DL Framework Performance Analysis")

        tab1, tab2 = st.tabs(["⏱️ Training Time", "⚡ Inference Speed"])

        with tab1:
            st.subheader("Training Time Comparison")
            st.plotly_chart(chart5_training_time(ml_data), use_container_width=True)

            with st.expander("📝 Analysis"):
                st.write(
                    """
                **Key Findings:**
                - XGBoost has fastest training time for this task
                - PyTorch autoencoders require significant training time
                - Scikit-learn Isolation Forest is very efficient
                """
                )

        with tab2:
            st.subheader("Inference Speed Comparison")
            st.plotly_chart(chart6_inference_speed(ml_data), use_container_width=True)

            with st.expander("📝 Analysis"):
                st.write(
                    """
                **Key Findings:**
                - Inference speed varies significantly between frameworks
                - XGBoost shows excellent inference performance
                - Deep learning frameworks have different optimization trade-offs
                """
                )

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON PAGE
    # ═══════════════════════════════════════════════════════════════════════

    elif page == "📈 Comparison":
        st.header("Cross-Library Comparison")

        st.subheader("📊 Visualization Library Comparison")

        # Library comparison table
        comparison_data = {
            "Library": ["Bokeh", "Holoviews", "Matplotlib", "Plotly", "Streamlit"],
            "Avg LOC": [25, 13, 20, 9, 16],
            "Format": ["HTML", "HTML", "PNG", "HTML", "Web App"],
            "Interactive": ["Yes", "Yes", "No", "Yes", "Yes"],
            "Best For": [
                "Fine control",
                "Clean code",
                "Publications",
                "Speed",
                "Dashboards",
            ],
        }

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)

        # LOC Comparison Chart
        st.subheader("Lines of Code Comparison")
        fig_loc = px.bar(
            df_comparison,
            x="Library",
            y="Avg LOC",
            color="Library",
            title="Average Lines of Code per Chart",
            labels={"Avg LOC": "Average Lines of Code"},
        )
        st.plotly_chart(fig_loc, use_container_width=True)

        st.markdown("---")

        # Key Insights
        st.subheader("🔍 Key Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **Simplicity Rankings:**
            1. 🥇 Plotly (9 LOC avg)
            2. 🥈 Holoviews (13 LOC)
            3. 🥉 Streamlit (16 LOC)
            4. Matplotlib (20 LOC)
            5. Bokeh (25 LOC)
            """
            )

        with col2:
            st.markdown(
                """
            **Best Use Cases:**
            - **Plotly:** Fast prototyping
            - **Matplotlib:** Academic publications
            - **Streamlit:** Interactive dashboards
            - **Holoviews:** Clean, declarative code
            - **Bokeh:** Maximum control
            """
            )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center'>
        <p>📊 Master's Thesis - Data Visualization Library Comparison</p>
        <p>Generated from: comparative_visualization_thesis.py</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()



