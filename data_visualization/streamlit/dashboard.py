"""
Streamlit Interactive Dashboard for Thesis Visualizations
Combines Data Processing and ML/DL Framework comparisons
"""

import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Data Processing & ML/DL Framework Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ThesisDashboard:
    """Streamlit dashboard for thesis visualizations"""

    def __init__(self):
        self.data_processing_dir = Path("../../results")
        self.ml_dir = Path("../../models/results")

        self.libraries = ['pandas', 'polars', 'pyarrow', 'dask', 'spark']
        self.dataset_sizes = ['5M', '10M', '50M']
        self.operations = ['loading', 'cleaning', 'aggregation', 'sorting', 'filtering', 'correlation']

        self.frameworks = ['sklearn', 'pytorch', 'tensorflow', 'xgboost', 'jax']
        self.framework_names = {
            'sklearn': 'Scikit-learn',
            'pytorch': 'PyTorch',
            'tensorflow': 'TensorFlow',
            'xgboost': 'XGBoost',
            'jax': 'JAX'
        }

    def load_data_processing_results(self):
        """Load data processing benchmark results"""
        data = {}
        for lib in self.libraries:
            data[lib] = {}
            for size in self.dataset_sizes:
                filepath = self.data_processing_dir / f"performance_metrics_{lib}_{size}.json"
                if filepath.exists():
                    with open(filepath, 'r') as f:
                        data[lib][size] = json.load(f)
        return data

    def load_ml_results(self):
        """Load ML/DL framework results"""
        data = {}
        for fw in self.frameworks:
            filepath = self.ml_dir / f"{fw}_anomaly_detection_results.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data[fw] = json.load(f)
        return data

    def render_data_processing_section(self, data):
        """Render data processing library comparison section"""
        st.markdown('<p class="sub-header">📊 Data Processing Libraries Comparison</p>',
                   unsafe_allow_html=True)

        # Dataset size selector
        selected_size = st.selectbox(
            "Select Dataset Size",
            self.dataset_sizes,
            index=1,  # Default to 10M
            key="dp_size"
        )

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "⏱️ Performance Overview",
            "🔍 Operation Breakdown",
            "💾 Memory Usage",
            "📈 Scalability"
        ])

        with tab1:
            self.plot_dp_performance_overview(data, selected_size)

        with tab2:
            self.plot_dp_operation_breakdown(data, selected_size)

        with tab3:
            self.plot_dp_memory_usage(data, selected_size)

        with tab4:
            self.plot_dp_scalability(data)

    def plot_dp_performance_overview(self, data, dataset_size):
        """Data processing performance overview"""
        st.subheader(f"Total Execution Time - {dataset_size} Dataset")

        # Create metrics row
        cols = st.columns(len(self.libraries))

        for idx, lib in enumerate(self.libraries):
            if dataset_size in data[lib]:
                time = data[lib][dataset_size].get('total_operation_time_mean', 0)
                with cols[idx]:
                    st.metric(
                        label=lib.capitalize(),
                        value=f"{time:.2f}s",
                        delta=None
                    )

        # Bar chart
        fig = go.Figure()

        times = []
        lib_names = []

        for lib in self.libraries:
            if dataset_size in data[lib]:
                time = data[lib][dataset_size].get('total_operation_time_mean', 0)
                times.append(time)
                lib_names.append(lib.capitalize())

        fig.add_trace(go.Bar(
            x=lib_names,
            y=times,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(times)],
            text=[f'{t:.2f}s' for t in times],
            textposition='outside'
        ))

        fig.update_layout(
            title=f'Total Execution Time Comparison - {dataset_size}',
            xaxis_title='Library',
            yaxis_title='Time (seconds)',
            template='plotly_white',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_dp_operation_breakdown(self, data, dataset_size):
        """Operation breakdown visualization"""
        st.subheader(f"Operation Time Breakdown - {dataset_size} Dataset")

        # Stacked bar chart
        fig = go.Figure()

        for op in self.operations:
            times = []
            for lib in self.libraries:
                if dataset_size in data[lib]:
                    key = f"{op}_time_mean"
                    times.append(data[lib][dataset_size].get(key, 0))
                else:
                    times.append(0)

            fig.add_trace(go.Bar(
                name=op.capitalize(),
                x=[lib.capitalize() for lib in self.libraries],
                y=times
            ))

        fig.update_layout(
            title=f'Operation Breakdown - {dataset_size}',
            xaxis_title='Library',
            yaxis_title='Time (seconds)',
            barmode='stack',
            template='plotly_white',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Create comparison table
        st.subheader("Detailed Operation Times")

        table_data = []
        for lib in self.libraries:
            if dataset_size in data[lib]:
                row = {'Library': lib.capitalize()}
                for op in self.operations:
                    key = f"{op}_time_mean"
                    row[op.capitalize()] = f"{data[lib][dataset_size].get(key, 0):.2f}s"
                table_data.append(row)

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def plot_dp_memory_usage(self, data, dataset_size):
        """Memory usage visualization"""
        st.subheader(f"Memory Usage - {dataset_size} Dataset")

        # Create metrics
        cols = st.columns(len(self.libraries))

        for idx, lib in enumerate(self.libraries):
            if dataset_size in data[lib]:
                load_mem = data[lib][dataset_size].get('loading_memory_mean', 0)
                clean_mem = data[lib][dataset_size].get('cleaning_memory_mean', 0)
                total_mem = (load_mem + clean_mem) / 1024  # GB

                with cols[idx]:
                    st.metric(
                        label=lib.capitalize(),
                        value=f"{total_mem:.2f} GB"
                    )

        # Bar chart
        fig = go.Figure()

        lib_names = []
        memory_values = []

        for lib in self.libraries:
            if dataset_size in data[lib]:
                load_mem = data[lib][dataset_size].get('loading_memory_mean', 0)
                clean_mem = data[lib][dataset_size].get('cleaning_memory_mean', 0)
                total_mem = (load_mem + clean_mem) / 1024

                lib_names.append(lib.capitalize())
                memory_values.append(total_mem)

        fig.add_trace(go.Bar(
            x=lib_names,
            y=memory_values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(memory_values)],
            text=[f'{m:.2f} GB' for m in memory_values],
            textposition='outside'
        ))

        fig.update_layout(
            title='Memory Usage Comparison',
            xaxis_title='Library',
            yaxis_title='Memory (GB)',
            template='plotly_white',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_dp_scalability(self, data):
        """Scalability analysis"""
        st.subheader("Scalability Analysis")

        fig = go.Figure()

        sizes_numeric = [5, 10, 50]

        for lib in self.libraries:
            times = []
            for size in self.dataset_sizes:
                if size in data[lib]:
                    times.append(data[lib][size].get('total_operation_time_mean', 0))
                else:
                    times.append(None)

            valid_sizes = [s for s, t in zip(sizes_numeric, times) if t is not None and t > 0]
            valid_times = [t for t in times if t is not None and t > 0]

            if valid_times:
                fig.add_trace(go.Scatter(
                    name=lib.capitalize(),
                    x=valid_sizes,
                    y=valid_times,
                    mode='lines+markers',
                    marker=dict(size=10),
                    line=dict(width=2)
                ))

        fig.update_layout(
            title='Performance Scaling with Dataset Size',
            xaxis_title='Dataset Size (Million Rows)',
            yaxis_title='Execution Time (seconds)',
            xaxis_type='log',
            yaxis_type='log',
            template='plotly_white',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_ml_section(self, data):
        """Render ML/DL framework comparison section"""
        st.markdown('<p class="sub-header">🤖 ML/DL Frameworks Comparison</p>',
                   unsafe_allow_html=True)

        if not data:
            st.warning("No ML/DL results found!")
            return

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡ Performance Metrics",
            "📊 Framework Comparison",
            "🎯 Trade-off Analysis",
            "📈 Rankings"
        ])

        with tab1:
            self.plot_ml_performance_metrics(data)

        with tab2:
            self.plot_ml_framework_comparison(data)

        with tab3:
            self.plot_ml_tradeoff_analysis(data)

        with tab4:
            self.plot_ml_rankings(data)

    def plot_ml_performance_metrics(self, data):
        """ML performance metrics overview"""
        st.subheader("Performance Metrics Overview")

        # Create metric cards
        metrics = ['Training Time', 'Inference Speed', 'Memory Usage', 'Anomaly Rate']

        for metric in metrics:
            st.markdown(f"**{metric}**")
            cols = st.columns(len(self.frameworks))

            for idx, fw in enumerate(self.frameworks):
                if fw not in data:
                    continue

                if fw == 'sklearn':
                    model_data = data[fw].get('isolation_forest', {})
                elif fw == 'pytorch':
                    model_data = data[fw].get('pytorch_autoencoder', {})
                elif fw == 'tensorflow':
                    model_data = data[fw].get('tensorflow_autoencoder', {})
                elif fw == 'jax':
                    model_data = data[fw].get('jax_autoencoder', {})
                elif fw == 'xgboost':
                    model_data = data[fw].get('xgboost_detector', {})
                else:
                    continue

                with cols[idx]:
                    if metric == 'Training Time':
                        val = model_data.get('training_time', 0)
                        st.metric(self.framework_names[fw], f"{val:.2f}s")
                    elif metric == 'Inference Speed':
                        val = model_data.get('inference_speed', 0)
                        st.metric(self.framework_names[fw], f"{val:,.0f} s/s")
                    elif metric == 'Memory Usage':
                        val = model_data.get('memory_usage_gb', 0)
                        st.metric(self.framework_names[fw], f"{val:.2f} GB")
                    elif metric == 'Anomaly Rate':
                        val = model_data.get('anomaly_rate', 0)
                        st.metric(self.framework_names[fw], f"{val:.2f}%")

    def plot_ml_framework_comparison(self, data):
        """Framework comparison charts"""
        st.subheader("Multi-Metric Comparison")

        # Create 2x2 subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Time', 'Inference Speed', 'Memory Usage', 'Anomaly Rate')
        )

        metrics = [
            ('training_time', 1, 1),
            ('inference_speed', 1, 2),
            ('memory_usage_gb', 2, 1),
            ('anomaly_rate', 2, 2)
        ]

        for metric, row, col in metrics:
            frameworks = []
            values = []

            for fw in self.frameworks:
                if fw not in data:
                    continue

                if fw == 'sklearn':
                    model_data = data[fw].get('isolation_forest', {})
                elif fw == 'pytorch':
                    model_data = data[fw].get('pytorch_autoencoder', {})
                elif fw == 'tensorflow':
                    model_data = data[fw].get('tensorflow_autoencoder', {})
                elif fw == 'jax':
                    model_data = data[fw].get('jax_autoencoder', {})
                elif fw == 'xgboost':
                    model_data = data[fw].get('xgboost_detector', {})
                else:
                    continue

                val = model_data.get(metric, 0)
                if val != 0 or metric == 'memory_usage_gb':
                    frameworks.append(self.framework_names[fw])
                    values.append(val)

            fig.add_trace(
                go.Bar(x=frameworks, y=values, showlegend=False),
                row=row, col=col
            )

        fig.update_layout(height=700, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    def plot_ml_tradeoff_analysis(self, data):
        """Training vs inference trade-off"""
        st.subheader("Training Time vs Inference Speed Trade-off")

        fig = go.Figure()

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == 'sklearn':
                model_data = data[fw].get('isolation_forest', {})
            elif fw in ['pytorch', 'tensorflow', 'jax']:
                model_data = data[fw].get('autoencoder', {})
            elif fw == 'xgboost':
                model_data = data[fw].get('xgboost_detector', {})
            else:
                continue

            train_time = model_data.get('training_time', 0)
            infer_speed = model_data.get('inference_speed', 0)

            if train_time > 0 and infer_speed > 0:
                fig.add_trace(go.Scatter(
                    x=[train_time],
                    y=[infer_speed],
                    mode='markers+text',
                    name=self.framework_names[fw],
                    text=[self.framework_names[fw]],
                    textposition="top center",
                    marker=dict(size=15)
                ))

        fig.update_layout(
            title='Training Time vs Inference Speed',
            xaxis_title='Training Time (seconds) - Lower is Better',
            yaxis_title='Inference Speed (samples/s) - Higher is Better',
            template='plotly_white',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_ml_rankings(self, data):
        """Framework rankings"""
        st.subheader("Framework Rankings")

        # Create ranking table
        rankings = []

        for fw in self.frameworks:
            if fw not in data:
                continue

            if fw == 'sklearn':
                model_data = data[fw].get('isolation_forest', {})
            elif fw in ['pytorch', 'tensorflow', 'jax']:
                model_data = data[fw].get('autoencoder', {})
            elif fw == 'xgboost':
                model_data = data[fw].get('xgboost_detector', {})
            else:
                continue

            rankings.append({
                'Framework': self.framework_names[fw],
                'Training Time': f"{model_data.get('training_time', 0):.2f}s",
                'Inference Speed': f"{model_data.get('inference_speed', 0):,.0f}",
                'Memory Usage': f"{model_data.get('memory_usage_gb', 0):.2f} GB",
                'Anomaly Rate': f"{model_data.get('anomaly_rate', 0):.2f}%"
            })

        df = pd.DataFrame(rankings)
        st.dataframe(df, use_container_width=True)

    def run(self):
        """Run the dashboard"""
        # Header
        st.markdown('<p class="main-header">📊 Data Processing & ML/DL Framework Comparison Dashboard</p>',
                   unsafe_allow_html=True)

        st.markdown("---")

        # Sidebar
        with st.sidebar:
            st.title("Navigation")
            section = st.radio(
                "Select Section",
                ["🏠 Home", "📊 Data Processing", "🤖 ML/DL Frameworks", "📈 Combined Analysis"],
                index=0
            )

            st.markdown("---")
            st.markdown("### About")
            st.info("""
            This dashboard provides comprehensive visualizations for:
            - Data processing library comparisons (Pandas, Polars, PyArrow, Dask, Spark)
            - ML/DL framework comparisons (Scikit-learn, PyTorch, TensorFlow, XGBoost, JAX)
            """)

        # Main content based on section
        if section == "🏠 Home":
            st.header("Welcome to the Thesis Visualization Dashboard")
            st.write("""
            ### Overview
            This interactive dashboard presents comprehensive performance comparisons for:

            #### 📊 Data Processing Libraries
            - **Pandas**: Traditional Python data analysis
            - **Polars**: Fast DataFrame library in Rust
            - **PyArrow**: Apache Arrow for Python
            - **Dask**: Parallel computing library
            - **PySpark**: Apache Spark for Python

            #### 🤖 ML/DL Frameworks
            - **Scikit-learn**: Traditional ML (Isolation Forest, LOF)
            - **PyTorch**: Deep learning framework
            - **TensorFlow**: Google's ML platform
            - **XGBoost**: Gradient boosting framework
            - **JAX**: High-performance numerical computing

            ### Features
            - Interactive visualizations with Plotly
            - Real-time metric comparisons
            - Scalability analysis
            - Trade-off analysis
            """)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Processing Libraries", "5")
                st.metric("Dataset Sizes Tested", "3 (5M, 10M, 50M)")
            with col2:
                st.metric("ML/DL Frameworks", "5")
                st.metric("Operations Benchmarked", "6")

        elif section == "📊 Data Processing":
            data = self.load_data_processing_results()
            self.render_data_processing_section(data)

        elif section == "🤖 ML/DL Frameworks":
            data = self.load_ml_results()
            self.render_ml_section(data)

        elif section == "📈 Combined Analysis":
            st.header("Combined Analysis")
            st.info("This section provides combined insights from both data processing and ML/DL comparisons.")

            # Load both datasets
            dp_data = self.load_data_processing_results()
            ml_data = self.load_ml_results()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Best Performers - Data Processing")
                st.write("**Fastest (10M dataset):**")
                # Find fastest library
                fastest_lib = None
                fastest_time = float('inf')
                for lib in self.libraries:
                    if '10M' in dp_data[lib]:
                        time = dp_data[lib]['10M'].get('total_operation_time_mean', float('inf'))
                        if time < fastest_time:
                            fastest_time = time
                            fastest_lib = lib
                if fastest_lib:
                    st.success(f"**{fastest_lib.capitalize()}**: {fastest_time:.2f}s")

            with col2:
                st.subheader("Best Performers - ML/DL")
                st.write("**Fastest Training:**")
                # Find fastest framework
                fastest_fw = None
                fastest_time = float('inf')
                for fw in self.frameworks:
                    if fw in ml_data:
                        if fw == 'sklearn':
                            time = ml_data[fw].get('isolation_forest', {}).get('training_time', float('inf'))
                        elif fw == 'pytorch':
                            time = ml_data[fw].get('pytorch_autoencoder', {}).get('training_time', float('inf'))
                        elif fw == 'tensorflow':
                            time = ml_data[fw].get('tensorflow_autoencoder', {}).get('training_time', float('inf'))
                        elif fw == 'jax':
                            time = ml_data[fw].get('jax_autoencoder', {}).get('training_time', float('inf'))
                        elif fw == 'xgboost':
                            time = ml_data[fw].get('xgboost_detector', {}).get('training_time', float('inf'))
                        else:
                            continue

                        if time < fastest_time:
                            fastest_time = time
                            fastest_fw = fw
                if fastest_fw:
                    st.success(f"**{self.framework_names[fastest_fw]}**: {fastest_time:.2f}s")


if __name__ == "__main__":
    dashboard = ThesisDashboard()
    dashboard.run()
