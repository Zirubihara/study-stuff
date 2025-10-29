
def chart1_execution_time_streamlit(dp_data):
    """Streamlit implementation - requires 'streamlit run'"""
    import streamlit as st
    import plotly.express as px
    
    st.subheader("Chart 1: Execution Time Comparison")
    
    # Prepare data
    df = Chart1_ExecutionTime.prepare_data(dp_data)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Fastest", df.loc[df['Time'].idxmin(), 'Library'], 
                f"{df['Time'].min():.2f}s")
    col2.metric("Average", "All Libraries", f"{df['Time'].mean():.2f}s")
    col3.metric("Slowest", df.loc[df['Time'].idxmax(), 'Library'],
                f"{df['Time'].max():.2f}s")
    
    # Chart
    fig = px.bar(df, x='Library', y='Time', color='Library',
                title='Data Processing Performance - 10M Dataset')
    st.plotly_chart(fig, use_container_width=True)
