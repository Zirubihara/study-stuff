
def chart3_memory_usage_dp_streamlit(dp_data):
    import streamlit as st
    import plotly.express as px
    
    st.subheader("Chart 3: Memory Usage (Data Processing)")
    df = Chart3_MemoryUsage_DP.prepare_data(dp_data)
    
    fig = px.bar(df, x='Library', y='Memory (GB)', color='Library',
                title='Memory Usage Comparison')
    st.plotly_chart(fig, use_container_width=True)
