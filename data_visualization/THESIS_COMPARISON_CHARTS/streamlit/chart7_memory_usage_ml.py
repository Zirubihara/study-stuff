
def chart7_memory_usage_ml_streamlit(ml_data):
    import streamlit as st
    import plotly.express as px
    
    st.subheader("Chart 7: ML/DL Memory Usage")
    df = Chart7_MemoryUsage_ML.prepare_data(ml_data)
    
    col1, col2 = st.columns(2)
    col1.metric("Lowest", df.loc[df['Memory (GB)'].idxmin(), 'Framework'],
                f"{df['Memory (GB)'].min():.2f} GB")
    col2.metric("Highest", df.loc[df['Memory (GB)'].idxmax(), 'Framework'],
                f"{df['Memory (GB)'].max():.2f} GB")
    
    fig = px.bar(df, x='Framework', y='Memory (GB)', color='Framework',
                title='Memory Usage Comparison')
    st.plotly_chart(fig, use_container_width=True)
