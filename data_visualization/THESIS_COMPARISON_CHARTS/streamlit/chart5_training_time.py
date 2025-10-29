
def chart5_training_time_streamlit(ml_data):
    import streamlit as st
    import plotly.express as px
    
    st.subheader("Chart 5: ML/DL Training Time")
    df = Chart5_TrainingTime.prepare_data(ml_data)
    
    col1, col2 = st.columns(2)
    col1.metric("Fastest", df.loc[df['Training Time'].idxmin(), 'Framework'],
                f"{df['Training Time'].min():.1f}s")
    col2.metric("Slowest", df.loc[df['Training Time'].idxmax(), 'Framework'],
                f"{df['Training Time'].max():.1f}s")
    
    fig = px.bar(df, x='Framework', y='Training Time', color='Framework',
                title='Training Time Comparison')
    st.plotly_chart(fig, use_container_width=True)
