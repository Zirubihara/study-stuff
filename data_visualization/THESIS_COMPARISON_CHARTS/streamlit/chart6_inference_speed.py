
def chart6_inference_speed_streamlit(ml_data):
    import streamlit as st
    import plotly.express as px
    
    st.subheader("Chart 6: ML/DL Inference Speed")
    df = Chart6_InferenceSpeed.prepare_data(ml_data)
    
    col1, col2 = st.columns(2)
    col1.metric("Fastest", df.loc[df['Inference Speed'].idxmax(), 'Framework'],
                f"{df['Inference Speed'].max():,.0f} samp/s")
    col2.metric("Average", "All Frameworks",
                f"{df['Inference Speed'].mean():,.0f} samp/s")
    
    fig = px.bar(df, x='Framework', y='Inference Speed', color='Framework',
                title='Inference Speed Comparison')
    st.plotly_chart(fig, use_container_width=True)
