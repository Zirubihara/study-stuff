
def chart4_scalability_streamlit(dp_data):
    import streamlit as st
    import plotly.express as px
    
    st.subheader("Chart 4: Scalability Analysis")
    df = Chart4_Scalability.prepare_data(dp_data)
    
    selected_libs = st.multiselect(
        "Select libraries to display:",
        options=df['Library'].unique().tolist(),
        default=df['Library'].unique().tolist()
    )
    
    filtered_df = df[df['Library'].isin(selected_libs)]
    
    fig = px.line(filtered_df, x='Dataset Size (M)', y='Time',
                 color='Library', markers=True,
                 title='Scalability Analysis')
    st.plotly_chart(fig, use_container_width=True)
