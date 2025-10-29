
def chart2_operation_breakdown_streamlit(dp_data):
    import streamlit as st
    import plotly.express as px
    
    st.subheader("Chart 2: Operation Breakdown")
    df = Chart2_OperationBreakdown.prepare_data(dp_data)
    
    fig = px.bar(df, x='Operation', y='Time', color='Library',
                barmode='group', title='Operation Breakdown')
    st.plotly_chart(fig, use_container_width=True)
