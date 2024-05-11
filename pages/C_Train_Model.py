import pandas as pd                   
import streamlit as st   

if 'processed_df' in st.session_state:
    mbti_data = st.session_state.processed_df
    row_count = st.slider('Select number of rows to view', 1, len(mbti_data), 100)
    st.dataframe(mbti_data.head(row_count))