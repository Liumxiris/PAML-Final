import pandas as pd                   
import streamlit as st   

if 'dataframe' in st.session_state:
    mbti_data = st.session_state.dataframe
    row_count = st.slider('Select number of rows to view', 1, len(mbti_data), 100)
    st.dataframe(mbti_data.head(row_count))