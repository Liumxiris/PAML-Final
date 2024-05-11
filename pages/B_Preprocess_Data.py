import pandas as pd                   
import streamlit as st   

# Data Analysis
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import pickle as pkl
from scipy import sparse

# Text Processing
import re
import itertools
import string
import collections
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def remove_links_helper(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_links(df):
    df['posts'] = df['posts'].apply(remove_links_helper)
    return df
   
def keep_end_of_sentence_char(df):
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))
    return df

def replace_pattern_helper(posts):
    pers_types = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,'ESFJ' ,'ESTJ']
    pattern = '|'.join([re.escape(p) for p in pers_types])
    return re.sub(pattern, '', posts, flags=re.IGNORECASE)

def replace_pattern(df):
    df['posts'] = df['posts'].apply(replace_pattern_helper)
    return df


def turn_lower_case(df):
    df["posts"] = df["posts"].apply(lambda x: x.lower())
    return df

st.header('Dataset Preprocess')
if 'dataframe' in st.session_state:
    mbti_data = st.session_state.dataframe
    new_df = mbti_data
    texts = new_df['posts'].copy()
    labels = new_df['type'].copy()
    new_df = new_df.copy()


    preprocess_options = {
       'Remove Linkes in Posts': remove_links,
       'Remove Special Characters': keep_end_of_sentence_char,
       'Remove MBTI Personality Words': replace_pattern,
       'Turn Posts into Lowercase': turn_lower_case
    }

    selected_options = st.sidebar.multiselect('Choose preprocess functions:', options=list(preprocess_options.keys()))
    processed_df = new_df.copy()
    for option in selected_options:
        processed_df = preprocess_options[option](processed_df)

    # display the preprocessed dataframe
    st.subheader('Processed DataFrame')
    row_count = st.slider('Select number of rows to view', 1, len(mbti_data), 100)
    st.dataframe(processed_df.head(row_count))

    # update the dataframe in the session with preprocessed data
    st.session_state.processed_df = processed_df