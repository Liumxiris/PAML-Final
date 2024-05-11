import pandas as pd                   
import streamlit as st   

# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import wordcloud
from wordcloud import WordCloud, STOPWORDS

def load_dataset(filepath):
    data=pd.read_csv(filepath)
    return data

def display_data(df):
    row_count = st.slider('Select number of rows to view', 1, len(df), 100)
    st.dataframe(df.head(row_count))

def extract(posts, new_posts):
    for post in posts[1].split("|||"):
        new_posts.append((posts[0], post))

df = None
# uploaded_file = st.file_uploader('Upload the dataset', type=".csv")
uploaded_file = load_dataset('./sample_data/mbti_1.csv')
#   store the file into the session if nothing exists OR upload a different file
if 'dataframe' not in st.session_state:
    dataframe = uploaded_file
    st.session_state.dataframe = dataframe

if 'dataframe' in st.session_state:
    mbti_data = st.session_state.dataframe

    #### Overview of Dataset ######
    st.header('Dataset Overview')
    st.subheader('Data Display')
    # Checking for null data
    mbti_data.isnull().any()
    # Prints dataset info
    nRow, nCol = mbti_data.shape
    st.write(f"There are {nRow} rows and {nCol} columns.")
    ## Display the entire dataset
    display_data(mbti_data)

    # getting all MBTI types
    types = np.unique(np.array(mbti_data['type']))
    # count no. of posts for each MBTI
    total = mbti_data['type'].value_counts()

    fig = px.pie(mbti_data,names='type',title='Personality type percentile',hole=0.3)
    st.plotly_chart(fig)

    #Plotting number of posts in descending order
    st.markdown("**No. of posts by different MBTI type**")
    cnt_srs = mbti_data['type'].value_counts()
    plt.figure(figsize=(12,4))
    sns.barplot(x = cnt_srs.index, y = cnt_srs.values, alpha = 0.8)
    plt.xlabel('Personality types', fontsize=12)
    plt.ylabel('No. of posts availables', fontsize=12)
    st.pyplot(plt)

    st.markdown("**Distribution of Lengths of all 50 Posts**")
    df = mbti_data.copy()
    df["length_posts"] = df["posts"].apply(len)
    fig, ax = plt.subplots()
    sns.distplot(df["length_posts"], ax=ax)
    ax.set_title("Distribution of Lengths of all 50 Posts")
    st.pyplot(fig)

    #Finding the most common words in all posts.
    words = list(df["posts"].apply(lambda x: x.split()))
    words = [x for y in words for x in y]

    st.markdown("**Most Popular Words in Each MBTI's posts**")
    # plotting the most popular words in each MBTI's posts
    fig, ax = plt.subplots(len(df['type'].unique()), sharex=True, figsize=(15,len(df['type'].unique())))
    k = 0
    for i in df['type'].unique():
        df_4 = df[df['type'] == i]
        wordcloud = WordCloud(max_words=1628,relative_scaling=1,normalize_plurals=False).generate(df_4['posts'].to_string())
        plt.subplot(4,4,k+1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(i)
        ax[k].axis("off")
        k+=1
    st.pyplot(fig)

    posts = []
    df.apply(lambda x: extract(x, posts), axis=1)
    st.write(f"There are {len(df)} users and {len(posts)} posts.")
