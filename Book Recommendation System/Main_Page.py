import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re

st.set_page_config(layout='wide')



base_path = os.path.dirname(__file__)

csv1_path = os.path.join(base_path, "Audible_Catlog.csv")
csv1 = pd.read_csv(csv1_path)
csv1 = csv1.drop_duplicates()

csv2_path = os.path.join(base_path, "Audible_Catlog_Advanced_Features.csv")
csv2 = pd.read_csv(csv2_path)
csv2 = csv2.drop_duplicates()

df = csv1.merge(csv2, how="inner", on="Book Name")
df = df.drop(['Author_y','Rating_y', 'Price_x'], axis=1)
df['Number of Reviews'] = df[['Number of Reviews_x', 'Number of Reviews_y']].max(axis=1)
df = df.drop(['Number of Reviews_x', 'Number of Reviews_y'], axis=1)
df = df.dropna(subset=['Description'])
df['Number of Reviews'] = df['Number of Reviews'].fillna(0).astype(int)
df = df.copy()
df['Ranks and Genre'] = df['Ranks and Genre'].str.lstrip(',')
df = df.copy()
df['Rating_x'] = df['Rating_x'].replace(-1.0, np.nan)
df['Listening Time'] = df['Listening Time'].replace("-1", "None")
df['Ranks and Genre'] = df['Ranks and Genre'].replace("-1", "None")
df = df.dropna(subset=['Rating_x'])
df = df[df["Listening Time"] != "None"]
df = df[df["Ranks and Genre"] != "None"]
df['Popularity'] = df['Rating_x'] * df['Number of Reviews']
df.columns = ['Book Name', 'Author Name', 'Rating', 'Price', 'Description', 'Listening Time', 'Ranks and Genre', 'Number of Reviews', 'Popularity']

with st.sidebar:
    page = st.radio("Go to", ['Home', 'EDA', 'NLP'])

if page == "Home":
    st.title("You are in Home Page")
    st.dataframe(df)

elif page == "EDA":

    st.subheader("Exploratory Data Analysis (EDA)")
    st.divider()

    col1, col2, col3 = st.columns([1,1,2])

    col1.metric("Total Books", len(df))
    col2.metric("Avg Rating", round(df['Rating'].mean(),2))
    col3.metric("Top Author", df['Author Name'].value_counts().idxmax())
    st.divider()

    optionselected = st.selectbox("Select one from Below", ['None', 'Ratings Distribution', 
                                             'Top 10 Popular Books',
                                             'Top 5 Authors',
                                             'Books with Highest Reviews',
                                             'Distribution of Listening Time',
                                             'Top 5 Most Listened Books',
                                             'Top Rated Books'])
    if optionselected == "Ratings Distribution":
        min_Rating = df['Rating'].min()
        max_Rating = df['Rating'].max()
        st.write("")
        rating_Selected = st.slider("Select a Rating",min_value=min_Rating, max_value=max_Rating, value=max_Rating, step=0.1)
        result_df = df[df['Rating']==rating_Selected]
        st.dataframe(result_df[['Book Name', 'Author Name', 'Rating', 'Description']], hide_index=True)
        st.divider()
    
    elif optionselected == "Top 10 Popular Books":
        result_df = df.sort_values(by='Popularity', ascending=False)
        st.divider()
        st.subheader("Top Popular Books based on Reviews")
        st.dataframe(result_df[['Book Name','Author Name','Description', 'Number of Reviews']].head(10),hide_index=True)
        st.divider()

    elif optionselected == "Top 5 Authors":
        tab1, tab2 = st.tabs(["📈 Charts", "📋 Table"])
        with tab1:
            st.subheader("Distribution of Top Authors")
            result_df = df['Author Name'].value_counts().head(5).reset_index()
            result_df.columns = ["Author Name", "No. of Books"]
            fig = px.pie(data_frame=result_df,names='Author Name',values='No. of Books',title='Top 5 Authors')
            st.plotly_chart(fig)
        with tab2:
            st.subheader("Distribution of Top Authors")
            st.dataframe(result_df, hide_index=True)
        st.divider()

    elif optionselected == "Books with Highest Reviews":
        result_df = df.sort_values(by='Number of Reviews', ascending=False)
        st.divider()
        st.subheader("10 Books with highest No. of Reviews")
        st.dataframe(result_df[['Book Name','Author Name','Description', 'Number of Reviews']].head(10),hide_index=True)        
        st.divider()

    elif  optionselected == "Distribution of Listening Time":
        result_df = df[['Book Name','Listening Time']]

        def to_hours(text):
            h = re.search(r'(\d+)\s*hour', str(text))
            m = re.search(r'(\d+)\s*minute', str(text))
            hours = int(h.group(1)) if h else 0
            minutes = int(m.group(1)) if m else 0
            return hours + minutes/60
        result_df["Hours"] = result_df["Listening Time"].apply(to_hours)
        result_df["Hours"] = result_df["Hours"].astype(int)
        fig = px.histogram(
            result_df,
            x="Hours",
            title="Listening Time Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

    elif optionselected == "Top 5 Most Listened Books":
        result_df = df[['Book Name', 'Author Name', 'Rating', 'Description', 'Listening Time', 'Number of Reviews']]

        def to_hours(text):
            h = re.search(r'(\d+)\s*hour', str(text))
            m = re.search(r'(\d+)\s*minute', str(text))
            hours = int(h.group(1)) if h else 0
            minutes = int(m.group(1)) if m else 0
            return hours + minutes/60
        result_df["Hours"] = result_df["Listening Time"].apply(to_hours)
        result_df["Hours"] = result_df["Hours"].astype(int)
        result_df = result_df.sort_values(by='Hours', ascending=False)
        st.divider()
        st.subheader("Top 5 Most Listened Books")
        st.dataframe(result_df[['Book Name', 'Author Name', 'Description', 'Listening Time']].head(5), hide_index=True)
        st.divider()

    elif optionselected == "Top Rated Books":
        result_df = df[['Book Name', 'Author Name', 'Description', 'Rating', 'Number of Reviews']]
        result_df = result_df.sort_values(
            by=['Rating', 'Number of Reviews'],
            ascending=[False, False]
        )
        st.divider()
        st.subheader("Top Rated Books based on Ratings and No. of Reviews")
        st.dataframe(result_df[['Book Name', 'Author Name']].head(10), hide_index=True)
        st.divider()

elif page == "NLP":
    st.error("No")


