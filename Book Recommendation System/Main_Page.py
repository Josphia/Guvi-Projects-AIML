import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(layout='wide')



base_path = os.path.dirname(__file__)

csv1_path = os.path.join(base_path, "Audible_Catlog.csv")
csv1 = pd.read_csv(csv1_path)

csv2_path = os.path.join(base_path, "Audible_Catlog_Advanced_Features.csv")
csv2 = pd.read_csv(csv2_path)

df = csv1.merge(csv2, how="inner", on="Book Name")
df = df.drop(['Author_y','Rating_y', 'Price_x'], axis=1)
df['Number of Reviews'] = df[['Number of Reviews_x', 'Number of Reviews_y']].max(axis=1)
df = df.drop(['Number of Reviews_x', 'Number of Reviews_y'], axis=1)
df = df.drop_duplicates()
df = df.dropna(subset=['Description'])
df['Number of Reviews'] = df['Number of Reviews'].fillna(0).astype(int)
df = df.copy()
df['Ranks and Genre'] = df['Ranks and Genre'].str.lstrip(',')
df = df.copy()
df['Rating_x'] = df['Rating_x'].replace(-1.0, np.nan)
df['Listening Time'] = df['Listening Time'].replace("-1", "None")
df['Ranks and Genre'] = df['Ranks and Genre'].replace("-1", "None")
df = df.dropna(subset=['Rating_x'])
df['Popularity'] = df['Rating_x'] * df['Number of Reviews']
df.columns = ['Book Name', 'Author Name', 'Rating', 'Price', 'Description', 'Listening Time', 'Ranks and Genre', 'Number of Reviews', 'Popularity']

with st.sidebar:
    page = st.radio("Go to", ['Home', 'EDA', 'NLP'])

if page == "Home":
    st.title("You are in Home Page")

elif page == "EDA":
    st.subheader("Exploratory Data Analysis (EDA)")
    optionselected = st.selectbox("Select one from Below", ['None', 'Ratings Distribution', 
                                             'Top 10 Popular Books',
                                             'Top 5 Authors'])
    if optionselected == "Ratings Distribution":
        min_Rating = df['Rating'].min()
        max_Rating = df['Rating'].max()
        rating_Selected = st.slider("Select a Rating",min_value=min_Rating, max_value=max_Rating, value=max_Rating, step=0.1)
        result_df = df[df['Rating']==rating_Selected]
        st.dataframe(result_df, hide_index=True)
    
    elif optionselected == "Top 10 Popular Books":
        result_df = df.sort_values(by='Popularity', ascending=False)
        st.dataframe(result_df[['Book Name','Author Name','Description', 'Number of Reviews']].head(10),hide_index=True)

    elif optionselected == "Top 5 Authors":
        result_df = df['Author Name'].value_counts().head(5).reset_index()
        result_df.columns = ["Author Name", "No. of Books"]
        fig = px.pie(data_frame=result_df,names='Author Name',values='No. of Books',title='Top 5 Authors')
        st.plotly_chart(fig)

elif page == "NLP":
    st.error("No")


st.dataframe(df, hide_index=True)
