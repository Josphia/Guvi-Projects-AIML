import os
import pandas as pd
import numpy as np
import streamlit as st

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

st.sidebar.subheader("EDA")


st.subheader("Hii")
name = st.text_input("Enter your Name")
submit = st.button("Submit")
if submit and name:
    st.success(f"Hii {name}..")

#st.dataframe(df, hide_index=True)
