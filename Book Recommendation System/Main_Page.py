import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout='wide')



base_path = os.path.dirname(__file__)

csv1_path = os.path.join(base_path, "Audible_Catlog.csv")
csv1 = pd.read_csv(csv1_path)
csv1 = csv1.drop_duplicates()

csv2_path = os.path.join(base_path, "Audible_Catlog_Advanced_Features.csv")
csv2 = pd.read_csv(csv2_path)
csv2 = csv2.drop_duplicates()

def to_hours(text):
    h = re.search(r'(\d+)\s*hour', str(text))
    m = re.search(r'(\d+)\s*minute', str(text))
    hours = int(h.group(1)) if h else 0
    minutes = int(m.group(1)) if m else 0
    return hours + minutes/60
def extract_genres(text):
    if not isinstance(text, str): return []
    
    parts = text.split(',')
    genres = []
    
    for part in parts:
        match = re.search(r'in (.*)', part)
        if match:
            genre = match.group(1).strip()
            genre = re.sub(r'\(.*\)', '', genre).strip()
            if genre and "Audible Audiobooks" not in genre:
                genres.append(genre)
    return genres

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
df["Hours"] = df["Listening Time"].apply(to_hours)
df['genre_list'] = df['Ranks and Genre'].apply(extract_genres)
df['main_genre'] = df['genre_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
df = df[df["main_genre"] != "Unknown"]



df.columns = ['Book Name', 'Author Name', 'Rating', 'Price', 'Description', 'Listening Time', 'Ranks and Genre', 'Number of Reviews', 'Popularity', 'Listening Hours', 'Genre List', 'Main Genre']

with st.sidebar:
    page = st.radio("Go to", ['Home', 'EDA'])

if page == "Home":
    st.subheader("🪄 Recommendation Engine ⚙️")

    tab1, tab2 = st.tabs(["Content-Based Recommendations 🧩", "Genre-Based Recommendations 🎭"])

    with tab1:
        st.write("#### 📚 Browse Books by Content 🔍")
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(''))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        book_choice = st.selectbox("Choose a book:", ['None'] + df['Book Name'].tolist())

        if st.button("Get Recommendations"):
            if book_choice == 'None':
                st.warning("Please select a book")
            else:
                matches = df[df['Book Name'] == book_choice]
                if len(matches) > 0:
                    idx = matches.index[0]
                    sim_scores = list(enumerate(cosine_sim[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    book_indices = [i[0] for i in sim_scores[1:6]]
                    st.write(f"##### Book Recommendations similar to ***{book_choice}*** 🧩 ")
                    st.dataframe(
                        df.iloc[book_indices][['Book Name', 'Author Name', 'Rating', 'Main Genre']],
                        hide_index=True
                    )
                else:
                    st.error("Book not found")

    with tab2:
        st.write("#### 📚 Browse Books by Genre 🔍")
        genres = ['None'] + sorted(df['Main Genre'].dropna().unique())
        selected_genre = st.selectbox("Select a genre:", genres)
        if st.button("Get Recommendations "):
            if selected_genre == 'None':
                st.warning("Please select a genre")
            else:
                genre_books = df[df['Main Genre'] == selected_genre] \
                    .sort_values(by='Rating', ascending=False)
                if len(genre_books) > 0:
                    st.write(f"##### Top books in ***{selected_genre}*** 🎭")
                    st.dataframe(
                        genre_books[['Book Name', 'Author Name', 'Main Genre', 'Rating']].head(10),
                        hide_index=True
                    )
                else:
                    st.info("No books found")

    

elif page == "EDA":

    st.subheader("📊 Exploratory Data Analysis (EDA)")

    col1, col2, col3 = st.columns([2,2,3])  

    col1.metric("Total Books", len(df), border=True)
    col2.metric("Avg Rating", round(df['Rating'].mean(),2), border=True)
    col3.metric("Top Author", df['Author Name'].value_counts().idxmax(), border=True)
    st.divider()

    optionselected = st.selectbox("Select one from Below", ['None', 'Most Popular Genres',
                                                            'Authors with Highest-Rated Books',
                                                            'Ratings Distribution across Books',
                                                            'Distribution of Ratings across Review',
                                                            'Books frequently Clustered together based on Descriptions',
                                                            'How does genre similarity affect book recommendations?',
                                                            'Effect of Author Popularity on Book Ratings',
                                                            'Top 10 Popular Books',
                                                            'Top 5 Authors',
                                                            'Books with Highest Reviews',
                                                            'Distribution of Listening Time',
                                                            'Top 5 Most Listened Books',
                                                            'Top Rated Books'])

    if optionselected == "Most Popular Genres":
        result_df = df['Main Genre'].value_counts().head(5).reset_index()
        result_df.columns = ['Genre', 'Count']
        st.subheader("Top 5 Popular Genre")
        tab1, tab2 = st.tabs(["📈 Charts", "📋 Table"])
        with tab1:
            fig = px.pie(data_frame=result_df,names='Genre',values='Count',title='Popular Genre Distribution')
            st.plotly_chart(fig)
        with tab2:
            st.dataframe(result_df, hide_index=True)
        st.divider()

    elif optionselected == "Authors with Highest-Rated Books":
        result_df = df[['Book Name', 'Author Name', 'Rating', 'Popularity']].sort_values(by=['Rating', 'Popularity'], ascending=[False, False])
        st.subheader("Authors with Highest-Rated Books")
        st.dataframe(result_df[['Book Name', 'Author Name']].head(5), hide_index=True)
        st.divider()

    elif optionselected == "Ratings Distribution across Books":
        min_Rating = df['Rating'].min()
        max_Rating = df['Rating'].max()
        st.write("")
        rating_Selected = st.slider("Select a Rating",min_value=min_Rating, max_value=max_Rating, value=max_Rating, step=0.1, width="stretch")
        result_df = df[df['Rating']==rating_Selected]
        st.dataframe(result_df[['Book Name', 'Author Name', 'Rating', 'Description']], hide_index=True)
        st.divider()

    elif optionselected == "Distribution of Ratings across Review":
        abc = df.copy()
        abc['Number of Reviews'] = pd.to_numeric(abc['Number of Reviews'], errors='coerce')
        abc['Review_Bucket'] = pd.qcut(abc['Number of Reviews'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Review_Bucket', y='Rating', data=abc, palette='pastel', ax=ax)
        ax.set_title('Distribution of Ratings across Review Volume Buckets')
        ax.set_xlabel('Number of Reviews (Grouped)')
        ax.set_ylabel('Rating')
        st.pyplot(fig)
        st.divider()

    elif optionselected == "Books frequently Clustered together based on Descriptions":
        result_df = df.copy()
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(result_df['Description'].fillna(''))
        kmeans = KMeans(n_clusters=5, random_state=42)
        result_df['Cluster'] = kmeans.fit_predict(X)
        st.subheader("Books frequently Clustered together based on Descriptions")
        for i in range(5):
            st.write(f"\n📚 Cluster Group {i+1}")
            st.dataframe(result_df[result_df['Cluster'] == i]['Book Name'].head(7), hide_index=True)

    elif optionselected == "How does genre similarity affect book recommendations?":
        st.subheader("📚 Genre-Based Recommendations")
        genre_dummies = df['Genre List'].str.join('|').str.get_dummies()
        similarity = cosine_similarity(genre_dummies)
        book_name = st.selectbox("Choose a book", df['Book Name'])
        idx = df[df['Book Name'] == book_name].index[0]
        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_books = [df.iloc[i[0]]['Book Name'] for i in scores[1:6]]
        st.write("### 🔍 Recommended Books (Similar Genres)")
        for book in top_books:
            st.write("•", book)

    elif optionselected == "Effect of Author Popularity on Book Ratings":
        st.subheader("📊 Effect of Author Popularity on Book Ratings")
        author_books = df.groupby('Author Name')['Book Name'].count().reset_index()
        author_books.columns = ['Author Name', 'Book_Count']
        author_rating = df.groupby('Author Name')['Rating'].mean().reset_index()
        author_rating.columns = ['Author Name', 'Avg_Rating']
        author_stats = author_books.merge(author_rating, on='Author Name')
        corr = author_stats['Book_Count'].corr(author_stats['Avg_Rating'])
        st.metric("Correlation (Author's Popularity vs Book's Rating)", round(corr, 2))
        if corr > 0:
            st.success("Positive correlation → Popular authors tend to have higher ratings.")
        elif corr < 0:
            st.warning("Negative correlation → Popular authors tend to have lower ratings.")
        else:
            st.info("No correlation → Popularity does not affect ratings.")  
    
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

        ########Changes needed
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

