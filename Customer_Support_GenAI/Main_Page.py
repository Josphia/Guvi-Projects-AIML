import streamlit as st
import pandas as pd
import numpy as np
import warnings
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

warnings.filterwarnings("ignore")
st.set_page_config(layout='wide', page_title="Customer Support NLP")

from datasets import load_dataset
dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
df = dataset['train'].to_pandas()

contractions_dict = {
    "don't": "do not", "can't": "cannot", "i'm": "i am", "it's": "it is",
    "i've": "i have", "you're": "you are", "doesn't": "does not"
}

stop_words = set(stopwords.words('english'))
custom_stops = {
    'dear', 'hello', 'hi', 'team', 'support', 'please', 
    'thank', 'thanks', 'regards', 'sincerely', 'writing', 'message'
}
stop_words.update(custom_stops)

def expand_contractions(text):
    for word, expanded in contractions_dict.items():
        text = text.replace(word, expanded)
    return text

def clean_text(text):
    text = str(text).lower()
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'<name>', '[NAME]', text)
    text = re.sub(r'<tel_num>', '[PHONE]', text)
    text = re.sub(r'<email>', '[EMAIL]', text)
    text = re.sub(r'<.*?>', '', text) # Remove any other bracketed tags
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

df = df[df['language'] == 'en'].copy()
df = df.dropna(subset=['body', 'queue'])
df['full_text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
df['full_text'] = df['full_text'].apply(expand_contractions)
df['full_text'] = df['full_text'].apply(clean_text)
df['full_text'] = df['full_text'].apply(remove_stopwords)
df = df[df['full_text'].str.strip() != '']
df = df.drop_duplicates(subset=['full_text'])
df = df.reset_index(drop=True)

le = LabelEncoder()
df['label'] = le.fit_transform(df['queue']) # 
target_names = le.classes_











st.title("Cleaned English Customer Support Dataset")

st.write("Processed Data Sample")
st.dataframe(df)
