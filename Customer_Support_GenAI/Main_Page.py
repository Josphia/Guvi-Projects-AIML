import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

st.set_page_config(layout='wide')

dataset = load_dataset("Tobi-Bueck/customer-support-tickets")

df = dataset['train'].to_pandas()

print(df.isnull().sum())
print(df.duplicated().sum())

st.dataframe(df)
