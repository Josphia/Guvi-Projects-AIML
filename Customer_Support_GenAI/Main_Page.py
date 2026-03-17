import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout='wide', page_title="Customer Support NLP")

from datasets import load_dataset
dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
df = dataset['train'].to_pandas()









le = LabelEncoder()
df['label'] = le.fit_transform(df['queue']) # 
target_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    df['full_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

max_words = 10000
max_len = 150
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post')

st.title("AI-Powered Customer Support Automation") 

tabs = st.tabs(["Data Preview", "Baseline Model", "Deep Learning Preprocessing"])

with tabs[0]:
    st.subheader("Processed Data Sample")
    st.dataframe(df[['full_text', 'queue']].head())

with tabs[1]:
    st.subheader("Baseline: Logistic Regression") # [cite: 56]
    if st.button("Train & Evaluate Baseline"):
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_tfidf, y_train)
        preds = lr_model.predict(X_test_tfidf)
        
        st.text("Classification Report:")
        st.text(classification_report(y_test, preds, target_names=target_names)) # [cite: 74]

with tabs[2]:
    st.subheader("LSTM Preprocessing (Tokenization)") # [cite: 43]
    st.write(f"Vocabulary Size: {len(tokenizer.word_index)}")
    st.write("Example Padded Sequence (Input for LSTM):")
    st.code(X_train_seq[0])


