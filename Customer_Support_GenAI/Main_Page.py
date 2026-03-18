import streamlit as st
import joblib
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai

st.set_page_config(page_title="AI Support Chat", layout="centered")

@st.cache_resource
def load_resources():
    genai.configure(api_key="AIzaSyCYHXkwd-2LEB0Kk8-MIiJMhzcz2Kd0Ps0") 
    model_gemini = genai.GenerativeModel(model_name="gemini-2.5-flash")
    model = tf.keras.models.load_model("customer_model.h5", compile=False)
    tokenizer = joblib.load("tokenizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return model_gemini, model, tokenizer, label_encoder

model_gemini, model, tokenizer, label_encoder = load_resources()
max_len = 100

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_ticket(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    label_index = np.argmax(pred)
    return label_encoder.inverse_transform([label_index])[0]

st.title("🤖 AI Support Assistant")
st.caption("How can I help you with your train booking today?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your issue here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        category = predict_ticket(prompt)

        ai_prompt = f"You are a polite and professional customer support assistant for a train ticket booking system. User Query: {prompt}\nCategory: {category}\nGive a helpful, polite, friendly short response."
        
        try:
            response_text = model_gemini.generate_content(ai_prompt).text
            full_response = f"**Category: {category}**\n\n{response_text}"
        except Exception as e:
            full_response = "I'm sorry, I'm having trouble connecting to the server."

    with st.chat_message("assistant"):
        st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})