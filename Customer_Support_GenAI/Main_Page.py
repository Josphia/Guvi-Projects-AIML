import streamlit as st
import base64
import joblib
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
#Customer_Support_GenAI
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="AI Support Chat", layout="wide", page_icon="🎫")

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(image_file):
    base64_img = get_base64(image_file)
    page_bg = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                    url("data:image/png;base64,{base64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

set_bg("background.png")

@st.cache_resource
def load_resources():
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel(model_name="gemini-3.1-flash-lite-preview")
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
    confidence = np.max(pred)
    category = label_encoder.inverse_transform([label_index])[0]
    return category, confidence

def get_gemini_response(prompt):
    for attempt in range(3):
        try:
            response = model_gemini.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):
                time.sleep(5)  
            else:
                return f"Error: {e}"
    return "Server is busy. Please try again in a few seconds."

st.title("AI Support Assistant 🤖")
st.caption("How can I help you with your train booking today?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = "user.png"
    else:
        avatar = "bot.png"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your issue here..."):

    st.chat_message("user", avatar="user.png").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):

        category, confidence = predict_ticket(prompt)

        ai_prompt = f"You are a polite and professional customer support assistant for a train ticket booking system. User Query: {prompt}\nCategory: {category}\nGive a helpful, polite, friendly short response."
        
        try:
            time.sleep(15)
            #response_text = get_gemini_response(ai_prompt)
            response_text = model_gemini.generate_content(ai_prompt).text
            full_response = f"**Category: {category} (Confidence: {confidence*100:.2f}%)**\n\n{response_text}"
        except Exception as e:
            full_response = f"Error: {e}" 
            #full_response = "I'm sorry, I'm having trouble connecting to the server."

    with st.chat_message("assistant", avatar="bot.png"):
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})