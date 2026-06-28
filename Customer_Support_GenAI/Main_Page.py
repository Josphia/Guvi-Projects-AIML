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
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

def set_bg(image_file):
    base64_img = get_base64(image_file)
    if base64_img:
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
max_len = 250

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
    avatar = "user.png" if message["role"] == "user" else "bot.png"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your issue here..."):

    st.chat_message("user", avatar="user.png").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        category, confidence = predict_ticket(prompt)

        ai_prompt = (
            f"You are a polite and professional customer support assistant for a train ticket booking system.\n"
            f"User Query: {prompt}\n"
            f"Category: {category}\n"
            f"Give a helpful, polite, friendly short response."
        )
        
        response_text = get_gemini_response(ai_prompt)
        
        if "Error:" not in response_text:
            full_response = f"**Category: {category} (Confidence: {confidence*100:.2f}%)**\n\n{response_text}"
        else:
            full_response = response_text

    with st.chat_message("assistant", avatar="bot.png"):
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})