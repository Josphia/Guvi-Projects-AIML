import streamlit as st
import joblib
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai

genai.configure(api_key="AIzaSyC1ldwx-sND14YE5v9cHCV8AmnNSPeK0zg") 

model_gemini = genai.GenerativeModel("gemini-pro")

model = tf.keras.models.load_model("customer_model.h5", compile=False, custom_objects={})
tokenizer = joblib.load("tokenizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")

max_len = 150

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_ticket(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded)
    label_index = np.argmax(pred)

    return label_encoder.inverse_transform([label_index])[0]

def generate_response(ticket, category):

    prompt = f"""
You are a professional customer support assistant.

A customer has submitted the following support ticket:
"{ticket}"

The issue has been classified under: {category}

Write a polite, empathetic, and professional response:
- Acknowledge the issue
- Apologize if needed
- Assure that the issue is being handled
- Keep it short and clear
- Do NOT mention AI or classification

Response:
"""

    try:
        response = model_gemini.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Our system is currently unable to generate a response. Please try again shortly.{e}"

st.set_page_config(page_title="AI Customer Support", layout="centered")

st.title("🤖 AI Customer Support Assistant")
st.write("Enter your issue and get an instant response!")

user_input = st.text_area("✍️ Enter your support ticket:")

if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("Please enter a valid ticket.")
    else:
        with st.spinner("Analyzing your request..."):

            category = predict_ticket(user_input)
            response = generate_response(user_input, category)

        st.success("✅ Done!")

        st.subheader("📂 Predicted Category:")
        st.write(f"**{category}**")

        st.subheader("💬 AI Response:")
        st.write(response)