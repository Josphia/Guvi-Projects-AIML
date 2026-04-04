# AI-Powered Customer Support Automation 🤖

## Overview
Automates customer ticket classification and generates empathetic responses using Bidirectional LSTM and Google Gemini API.

## Features
- Automated ticket classification (Billing, Technical, Account)
- AI-generated responses (Gemini 2.5 Flash)
- Streamlit dashboard (real-time chat)
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

## Tech Stack
- TensorFlow, Keras (BiLSTM)
- Scikit-learn
- Google Gemini API
- NLTK, Hugging Face
- Streamlit

## Project Structure
Customer_Support_GenAI/
- baseline_lr_model.joblib
- customer_model.h5
- label_encoder.joblib
- Main_Page.py
- tfidf_vectorizer.joblib
- tokenizer.joblib
- train_model.py

## Run
streamlit run Main_Page.py

## Metrics
Accuracy | Precision | Recall | F1-score