import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Future Price 5Y",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Real Estate Investment Advisor")
st.subheader("Predicting the Future Price in 5 Years")

@st.cache_resource
def load_assets():
    base_path = os.path.dirname(os.path.dirname(__file__))
    models = joblib.load(os.path.join(base_path, "models_r.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler_r.pkl"))
    features = joblib.load(os.path.join(base_path, "features_r.pkl"))
    le_city = joblib.load(os.path.join(base_path, "le_city.pkl"))
    return models, scaler, features, le_city

models, scaler, features, le_city = load_assets()

model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

bhk = st.number_input("BHK (between 1–10)", 1, 10, 2)
sqft = st.number_input("Size in SqFt", 100, 10000, 1200)
price = st.number_input("Current Price (in Lakhs)", 1.0, 1000.0, 50.0)
year_built = st.number_input("Year Built", 1950, 2026, 2015)
city = st.selectbox("City", le_city.classes_)

st.write("\n")

if st.button("Predict Future Price and Evaluate Model Metrics"):

    city_enc = le_city.transform([city])[0]

    user_df = pd.DataFrame([[bhk, sqft, price, year_built, city_enc]], columns=features) 

    user_scaled = scaler.transform(user_df)
    future_price = model.predict(user_scaled)[0]

    st.success(f"💰 Estimated Future Price (5 Years): ₹{future_price:.2f} Lakhs")
