import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Future_Price_Regression")

base_path = r"E:\VS Code Projects\Guvi-Projects-AIML\Real Estate Investment Advisor Project"
data_path = f"{base_path}/india_housing_prices.csv"

df = pd.read_csv(data_path)

city_growth_rates = {
    'Ahmedabad': 1.45, 'Amritsar': 1.25, 'Bangalore': 1.70, 'Bhopal': 1.30,
    'Bhubaneswar': 1.35, 'Bilaspur': 1.20, 'Chennai': 1.48, 'Coimbatore': 1.40,
    'Cuttack': 1.28, 'Dehradun': 1.35, 'Durgapur': 1.22, 'Dwarka': 1.55,
    'Faridabad': 1.42, 'Gaya': 1.20, 'Gurgaon': 1.65, 'Guwahati': 1.32,
    'Haridwar': 1.25, 'Hyderabad': 1.62, 'Indore': 1.44, 'Jaipur': 1.38,
    'Jamshedpur': 1.28, 'Jodhpur': 1.25, 'Kochi': 1.35, 'Kolkata': 1.30,
    'Lucknow': 1.45, 'Ludhiana': 1.32, 'Mangalore': 1.30, 'Mumbai': 1.55,
    'Mysore': 1.35, 'Nagpur': 1.38, 'New Delhi': 1.58, 'Noida': 1.60,
    'Patna': 1.35, 'Pune': 1.52, 'Raipur': 1.28, 'Ranchi': 1.30,
    'Silchar': 1.18, 'Surat': 1.45, 'Trivandrum': 1.35, 'Vijayawada': 1.32,
    'Vishakhapatnam': 1.40, 'Warangal': 1.25
}

city_growth_rate = df["City"].map(city_growth_rates)
bhk_rate = df['BHK'] * 0.01
sqft_rate = df["Size_in_SqFt"] / 1000 * 0.02
year_built_rate = (2026 - df['Year_Built']) * 0.005

df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (city_growth_rate + bhk_rate + sqft_rate - year_built_rate)

le_city = LabelEncoder()
df["City"] = le_city.fit_transform(df["City"])

features = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "City"]

X = df[features]
y = df["Future_Price_5Y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    "Linear SVR": LinearSVR(max_iter=10000),
    "XGBoost": XGBRegressor(n_estimators=50, max_depth=5, random_state=42)
}

trained_models = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_name", name)
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "regression_model")

        trained_models[name] = model
        print(f"Finished training {name}")

joblib.dump(trained_models, f"{base_path}/models_r.pkl", compress=3)
joblib.dump(scaler, f"{base_path}/scaler_r.pkl")
joblib.dump(features, f"{base_path}/features_r.pkl")
joblib.dump(le_city, f"{base_path}/le_city.pkl")

print("Regression Training Completed")