import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("mlruns") 
mlflow.set_experiment("Good_Investment_Classification")

df = pd.read_csv("india_housing_prices.csv")
df = df.sample(n=50000, random_state=42)

city_median = df.groupby("City")["Price_per_SqFt"].transform("median")
df["Good_Investment"] = (df["Price_per_SqFt"] < city_median).astype(int)

features = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals", "Property_Type", "Furnished_Status"
]

X = df[features].copy()
y = df["Good_Investment"]

le_property = LabelEncoder()
le_furnished = LabelEncoder()

X["Property_Type"] = le_property.fit_transform(X["Property_Type"])
X["Furnished_Status"] = le_furnished.fit_transform(X["Furnished_Status"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "Random Forest": RandomForestClassifier(n_estimators=20, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=20, random_state=42),
    "Naive Bayes": GaussianNB()
}

trained_models = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None

        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(model, "model_artifact")
        
        trained_models[name] = model
        print(f"Finished training {name}")

joblib.dump(trained_models, "models_c.pkl")
joblib.dump(scaler, "scaler_c.pkl")
joblib.dump(features, "features.pkl")
joblib.dump(le_property, "le_property.pkl")
joblib.dump(le_furnished, "le_furnished.pkl")

print("Classification Training Completed")