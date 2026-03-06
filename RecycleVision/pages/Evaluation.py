import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os

st.set_page_config(page_title="RecycleVision - Model Evaluation", page_icon="♻️")

st.subheader("📊 Model Evaluation")

model_choice = st.radio(
    "Choose Model:",
    ["MobileNetV2", "EfficientNetB0", "Compare Both"]
)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data")
mobilenet_model_path = os.path.join(BASE_DIR, "mobilenet_model.h5")
efficientnet_model_path = os.path.join(BASE_DIR, "efficientnet_model.h5")

val_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

y_true = val_generator.classes
class_names = list(val_generator.class_indices.keys())

def evaluate(model_path):
    model = load_model(model_path, compile=False)
    val_generator.reset()
    preds = model.predict(val_generator)
    y_pred = np.argmax(preds, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    avg_f1 = np.mean(f1)

    metrics_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    return accuracy, avg_f1, metrics_df, y_pred


if model_choice == "MobileNetV2":
    accuracy, avg_f1, metrics_df, y_pred = evaluate(mobilenet_model_path)

elif model_choice == "EfficientNetB0":
    accuracy, avg_f1, metrics_df, y_pred = evaluate(efficientnet_model_path)

elif model_choice == "Compare Both":

    acc1, f1_1, _, _ = evaluate(mobilenet_model_path)
    acc2, f1_2, _, _ = evaluate(efficientnet_model_path)

    comparison = pd.DataFrame({
        "Model": ["MobileNetV2", "EfficientNetB0"],
        "Accuracy": [acc1, acc2],
        "Average F1-Score": [f1_1, f1_2]
    })

    st.subheader("Comparison Table")
    st.dataframe(comparison, hide_index=True)

    if f1_1 > f1_2:
        st.success("The Best Model is MobileNetV2")
    else:
        st.success("The Best Model is EfficientNetB0")

    st.stop()

st.subheader("Metrics:")
st.dataframe(metrics_df, hide_index=True)

st.subheader("Accuracy:")
st.write(f"The overall accuracy is {accuracy*100:.2f}%")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.subheader("🧩 Confusion Matrix")
st.pyplot(fig)