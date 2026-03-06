import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="RecycleVision", page_icon="♻️")

st.subheader(" ♻️ RecycleVision: Garbage Classifier ")
st.write("Upload an image to find its Category..")

@st.cache_resource
def load_my_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "mobilenet_model.h5")

    return tf.keras.models.load_model(model_path)

model = load_my_model()

class_names = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image Uploaded', use_container_width=True)
    st.write("Classifying...")

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    result_class = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    st.success(f"Prediction: **{result_class.upper()}**")
    st.info(f"Confidence Score: **{confidence:.2f}%**")