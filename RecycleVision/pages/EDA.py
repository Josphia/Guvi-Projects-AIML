import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import random

st.set_page_config(page_title="RecycleVision - EDA", layout="wide")

st.sidebar.subheader("Exploratory Data Analysis")
app_mode = st.sidebar.radio("Choose an option:", ["Number of Images in each Class", "Sample Images in each Category", "Pixel Intensity Analysis"])

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "..", "data")
dataset_path = os.path.abspath(dataset_path)
categories = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

if app_mode == "Number of Images in each Class":
    st.subheader("♻️ RecycleVision: Image Classification") 
    st.divider()

    data_counts = []
    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        data_counts.append({"Category": category.capitalize(), "Count": count})
    
    df = pd.DataFrame(data_counts).set_index("Category")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Images", df["Count"].sum())
    col2.metric("Total Classes", len(categories))

    st.divider()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("rocket", len(df))
    plot = sns.barplot(data=df, x="Category", y="Count", palette=colors, ax=ax)

    for i in ax.containers:
        ax.bar_label(i)

    plt.title("Number of Images in Each Class", fontsize=14)
    plt.xlabel("Class Type", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=45)

    st.pyplot(fig)

elif app_mode == "Sample Images in each Category":

    st.subheader("♻️ RecycleVision: Sample Images in each Class") 
    st.divider()

    values = random.sample(range(1, 138), 3)

    for i, category in enumerate(categories):
        folder_path = os.path.join(dataset_path, category)
        images = os.listdir(folder_path)

        st.write(category)

        col1, col2, col3 = st.columns(3)
        with col1:
            img_path1 = os.path.join(folder_path, images[values[0]])
            img1 = Image.open(img_path1)
            st.image(img1, use_container_width=True)
        with col2:
            img_path2 = os.path.join(folder_path, images[values[1]])
            img2 = Image.open(img_path2)
            st.image(img2, use_container_width=True)
        with col3:
            img_path3 = os.path.join(folder_path, images[values[2]])
            img3 = Image.open(img_path3)
            st.image(img3, use_container_width=True)

        st.divider()    

elif app_mode == "Pixel Intensity Analysis":
    st.subheader("🎨 Pixel Intensity & Color Distribution")
    st.divider()
        
    selected_cat = st.selectbox("Select a category to analyze:", categories)
    
    valuess = random.sample(range(1, 138), 1)
    cat_folder = os.path.join(dataset_path, selected_cat)
    sample_img_name = os.listdir(cat_folder)[valuess[0]]
    img_path = os.path.join(cat_folder, sample_img_name)
    
    img = Image.open(img_path)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption=f"Sample: {selected_cat}", use_container_width=True)
    with col2:
        img_array = np.array(img)
        fig, ax = plt.subplots()
        colors = ('red', 'green', 'blue')
        
        for i, color in enumerate(colors):
            hist, bin_edges = np.histogram(img_array[:, :, i], bins=256, range=(0, 256))
            ax.plot(bin_edges[0:-1], hist, color=color, label=f'{color.title()} channel')
        
        ax.set_title(f"Color Distribution: {selected_cat.capitalize()}")
        ax.set_xlabel("Pixel Intensity (0-255)")
        ax.set_ylabel("Pixel Count")
        ax.legend()
        
        st.pyplot(fig)