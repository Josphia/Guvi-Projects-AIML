import streamlit as st
import pandas as pd
import os

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "india_housing_prices.csv")

df = pd.read_csv(file_path)

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="📈",
    layout="wide"
)

st.sidebar.header("Filter")
selected_city = st.sidebar.selectbox("By City", ["Select a City"] + sorted(df['City'].unique()))
selected_state = st.sidebar.selectbox("By State", ["Select a State"] + sorted(df['State'].unique()))
selected_property_type = st.sidebar.selectbox("By Property Type", ["Select a Property Type"] + sorted(df['Property_Type'].unique()))
selected_bhk = st.sidebar.selectbox("By BHK", ["Select BHK"] + sorted(df['BHK'].unique()))

st.sidebar.divider()

result = st.empty()

filtered_df = None
title = None

if selected_city != "Select a City":
    filtered_df = df[df["City"] == selected_city]
    title = f"Properties in {selected_city} City"

if selected_state != "Select a State":
    filtered_df = df[df["State"] == selected_state]
    title = f"Properties in {selected_state}"

if selected_property_type != "Select a Property Type":
    filtered_df = df[df["Property_Type"] == selected_property_type]
    title = f"Properties with Property Type: {selected_property_type}"

if selected_bhk != "Select BHK":
    filtered_df = df[df["BHK"] == selected_bhk]
    title = f"Properties with {selected_bhk} BHK"

if filtered_df is not None:
    display_df = filtered_df[
        ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Price_in_Lakhs", "Year_Built"]
    ]

    with result.container():
        st.subheader(title)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

else:
    st.divider()
    st.title("🏘️ Real Estate Investment Advisor")
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Properties", f"{len(df):,}")
    col2.metric("Avg. Price", f"₹{df['Price_in_Lakhs'].mean():.1f} L")
    col3.metric("Top Market", df['City'].mode()[0])

    st.divider()
    st.subheader("📊 Market Overview")
    top_cities = df['State'].value_counts().head(10)
    st.bar_chart(top_cities)

    st.divider()
    st.subheader("Average price of Properties in each State")
    avg_price_of_cities = df.groupby("State")["Price_in_Lakhs"].mean().head(15)
    st.line_chart(avg_price_of_cities)
    st.divider()