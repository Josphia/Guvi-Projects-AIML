import os
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

st.set_page_config(
    page_title="BrickView: Real Estate Analytics",
    page_icon="🏘️",
    layout="wide"
)

conn = sqlite3.connect("brickview_database.db")

base_path = os.path.dirname(__file__)

listings_path = os.path.join(base_path, "listings_20k.json")
listings = pd.read_json(listings_path)

property_attributes_path = os.path.join(base_path, "property_attributes_20k.json")
property_attributes = pd.read_json(property_attributes_path)

agents_path = os.path.join(base_path, "agents_20k.json")
agents = pd.read_json(agents_path)

agents_enhanced_path = os.path.join(base_path, "agents_enhanced_20k.json")
agents_enhanced = pd.read_json(agents_enhanced_path)

buyers_path = os.path.join(base_path, "buyers_20k.json")
buyers = pd.read_json(buyers_path)

sales_path = os.path.join(base_path, "sales_20k.csv")
sales = pd.read_csv(sales_path)


listings = listings.drop_duplicates(subset="Listing_ID")
sales = sales.drop_duplicates(subset="Listing_ID")
property_attributes = property_attributes.drop_duplicates(subset="listing_id")
agents = agents.drop_duplicates(subset="Agent_ID")
agents_enhanced = agents_enhanced.drop_duplicates(subset="agent_id")
buyers = buyers.drop_duplicates(subset="sale_id")

listings.to_sql("listings", conn, if_exists="replace", index=False)
property_attributes.to_sql("property_attributes", conn, if_exists="replace", index=False)
agents.to_sql("agents", conn, if_exists="replace", index=False)
agents_enhanced.to_sql("agents_enhanced", conn, if_exists="replace", index=False)
buyers.to_sql("buyers", conn, if_exists="replace", index=False)
sales.to_sql("sales", conn, if_exists="replace", index=False)

#pd.options.display.float_format = '{:,.1f}'.format

st.title("🏙️ BrickView Dashboard")

def run_query(query):
    return pd.read_sql(query, conn)

st.sidebar.header("Filters")

city_query = "SELECT DISTINCT City FROM listings"
city_df = run_query(city_query)
city_list = ["All"] + city_df["City"].tolist()

property_type_query = "SELECT DISTINCT Property_Type FROM listings"
property_type_df = run_query(property_type_query)
property_type_list = ["All"] + property_type_df["Property_Type"].tolist()

agent_query = "SELECT DISTINCT Name FROM agents"
agent_df = run_query(agent_query)
agent_list = ["All"] + agent_df["Name"].tolist()

with st.sidebar.expander("Search 🔍"):
    selected_city = st.selectbox("📍 By City", city_list)
    show_data_city = st.button("Show City Results")
    selected_property_type = st.selectbox("🏙️ By Property Type", property_type_list)
    show_data_property = st.button("Show Property Results")
    selected_agent = st.selectbox("🧔🏻 By Agent", agent_list)
    show_data_agent = st.button("Show Agent Results")

if show_data_city:
    if selected_city == "All":
        query = """
        SELECT *
        FROM listings
        """
        df = run_query(query)

        st.subheader("Properties in all Cities")
        st.dataframe(df)

    else:
        query = f"""
        SELECT *
        FROM listings
        WHERE City = '{selected_city}'
        """
        df = run_query(query)

        st.subheader(f"Properties in {selected_city}")
        st.dataframe(df)

if show_data_property:
    if selected_property_type == "All":
        query = """
            SELECT *
            FROM listings
        """
        df = run_query(query)

        st.subheader(f"All Properties")
        st.dataframe(df)

    else:
        query = f"""
            SELECT *
            FROM listings
            WHERE Property_Type = '{selected_property_type}'
        """
        df = run_query(query)

        st.subheader(f"Properties by {selected_property_type}")
        st.dataframe(df)

if show_data_agent:
    if selected_agent == "All":
        query = """
            SELECT Name AS Agent_Name, Agent_ID, Phone, Email
            FROM agents
        """
        df = run_query(query)

        st.subheader("All Agents details")
        st.dataframe(df)

    else:
        query = f"""
            SELECT Name AS Agent_Name, Agent_ID, Phone, Email
            FROM agents
            WHERE Name = '{selected_agent}'
        """
        df = run_query(query)

        st.subheader(f"Details of {selected_agent}")
        st.dataframe(df)

price_query = "SELECT MIN(price) AS min_price, MAX(price) AS max_price FROM listings"
price_df = run_query(price_query)

min_price = int(price_df["min_price"][0])
max_price = int(price_df["max_price"][0])

price_range = st.sidebar.slider(
    "💰 Select by Price Range ",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)
show_price_data = st.sidebar.button("Show Properties within the Range")

if show_price_data:
    query = f"""
    SELECT 
        Listing_ID AS Property_ID, City, Property_Type, ROUND(Price, 0) AS Property_Price
    FROM listings
    WHERE Price BETWEEN {price_range[0]} AND {price_range[1]}
    """

    df_price = run_query(query)

    st.subheader(f"Properties Ranging between {price_range[0]} - {price_range[1]}")
    st.dataframe(df_price)


date_query = """
SELECT MIN(Date_Sold) AS min_date,
       MAX(Date_Sold) AS max_date
FROM sales
"""
date_df = run_query(date_query)

min_date = pd.to_datetime(date_df["min_date"][0])
max_date = pd.to_datetime(date_df["max_date"][0])

date_range = st.sidebar.date_input(
    "📅 Sale Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

show_date_data = st.sidebar.button("Show Sales within the Dates")

if show_date_data:
    start_date = date_range[0]
    end_date = date_range[1]

    query_date = f"""
    SELECT *
    FROM sales
    WHERE Date_Sold BETWEEN '{start_date}' AND '{end_date}'
    """

    df_date = run_query(query_date)

    st.subheader(f"Showing Sales between {start_date} - {end_date}")
    st.dataframe(df_date)

st.subheader("📍 Property Listings Map")

citymap_query = "SELECT DISTINCT City FROM listings"
citymap_df = run_query(citymap_query)
citymap_list = ["All"] + citymap_df["City"].tolist()

selected_citymap = st.selectbox("Select City", citymap_list)

if selected_citymap == "All":
    map_query = """
    SELECT 
        Latitude, Longitude, City, Price
    FROM listings
    """
else:
    map_query = f"""
    SELECT 
        Latitude, Longitude, City, Price
    FROM listings
    WHERE city = '{selected_citymap}'
    """

map_df = run_query(map_query)

map_df = map_df.rename(
    columns={
        "Latitude": "lat",
        "Longitude": "lon"
    }
)

col1, col2 = st.columns([3, 1])

with col1:
    st.map(map_df)

with col2:
    st.metric("Total Properties", len(map_df))

    avg_price = int(map_df["Price"].mean())
    st.metric("Average Price", f"₹{avg_price:,}")

    st.metric("No. of Cities", citymap_df["City"].count() )

st.divider()

col1, col2 = st.columns(2)

with col1:
    query = """
        SELECT City, COUNT(*)
        FROM listings
        GROUP BY City
    """
    df = run_query(query)

    st.subheader("📊 Bar Chart")
    st.text("No. of listings by city")
    st.bar_chart(df.set_index("City")) 

with col2:
    pie_query = f"""
    SELECT Property_Type, COUNT(*) AS Total
    FROM listings
    GROUP BY property_type
    """
    pie_df = run_query(pie_query)

    st.subheader("⭕ Pie Chart")
    st.text("Property Type Distribution")
    res = px.pie(pie_df, names="Property_Type", values="Total", hole=0.4)

    st.plotly_chart(res)

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("📈 Line Chart")
    st.text("Monthly Listings & Sales Trend")

    listings_trend_query = """
    SELECT 
        strftime('%Y-%m', Date_Listed) AS Month,
        COUNT(*) AS Total_Listings
    FROM listings
    GROUP BY Month
    ORDER BY Month
    """
    listings_trend_df = run_query(listings_trend_query)

    sales_trend_query = """
    SELECT 
        strftime('%Y-%m', Date_Sold) AS Month,
        COUNT(*) AS Total_Sales
    FROM sales
    GROUP BY Month
    ORDER BY Month
    """
    sales_trend_df = run_query(sales_trend_query)

    trend_df = pd.merge(listings_trend_df, sales_trend_df, on="Month", how="outer").fillna(0)

    st.line_chart(trend_df.set_index("Month"))

with col2:

    st.subheader("𝄜 Pagination and Sorting Table")

    table_query = """
    SELECT 
        Listing_ID, City, Property_Type, Price, Date_Listed
    FROM listings
    """

    table_df = run_query(table_query)

    property_types = ["All"] + sorted(table_df["Property_Type"].unique().tolist())

    selected_type = st.selectbox(
        "Select Property Type",
        property_types
    )

    if selected_type != "All":
        table_df = table_df[table_df["Property_Type"] == selected_type]

    table_df = table_df.sort_values(by="Price", ascending=True)

    rows_per_page = 10
    total_rows = len(table_df)
    total_pages = max(1, (total_rows // rows_per_page) + 1)

    page = st.number_input(
        "Page :",
        min_value=1,
        max_value=total_pages,
        step=1
    )

    start = (page - 1) * rows_per_page
    end = start + rows_per_page

    st.dataframe(table_df.iloc[start:end], width="stretch")


st.divider()

conn.close()