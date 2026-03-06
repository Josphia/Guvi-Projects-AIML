import os
import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(
    page_title="BrickView: Query Page",
    page_icon="🏘️",
    layout="wide"
)

conn = sqlite3.connect("brickview_database.db")

base_path = os.path.dirname(__file__)
root_path = os.path.abspath(os.path.join(base_path, ".."))

listings_path = os.path.join(root_path, "listings_20k.json")
listings = pd.read_json(listings_path)

property_attributes_path = os.path.join(root_path, "property_attributes_20k.json")
property_attributes = pd.read_json(property_attributes_path)

agents_path = os.path.join(root_path, "agents_20k.json")
agents = pd.read_json(agents_path)

agents_enhanced_path = os.path.join(root_path, "agents_enhanced_20k.json")
agents_enhanced = pd.read_json(agents_enhanced_path)

buyers_path = os.path.join(root_path, "buyers_20k.json")
buyers = pd.read_json(buyers_path)

sales_path = os.path.join(root_path, "sales_20k.csv")
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

col1, col2, col3 = st.columns([1,4,1])

with col2:
    st.subheader("❁❁ BrickView Real Estate - Queries ❁❁")

def run_query(query):
    return pd.read_sql(query, conn)

queries = {
    "1. What is the average listing price by city?": """
    SELECT City, ROUND(AVG(price),2) AS Average_Listing_Price
    FROM listings
    GROUP BY City
    ORDER BY Average_Listing_Price DESC""",
    "2. What is the average price per square foot by property type?": """
    SELECT property_type, ROUND(AVG(price/sqft),2) AS Avg_Price_Per_Sqft
    FROM listings
    GROUP BY property_type""",
    "3. How does furnishing status impact property prices?": """
    SELECT furnishing_status AS Furnishing_Status, ROUND(AVG(price),2) AS Prices
    FROM listings
    JOIN property_attributes
    ON listings.Listing_ID = property_attributes.Listing_ID
    GROUP BY Furnishing_Status""",
    "4. Do properties closer to metro stations command higher prices?": """
    SELECT 
        CASE
            WHEN metro_distance_km < 3 THEN "Close"
            WHEN metro_distance_km >= 3 AND metro_distance_km <= 6 THEN "Moderate"
            WHEN metro_distance_km > 6 THEN "Far Away"
        END AS Category,
    ROUND(AVG(price),2) as Price
    FROM property_attributes 
    JOIN listings
    ON property_attributes.listing_id = listings.listing_id
    GROUP BY Category""",
    "5. Are rented properties priced differently from non-rented ones?": """
    SELECT 
        CASE
            WHEN is_rented = '1' THEN 'Rented'
            WHEN is_rented = '0' THEN 'Not Rented'
        END AS Rented_Or_Not, 
        ROUND(AVG(price),2) AS Avg_Price
    FROM property_attributes 
    JOIN listings
    ON property_attributes.listing_id = listings.listing_id
    GROUP BY Rented_Or_Not""",
    "6. How do bedrooms and bathrooms affect pricing?": """
    SELECT bedrooms, bathrooms, ROUND(AVG(price),2) AS Avg_Price
    FROM listings
    JOIN property_attributes
    ON listings.listing_id = property_attributes.listing_id
    GROUP BY bedrooms, bathrooms""",
    "7. Do properties with parking and power backup sell at higher prices?": """
    SELECT 
        CASE 
            WHEN parking_available = '1' THEN 'Yes'
            WHEN parking_available = '0' THEN 'No'
        END AS Parking_Available,
        CASE 
            WHEN power_backup = '1' THEN 'Yes'
            WHEN power_backup = '0' THEN 'No'
        END AS Power_Backup, 
        ROUND(AVG(price),2) AS Price
    FROM property_attributes 
    JOIN listings 
    ON property_attributes.listing_id = listings.listing_id
    GROUP BY Parking_Available, Power_Backup
    ORDER BY Parking_Available desc, Power_Backup desc""",
    "8. How does year built influence listing price?": """
    SELECT Year_Built, ROUND(AVG(price),2) as Avg_Price
    FROM property_attributes
    JOIN listings 
    ON property_attributes.listing_id = listings.listing_id
    GROUP BY Year_Built""",
    "9. Which cities have the highest median property prices?": """
    SELECT city, price
    FROM listings""",
    "10. How are properties distributed across price buckets?": """
    SELECT 
        CASE
            WHEN price <= 500000 THEN 'Budget (<= 500000)'
            WHEN price > 500000 and price <= 1000000 THEN 'Mid-Range (> 500000 and <= 1000000)'
            WHEN price > 1000000 and price <= 1500000 THEN 'Premium (> 1000000 and <= 1500000)'
            ELSE 'Luxury (> 1500000)'
        END AS Price_Bucket_Label,
        COUNT(price)
    FROM listings 
    GROUP BY Price_Bucket_Label
    """,
    "11. What is the average days on market by city?": """
    SELECT City, ROUND(AVG(Days_On_Market), 2) AS Avg_Days_On_Market
    FROM listings 
    JOIN sales 
    ON listings.Listing_ID = sales.Listing_ID
    GROUP BY City""",
    "12. Which property types sell the fastest?": """
    SELECT Property_Type, ROUND(AVG(Days_On_Market), 2) AS Avg_Days_To_Sell
    FROM listings 
    JOIN sales 
    ON listings.Listing_ID = sales.Listing_ID
    GROUP BY Property_Type
    ORDER BY Avg_Days_To_Sell ASC""",
    "13. What percentage of properties are sold above listing price?": """
    SELECT 
        ROUND((SUM(CASE 
                WHEN sales.sale_price > listings.price THEN 1 
                ELSE 0 
            END) * 100.0 / COUNT(*)), 2) AS Percentage_Above_Listing
    FROM listings 
    JOIN sales 
    ON listings.listing_id = sales.listing_id""",
    "14. What is the sale-to-list price ratio by city?": """
    SELECT 
        l.city,
        SUM(s.sale_price) / SUM(l.price) AS Sale_To_List_Ratio
    FROM listings l
    JOIN sales s
    ON l.listing_id = s.listing_id
    GROUP BY l.city""",
    "15. Which listings took more than 90 days to sell?": """
    SELECT 
        listings.Listing_ID,
        listings.City,
        listings.Property_Type,
        listings.Price,
        listings.Sqft,
        listings.Date_Listed,
        sales.Sale_Price,
        sales.Date_Sold,
        sales.Days_on_Market
    FROM listings 
    JOIN sales
    ON listings.Listing_ID = sales.Listing_ID
    WHERE sales.Days_on_Market > 90
    ORDER BY sales.Days_on_Market ASC;""",
    "16. How does metro distance affect time on market?": """
    SELECT 
        CASE
            WHEN p.metro_distance_km <= 3 THEN 'Near'
            WHEN p.metro_distance_km <= 6 THEN 'Moderate'
            ELSE 'Far Away'
        END AS Distance_Scale,
        ROUND(AVG(s.days_on_market), 2) AS Time_On_Market
    FROM property_attributes p
    JOIN sales s
    ON p.Listing_ID = s.Listing_ID
    GROUP BY Distance_Scale
    ORDER BY Time_On_Market ASC""",
    "17. What is the monthly sales trend?": """
    SELECT
        strftime('%Y-%m', Date_Sold) AS Month_Year,
        COUNT(Listing_ID) AS Total_Sales
    FROM sales
    GROUP BY Month_Year
    ORDER BY Month_Year ASC;""",
    "18. Which properties are currently unsold?": """
    SELECT 
        listings.Listing_ID,
        listings.City,
        listings.Property_Type,
        listings.Price,
        listings.Sqft,
        listings.Date_Listed,
        listings.Agent_ID,
        listings.Latitude,
        listings.Longitude,
        sales.Sale_Price,
        sales.Date_Sold,
        sales.Days_on_Market
    FROM listings
    LEFT JOIN sales
    ON listings.Listing_ID = sales.Listing_ID
    WHERE sales.Sale_Price IS NULL""",
    "19. Which agents have closed the most sales?": """
    SELECT a.Name AS Agent_Name, ae.deals_closed AS Deals_Closed
    FROM agents a
    JOIN agents_enhanced ae
    ON a.Agent_ID = ae.agent_id
    ORDER BY ae.deals_closed DESC""",
    "20. Who are the top agents by total sales revenue?": """
    SELECT a.Name AS Agent_Name, ROUND(SUM(s.Sale_Price), 2) AS Total_Sales_Revenue
    FROM listings l
    JOIN sales s ON l.Listing_ID = s.Listing_ID
    JOIN agents a ON a.Agent_ID = l.Agent_ID
    GROUP BY A.Name
    ORDER BY Total_Sales_Revenue DESC""",
    "21. Which agents close deals fastest?": """
    SELECT a.agent_id AS Agent_ID , ae.avg_closing_days AS Avg_Closing_Days
    FROM agents_enhanced ae
    JOIN agents a ON a.Agent_ID = ae.agent_id
    ORDER BY avg_closing_days ASC""",
    "22. Does experience correlate with deals closed?": """
    SELECT 
        CASE
            WHEN experience_years <= 5 THEN '0-5 Years'
            WHEN experience_years <= 10 THEN '6-10 Years'
            WHEN experience_years <= 15 THEN '11-15 Years'
            WHEN experience_years <= 20 THEN '16-20 Years'
            ELSE 'More than 20 Years'
        END AS Experience_Category,
        SUM(deals_closed) AS Total_Deals_Closed
    FROM agents_enhanced 
    GROUP BY Experience_Category 
    ORDER BY Total_Deals_Closed DESC""",
    "23. Do agents with higher ratings close deals faster?": """
    SELECT
        CASE
            WHEN Rating >= 4.5 THEN '1. 4.5 - 5.0 (Excellent)'
            WHEN Rating >= 4.0 THEN '2. 4.0 - 4.4 (Very Good)'
            WHEN Rating >= 3.5 THEN '3. 3.5 - 3.9 (Good)'
            WHEN Rating >= 3.0 THEN '4. 3.0 - 3.4 (Average)'
            ELSE 'Below 3.0 (5. Low)'
        END AS Rating_Category,
        ROUND(AVG(Avg_Closing_Days), 3) AS Avg_Closing_Time
    FROM agents_enhanced
    GROUP BY Rating_Category
    ORDER BY Avg_Closing_Time ASC;""",
    "24. What is the average commission earned by each agent?": """
    SELECT a.Name, ROUND(AVG(s.Sale_Price * ae.commission_rate * 0.01), 2) AS Average_Commission_Earned
    FROM agents_enhanced ae
    JOIN agents a ON ae.agent_id = a.agent_id
    JOIN listings l ON a.agent_id = l.agent_id
    JOIN sales s ON s.Listing_ID = l.listing_id
    GROUP BY a.Name""",
    "25. Which agents currently have the most active listings?": """
    SELECT a.Name AS Agents_Name, count(*) AS Active_Listings
    FROM listings l
    JOIN agents_enhanced ae ON l.Agent_ID = ae.agent_id
    JOIN agents a ON l.Agent_ID = a.agent_id
    LEFT JOIN sales s ON l.Listing_ID = s.Listing_ID
    WHERE s.Sale_Price IS NULL
    GROUP BY Agents_Name
    ORDER BY Active_Listings DESC""",
    "26. What percentage of buyers are investors vs end users?": """
    SELECT 
        buyer_type AS Buyer_Type, (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM buyers)) AS Percentage_Of_Buyers
    FROM buyers 
    GROUP BY buyer_type
    ORDER BY buyer_type DESC""",
    "27. Which cities have the highest loan uptake rate?": """
    SELECT l.City, ROUND((COUNT(*) * 100.0 / 356), 2) AS Loan_Uptake_Rate
    FROM buyers b
    JOIN listings l ON b.sale_id = l.Listing_ID
    WHERE b.loan_taken = 1
    GROUP BY l.City
    ORDER BY Loan_Uptake_Rate DESC""",
    "28. What is the average loan amount by buyer type?": """
    SELECT buyer_type AS Buyer_Type, ROUND(AVG(loan_amount),2) AS Average_Loan_Amount
    FROM buyers 
    GROUP BY Buyer_Type""",
    "29. Which payment mode is most commonly used?": """
    SELECT payment_mode AS Payment_Mode, COUNT(*) AS Usage_Count
    FROM buyers
    GROUP BY Payment_Mode
    ORDER BY Usage_Count DESC""",
    "30. Do loan-backed purchases take longer to close?": """
    SELECT 
        CASE
            WHEN b.loan_taken = 0 THEN 'Loan Not Taken'
            ELSE 'Loan Taken'
        END AS Loan_Taken_Or_Not,
        ROUND(AVG(s.Days_On_Market), 2) AS Avg_Days_To_Close
    FROM sales s
    JOIN buyers b ON s.Listing_ID = b.sale_id
    GROUP BY b.loan_taken;"""
}

options = ["None"] + list(queries.keys())

selected_query = st.selectbox(
    "Select a query from below",
    options
)

st.divider()

if selected_query != "None":
    query = queries[selected_query]
    df = pd.read_sql(query, conn)

    if selected_query.startswith("9."):
        result9 = (
            df.groupby('City')['Price']
            .median()
            .reset_index()
        )
        result9.columns = ['City', 'Median_Property_Price']
        result9 = result9.sort_values(
            by='Median_Property_Price',
            ascending=False
        )
        st.dataframe(result9)
    else:
        st.dataframe(df)
