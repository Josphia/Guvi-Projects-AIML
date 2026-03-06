import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="EDA: Query Page",
    page_icon="📈",
    layout="wide"
)

st.subheader("🔍 Property Query Page")

base_path = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_path, "india_housing_prices.csv")

df = pd.read_csv(file_path)

option = st.selectbox("Select One from Below", options = [
    "1. What is the distribution of property prices?", 
    "2. What is the distribution of property sizes?", 
    "3. How does the price per sq ft vary by property type?", 
    "4. Is there a relationship between property size and price?",
    "5. Are there any outliers in price per sq ft or property size?",
    "6. What is the average price per sq ft by state?",
    "7. What is the average property price by city?",
    "8. What is the median age of properties by locality?",
    "9. How is BHK distributed across cities?",
    "10. What are the price trends for the top 5 most expensive localities?",
    "11. How are numeric features correlated with each other?",
    "12. How do nearby schools relate to price per sq ft?",
    "13. How do nearby hospitals relate to price per sq ft?",
    "14. How does price vary by furnished status?",
    "15. How does price per sq ft vary by property facing direction?",
    "16. How many properties belong to each owner type?",
    "17. How many properties are available under each availability status?",
    "18. Does parking space affect property price?",
    "19. How do amenities affect price per sq ft?",
    "20. How does public transport accessibility relate to price per sq ft?"
    ], index=None, placeholder="None")


if option == "1. What is the distribution of property prices?":
    st.subheader("📊 EDA - Property Price Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot( df['Price_in_Lakhs'], bins=25, kde=True, ax=ax, color="#ffc6c6" )
    ax.set_title("Distribution of Property Prices")
    ax.set_xlabel("Price (in Lakhs)")
    ax.set_ylabel("Number of Properties")
    st.pyplot(fig)  

elif option == "2. What is the distribution of property sizes?":
    st.subheader("📊 EDA - Property Sizes Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot( df['Size_in_SqFt'], bins=50, kde=True, ax=ax, color="#ffc6c6" )
    ax.set_title("Distribution of Property Sizes")
    ax.set_xlabel("Size in SqFt")
    ax.set_ylabel("Number of Properties")
    st.pyplot(fig)  

elif option == "3. How does the price per sq ft vary by property type?":
    count = df['Property_Type'].value_counts()
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(count, labels=count.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Distribution of Price Per Sqft by Property Types")
    st.pyplot(fig)

elif option == "4. Is there a relationship between property size and price?":
    fig, ax = plt.subplots(figsize=(7,4))
    df_sample = df.sample(1000)
    sns.regplot(data=df_sample, x='Size_in_SqFt', y='Price_in_Lakhs', line_kws={'color':'green'}, color='#ffc6c6')
    ax.set_title("Property Size vs Price with Trend Line")
    ax.set_xlabel("Size (in Sqft)")
    ax.set_ylabel("Price (in Lakhs)")
    st.pyplot(fig)

elif option == "5. Are there any outliers in price per sq ft or property size?":
    st.subheader("Outlier Detection: Size in SqFt")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, y='Size_in_SqFt', ax=ax, color='lightgreen')
    ax.set_title("Outliers in Property Size")
    st.pyplot(fig)

elif option == "6. What is the average price per sq ft by state?":
    st.subheader("Average Price per Sqft by State")
    df6 = df.groupby('State')['Price_per_SqFt'].mean()
    st.dataframe(df6)


elif option == "7. What is the average property price by city?":
    st.subheader("Average Price of Properties by Cities")
    df7 = df.groupby('City', as_index=False)['Price_in_Lakhs'].mean()
    df7.columns = ['City', 'Average_Price_(in_Lakhs)']
    st.dataframe(df7, hide_index=True)

elif option == "8. What is the median age of properties by locality?":
    st.subheader("Median Age of properties by Locality")
    df8 = df.groupby('Locality', as_index=False)['Age_of_Property'].median()
    df8.columns = ['Locality_Name', 'Median_Age_of_Properties']
    st.dataframe(df8, hide_index=True)

elif option == "9. How is BHK distributed across cities?":
    st.subheader("Distribution of BHKs across different Cities")
    df9 = df.groupby(['City','BHK'], as_index=False).size()
    df9.columns = ['City', 'BHK_Type', 'Count']
    st.dataframe(df9, hide_index=True)

elif option == "10. What are the price trends for the top 5 most expensive localities?":
    st.subheader("Price Trends for the Top 5 most Expensive Localities")
    idx = df.groupby('Locality')['Price_in_Lakhs'].idxmax()
    df10 = df.loc[idx]
    df10 = df10.sort_values(by='Price_in_Lakhs', ascending=False)
    df10=df10[['Locality','Price_in_Lakhs', 'ID', 'Property_Type', 'BHK', 'Size_in_SqFt', 'Year_Built', 'Furnished_Status', 'Availability_Status']]
    st.dataframe(df10.head(5), hide_index=True)

elif option == "11. How are numeric features correlated with each other?":
    st.subheader("Correlation between Numeric Features")
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap( corr, annot=True, cmap='rocket', fmt=".2f" )
    st.pyplot(plt)

elif option == "12. How do nearby schools relate to price per sq ft?":
    st.subheader("Relationship between Nearby Schools and Price per Sqft")
    df12 = df.groupby('Nearby_Schools', as_index=False)['Price_per_SqFt'].mean()
    df12.columns=['No. of Nearby Schools', 'Mean Price per SqFt']
    st.dataframe(df12, hide_index=True)

elif option == "13. How do nearby hospitals relate to price per sq ft?":
    st.subheader("Relationship between Nearby hospitals and Price per Sqft")
    df13 = df.groupby('Nearby_Hospitals', as_index=False)['Price_per_SqFt'].mean()
    df13.columns=['No. of Nearby Hospitals', 'Mean Price per SqFt']
    st.dataframe(df13, hide_index=True)

elif option == "14. How does price vary by furnished status?":
    st.subheader("Relationship between Price and Furnished Status")
    df14 = df.groupby('Furnished_Status', as_index=False)['Price_in_Lakhs'].mean()
    df14.columns=['Furnished Status', 'Mean Price (in Lakhs)']
    st.dataframe(df14, hide_index=True)

elif option == "15. How does price per sq ft vary by property facing direction?":
    st.subheader("Variation of Price per Sqft by Property Facing Direction")
    df15 = df.groupby('Facing', as_index=False)['Price_per_SqFt'].mean()
    df15.columns=['Facing Direction', 'Mean Price per SqFt']
    st.dataframe(df15, hide_index=True)

elif option == "16. How many properties belong to each owner type?":
    st.subheader("No. of Properties belong to each Owner Type")
    df16 = df.groupby('Owner_Type', as_index=False).size()
    df16.columns=['Owner Type', 'No. of Properties']
    st.dataframe(df16, hide_index=True) 

elif option == "17. How many properties are available under each availability status?":
    st.subheader("No. of Properties available under each Availability Status")
    df17 = df.groupby('Availability_Status', as_index=False).size()
    df17.columns=['Availability Status', 'No. of Properties']
    st.dataframe(df17, hide_index=True) 

elif option == "18. Does parking space affect property price?":
    st.subheader("Impact of Parking Space on Property Price")
    df18 = df.groupby('Parking_Space', as_index=False)['Price_in_Lakhs'].median()
    df18.columns=['Parking Space', 'Median Price (in_Lakhs)']
    st.dataframe(df18, hide_index=True)

elif option == "19. How do amenities affect price per sq ft?":
    st.subheader("Impact of Amenities on Price per Sqft")
    df['Amenity_Count'] = df['Amenities'].str.count(',') + 1
    df19 = df.groupby('Amenity_Count', as_index=False)['Price_per_SqFt'].mean()
    df19.columns=['Amenities Count', 'Price_per_SqFt']
    st.dataframe(df19, hide_index=True) 

elif option == "20. How does public transport accessibility relate to price per sq ft?":
    st.subheader("Relation between Public Transport Accessibility and Price per Sqft")
    df20 = df.groupby('Public_Transport_Accessibility', as_index=False)['Price_per_SqFt'].mean()
    df20.columns=['Public_Transport_Accessibility', 'Price_per_SqFt']
    st.dataframe(df20, hide_index=True)