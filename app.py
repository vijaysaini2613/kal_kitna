import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("output/predictions_2025.csv")

df = load_data()

# Sidebar filters
st.sidebar.title("🔍 Filters")
selected_city = st.sidebar.selectbox("Select City", df["warehouse_location"].unique())
selected_month = st.sidebar.selectbox("Select Month", sorted(df["month_number"].unique()))
top_n = st.sidebar.slider("Top N Products", min_value=1, max_value=20, value=5)

# Filtered Data
filtered_df = df[(df["warehouse_location"] == selected_city) & (df["month_number"] == selected_month)]

# Title
st.title("📦 Smart Warehouse Demand Prediction Dashboard")
st.subheader(f"📍 City: {selected_city} | 📅 Month: {selected_month}")

# Top N Products Bar Chart
product_columns = df.columns[3:]
top_products = filtered_df[product_columns].T
top_products.columns = ["Predicted Demand"]
top_products = top_products.sort_values(by="Predicted Demand", ascending=False).head(top_n)

st.markdown(f"### 🔝 Top {top_n} Products")
fig, ax = plt.subplots()
sns.barplot(y=top_products.index, x=top_products["Predicted Demand"], ax=ax, palette="viridis")
ax.set_xlabel("Predicted Units Sold")
st.pyplot(fig)

# Full Table View
st.markdown("### 📋 Full Prediction Table (Filtered)")
st.dataframe(filtered_df)
