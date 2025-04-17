import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from predict import predict_category
from db import insert_expense, fetch_all_expenses
from predict import predict_category, predict_forecast, predict_forecast_by_category
 

st.set_page_config(page_title="AI Expense Tracker 🧠💸")
st.title("AI Expense Tracker 🧠💸")

# Input
description = st.text_input("Enter expense description:")
amount = st.number_input("Enter amount", min_value=0.0, step=0.1)

if st.button("Predict & Save"):
    if description.strip() == "" or amount <= 0:
        st.warning("Please enter a valid description and amount.")
    else:
        # Predict the category based on description
        category = predict_category(description)
        
        # Predict forecasted amount based on category (forecasting logic)
        forecasted_amount = predict_forecast_by_category('2025-04-16', category)  # Use the category-based forecast
        
        # Save to database with predicted category
        insert_expense(description, amount, category)
        
        # Show the predicted category and forecasted amount
        st.success(f"Predicted Category: {category} ✅")
        st.write(f"Forecasted Amount for category '{category}': ₹{forecasted_amount:.2f}")
        st.balloons()


all_data = fetch_all_expenses()  # Fetch all expenses data

if st.button("Show All Expenses"):
    data = fetch_all_expenses()
    # Section: 📊 Expense Analysis
st.subheader("📊 Expense Analysis")

data = fetch_all_expenses()

if data:
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Description", "Amount", "Category", "Date"])
    df["Date"] = pd.to_datetime(df["Date"])

    # Pie Chart: Spending by Category
    st.markdown("### 🥧 Spending by Category")
    category_totals = df.groupby("Category")["Amount"].sum().reset_index()
    fig_pie = px.pie(category_totals, names="Category", values="Amount", hole=0.4)
    st.plotly_chart(fig_pie)

    # Line Chart: Spending Over Time
    st.markdown("### 📈 Spending Over Time")
    daily_totals = df.groupby("Date")["Amount"].sum().reset_index()
    fig_line = px.line(daily_totals, x="Date", y="Amount", markers=True)
    st.plotly_chart(fig_line)

    # Optional: Show raw data
    with st.expander("🔍 View Raw Expense Data"):
        st.dataframe(df)
else:
    st.info("No expense data available to analyze.")
    
    st.subheader("🔍 Filter & Analyze Expenses")

if all_data:
    df_all = pd.DataFrame(all_data, columns=["Description", "Amount", "Category", "Date"])
    df_all["Date"] = pd.to_datetime(df_all["Date"])

    # --- Filters ---
    categories = df_all["Category"].unique().tolist()
    selected_category = st.selectbox("Select Category", options=["All"] + categories)

    date_min = df_all["Date"].min()
    date_max = df_all["Date"].max()
    start_date, end_date = st.date_input(
        "Select Date Range",
        [date_min.date(), date_max.date()]
    )

    # --- Apply Filters ---
    filtered_df = df_all.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["Category"] == selected_category]

    filtered_df = filtered_df[
        (filtered_df["Date"] >= pd.to_datetime(start_date)) &
        (filtered_df["Date"] <= pd.to_datetime(end_date))
    ]

    # --- Chart ---
    if not filtered_df.empty:
        st.markdown("### 📊 Filtered Expense Trend:")
        chart = px.bar(filtered_df, x="Date", y="Amount", color="Category", title="Expenses Over Time")
        st.plotly_chart(chart, use_container_width=True)

        st.markdown("### 📋 Filtered Data:")
        st.dataframe(filtered_df)
    else:
        st.info("No expenses found for the selected filters.")


st.subheader("📈 Expense Forecast (Next 7 Days)")

if all_data:
    forecast_df = predict_forecast(df_all)
    st.dataframe(forecast_df)

    fig_forecast = px.line(forecast_df, x='Date', y='Predicted Amount', title="7-Day Expense Forecast")
    st.plotly_chart(fig_forecast)
else:
    st.info("Add expenses to generate a forecast.")

    st.subheader("📊 Saved Expenses:")
if data:
    for row in data:
        # Predict forecasted amount for each saved expense based on category
        forecasted_amount = predict_forecast_by_category('2025-04-16', row[2])  # row[2] is the category
        st.write(f"{row[3].strftime('%Y-%m-%d %H:%M:%S')} | ₹{row[1]:.2f} | {row[0]} → **{row[2]}** | Forecasted Amount: ₹{forecasted_amount:.2f}")
else:
    st.info("No expenses found.")
    

st.subheader("📂 Upload Expenses CSV to Auto-Categorize")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    df_upload = pd.read_csv(uploaded_file)

    # Check if required columns are present
    if "Description" in df_upload.columns and "Amount" in df_upload.columns:
        st.success("CSV loaded successfully! ✅")

        # Predict categories for each row
        df_upload["Predicted Category"] = df_upload["Description"].apply(predict_category)

        # Predict forecasted amounts for each row based on predicted category
        from datetime import datetime
        today_str = datetime.today().strftime('%Y-%m-%d')
        df_upload["Forecasted Amount"] = df_upload["Predicted Category"].apply(
            lambda x: predict_forecast_by_category(today_str, x)
        )

        # Show the updated dataframe with forecasted values
        st.markdown("### 📋 Predicted Results with Forecasts:")
        st.dataframe(df_upload)

        # Button to insert into DB
        if st.button("Save All to Database"):
            for _, row in df_upload.iterrows():
                insert_expense(row["Description"], row["Amount"], row["Predicted Category"])
            st.success("All expenses saved to the database! 🎉")

    else:
        st.error("❌ CSV must have 'Description' and 'Amount' columns.")

