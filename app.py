import streamlit as st
from predict import predict_category  # Import prediction function from the predict module
from db import insert_expense, fetch_all_expenses  # Import DB functions to interact with the database

# Streamlit UI setup
st.set_page_config(page_title="AI Expense Tracker with MySQL 🧠💸")
st.title("AI Expense Tracker with MySQL 🧠💸")

# Input fields for expense description and amount
description = st.text_input("Enter expense description:")
amount = st.number_input("Enter amount", min_value=0.0, step=0.1)

# If the "Predict & Save" button is clicked
if st.button("Predict & Save"):
    if description.strip() == "" or amount <= 0:
        st.warning("Please enter a valid description and amount.")
    else:
        # Predict the category using the description
        category = predict_category(description)
        
        # Insert the expense record into the database
        insert_expense(description, amount, category)
        
        # Show success message and trigger Streamlit balloons animation
        st.success(f"Predicted Category: {category} ✅")
        st.balloons()

# If the "Show All Expenses" button is clicked
if st.button("Show All Expenses"):
    # Fetch all saved expenses from the database
    data = fetch_all_expenses()
    
    # Display the list of expenses
    st.subheader("📊 Saved Expenses:")
    if data:
        for row in data:
            # Display expense details
            st.write(f"{row[3].strftime('%Y-%m-%d %H:%M:%S')} | ₹{row[1]} | {row[0]} → {row[2]}")
    else:
        st.write("No expenses found.")
