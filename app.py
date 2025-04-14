import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load the model and vectorizer
model = joblib.load('expense_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="AI Expense Categorizer", page_icon="💸")
st.title("💸 AI-Based Expense Tracker")
st.markdown("Enter a description and get the predicted category!")

# Input
desc = st.text_input("Enter expense description:")

if st.button("Predict Category"):
    if desc.strip() == "":
        st.warning("Please enter a valid description.")
    else:
        cleaned = clean_text(desc)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)
        st.success(f"Predicted Category: **{prediction[0]}**")
