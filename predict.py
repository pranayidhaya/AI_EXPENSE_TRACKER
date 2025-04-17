import joblib
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from db import fetch_all_expenses
from sklearn.linear_model import LinearRegression


nltk.download('stopwords')

# Load trained model & vectorizer
model = joblib.load('expense_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Predict function
def predict_category(description):
    cleaned = clean_text(description)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

def predict_forecast(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.groupby('Date')['Amount'].sum().reset_index()

    # Assign numerical index to dates
    df['DayIndex'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['DayIndex']]
    y = df['Amount']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast next 7 days
    last_day_index = df['DayIndex'].max()
    future_days = np.arange(last_day_index + 1, last_day_index + 8).reshape(-1, 1)
    predicted_amounts = model.predict(future_days)

    future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Amount': predicted_amounts
    })


    return forecast_df

# Version for single category input (per expense forecast)
def predict_forecast_by_category(today_str, category):
    df = pd.DataFrame(fetch_all_expenses(), columns=["Description", "Amount", "Category", "Date"])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df["Category"] == category]

    if df.empty:
        return 0.0

    df['Day'] = (df['Date'] - df['Date'].min()).dt.days
    daily_totals = df.groupby("Day")["Amount"].sum().reset_index()

    X = daily_totals[["Day"]]
    y = daily_totals["Amount"]
    model = LinearRegression()
    model.fit(X, y)

    today = pd.to_datetime(today_str)
    day_index = (today - df['Date'].min()).days
    forecasted_amount = model.predict([[day_index]])[0]

    return float(forecasted_amount)
# Optional CLI
if __name__ == "__main__":
    user_input = input("Enter an expense description: ")
    print("Predicted Category:", predict_category(user_input))
