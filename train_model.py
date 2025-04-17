import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

# Load dataset
data = pd.read_csv('expenses.csv')

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['cleaned_desc'] = data['Description'].apply(clean_text)

# Features and labels for categorization
X_cat = data['cleaned_desc']
y_cat = data['Category']

# Vectorize the description column using CountVectorizer
vectorizer = CountVectorizer()
X_cat_vec = vectorizer.fit_transform(X_cat)

# Train-test split for categorization model
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat_vec, y_cat, test_size=0.2, random_state=42)

# Model: Naive Bayes for categorization
cat_model = MultinomialNB()
cat_model.fit(X_train_cat, y_train_cat)

# Evaluate categorization model
y_pred_cat = cat_model.predict(X_test_cat)
accuracy_cat = accuracy_score(y_test_cat, y_pred_cat)
print("Categorization Model Accuracy: ", round(accuracy_cat * 100, 2), "%")

# Save the categorization model and vectorizer
joblib.dump(cat_model, 'expense_category_model.pkl')
joblib.dump(vectorizer, 'category_vectorizer.pkl')

# ====== Part 2: Forecasting Model ======

# Preprocess data for forecasting model
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfYear'] = data['Date'].dt.dayofyear  # Feature: Day of the year

# Label encode the categories for forecasting model
encoder = LabelEncoder()
data['CategoryEncoded'] = encoder.fit_transform(data['Category'])

# Features and labels for forecasting
X_forecast = data[['DayOfYear', 'CategoryEncoded']]
y_forecast = data['Amount']

# Train-test split for forecasting model
X_train_forecast, X_test_forecast, y_train_forecast, y_test_forecast = train_test_split(X_forecast, y_forecast, test_size=0.2, random_state=42)

# Model: Linear Regression for forecasting
forecast_model = LinearRegression()
forecast_model.fit(X_train_forecast, y_train_forecast)

# Evaluate forecasting model
y_pred_forecast = forecast_model.predict(X_test_forecast)
forecast_model_score = forecast_model.score(X_test_forecast, y_test_forecast)
print("Forecasting Model R^2 Score: ", round(forecast_model_score, 2))

# Save the forecasting model and encoder
joblib.dump(forecast_model, 'expense_forecasting_model.pkl')
joblib.dump(encoder, 'category_encoder.pkl')

print("Both models (Categorization and Forecasting) have been trained and saved successfully!")