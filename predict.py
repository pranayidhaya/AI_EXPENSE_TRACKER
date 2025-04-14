import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load trained model and vectorizer
model = joblib.load('expense_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to clean new input
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Input from user
new_expense = input("Enter an expense description: ")

# Clean and transform
cleaned = clean_text(new_expense)
vector = vectorizer.transform([cleaned])

# Predict
prediction = model.predict(vector)

print(f"Predicted Category: {prediction[0]}")
