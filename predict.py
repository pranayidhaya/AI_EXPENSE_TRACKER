import joblib
import re
import nltk
from nltk.corpus import stopwords

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

# Optional CLI
if __name__ == "__main__":
    user_input = input("Enter an expense description: ")
    print("Predicted Category:", predict_category(user_input))
