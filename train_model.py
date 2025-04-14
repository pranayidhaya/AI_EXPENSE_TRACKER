import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

# Load dataset
data = pd.read_csv('expenses.csv')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing
data['cleaned_desc'] = data['Description'].apply(clean_text)

# Features and labels
X = data['cleaned_desc']
y = data['Category']

# Convert text to numerical features
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model trained ✅")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save model and vectorizer
joblib.dump(model, 'expense_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
