# evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load model and vectorizer
model = joblib.load('email_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load dataset for testing
df = pd.read_csv('dataset.csv')
X = vectorizer.transform(df['text'])
y = df['label']

# Predict
y_pred = model.predict(X)

# Evaluation
print("📊 Model Evaluation Results")
print("-" * 40)
print("Accuracy:", round(accuracy_score(y, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
