# model.py
import pandas as pd
import string
import nltk
import time
import joblib
import psutil
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

def log_stage(stage_name):
    elapsed = time.time() - start_time
    mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    print(f"[{stage_name}] ✅ Completed | Time: {elapsed:.2f}s | Memory: {mem:.2f} MB")

start_time = time.time()
print("📊 Email Spam Model Training Tracker\n")

df = pd.read_csv('dataset.csv')
log_stage("Dataset Loaded")

nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess_text)
log_stage("Text Preprocessing Done")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']
log_stage("TF-IDF Vectorization Done")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_stage("Train-Test Split Done")

model = MultinomialNB()
model.fit(X_train, y_train)
log_stage("Model Training Done")

joblib.dump(model, 'email_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
log_stage("Model Saved")

print("\n✅ Model training completed successfully!")
print(f"⏰ Total time taken: {time.time() - start_time:.2f}s")
print("📁 Saved files: email_model.pkl, tfidf_vectorizer.pkl")
