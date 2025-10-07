# app.py
import joblib
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

# Load model and vectorizer
model = joblib.load('email_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# --- Preprocess Function (same as model.py) ---
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# --- Interactive Console ---
print("\n📬 Email Spam Classifier")
print("Type your email below to test it. Type 'exit' to quit.\n")

label_map = {0: "TRUST", 1: "SPAM"}

while True:
    email = input("Enter email text: ").strip()
    if email.lower() == "exit":
        print("Exiting... 👋")
        break

    clean_email = preprocess_text(email)
    vectorized_email = vectorizer.transform([clean_email])
    prediction = model.predict(vectorized_email)[0]

    print(f"🧠 Prediction: {label_map.get(prediction, prediction)}\n")
    print("Type your email below to test it. Type 'exit' to quit.\n")