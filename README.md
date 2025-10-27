# 📧 SENTINEL: Email Spam Classifier (Python + scikit-learn)

## Overview
This project is a **Machine Learning-based Email Spam Classifier** using Python and scikit-learn.  
It can classify emails as **Spam** or **Non-Spam (Trust/Ham)** based on text content.  

The project is divided into three parts:

1. **model.py** → Trains the model and saves it (`.pkl` files).  
2. **evaluate.py** → Evaluates the trained model on the dataset.  
3. **app.py** → Interactive console app to test custom emails.

---

## Features
- Text preprocessing: lowercase, punctuation removal, stopwords removal  
- Feature extraction using **TF-IDF Vectorizer**  
- Trained **Naive Bayes (MultinomialNB)** classifier  
- Interactive testing for custom emails  
- Saves model and vectorizer using `.pkl` files for reuse  
- Model training tracker with time and memory usage  

---

## Dataset
This project uses the **Email Spam Classification Dataset** available on Kaggle: [Dataset Link](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset?resource=download)

- Must be a CSV file with **two columns**:
  - `text` → Email content  
  - `label` → 0/1 or `ham/spam` depending on dataset  
- Example:
| text                                      | label |
|------------------------------------------|-------|
| "You won a free iPhone!"                  | spam  |
| "Hey, are we meeting tomorrow?"           | ham   |

---

## Folder Structure
```
email-spam-classifier/
│
├── dataset.csv          # CSV dataset of emails with 'text' and 'label' columns
├── model.py             # Training script
├── evaluate.py          # Model evaluation script
├── app.py               # Interactive email classifier
├── email_model.pkl      # Saved trained model (generated)
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer (generated)
└── README.md            # Project documentation
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IndronathBasu/Spam-Email-Classifier.git
cd spam-email-classifier
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn nltk joblib psutil
```

3. Download NLTK stopwords (done automatically in the scripts).

---

## Usage

### 1️⃣ Train the Model
```bash
python model.py
```
- This script will train the **Naive Bayes model** on your dataset.  
- Saves `email_model.pkl` and `tfidf_vectorizer.pkl`.

---

### 2️⃣ Evaluate the Model
```bash
python evaluate.py
```
- Prints **Accuracy**, **Confusion Matrix**, and **Classification Report**.  
- Helps check model performance.

---

### 3️⃣ Interactive Testing
```bash
python app.py
```
- Type any email text to check if it’s **Spam** or **Trust/Ham**.  
- Type `exit` to quit.

**Example:**
```
Enter email text: You have won a $1000 gift card!
🧠 Prediction: SPAM

Enter email text: Let’s meet at 3pm for the project discussion.
🧠 Prediction: TRUST
```

---

## Notes
- The project uses **`.pkl` files** to save the trained model and vectorizer.  
- Use **`joblib.load()`** to reload them without retraining.  
- The `model.py` script includes a **training tracker** showing time and memory usage.  

---

## Future Improvements
- Use **Logistic Regression** or **Deep Learning** models for better accuracy  
- Implement a **GUI** using Tkinter or **Web App** using Flask/Streamlit  
- Add **more preprocessing** (lemmatization, handling emojis, etc.)  
- Deploy as a **REST API** for real-time spam detection  

---

## License
This project is made by Indronath Basu
