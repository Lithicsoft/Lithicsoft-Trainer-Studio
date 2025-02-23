# License: Apache-2.0
 #
 # text_classification/tester.py: Tester for Text Classification model in Trainer Studio
 #
 # (C) Copyright 2024 Lithicsoft Organization
 # Author: Bui Nguyen Tan Sang <tansangbuinguyen52@gmail.com>
 #

import os
import joblib
import re
import string
from dotenv import load_dotenv

dir_path = os.path.dirname(os.path.realpath(__file__))

# Load environment variables
load_dotenv()

# Load hyperparameters from .env
OUTPUT_DIR = os.getenv("OUTPUT_DIR", f"{dir_path}\\outputs")

# -----------------------------
# Load Saved Model & Vectorizer
# -----------------------------
clf = joblib.load(f"{OUTPUT_DIR}\\logistic_regression_model.pkl")
vectorizer = joblib.load(f"{OUTPUT_DIR}\\tfidf_vectorizer.pkl")

# -----------------------------
# Preprocess Function (Same as Before)
# -----------------------------
def preprocess_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# -----------------------------
# Prediction Function
# -----------------------------
def predict_emotion(text):
    text_clean = preprocess_text(text)
    text_vectorized = vectorizer.transform([text_clean])
    predicted_label = clf.predict(text_vectorized)[0]  # Get single prediction
    return predicted_label

# -----------------------------
# Example Predictions
# -----------------------------
sample_texts = input("Your text: ")

print(f"Text: {text} --> Predicted Emotion: {predict_emotion(sample_texts)}")
