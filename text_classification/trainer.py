# License: Apache-2.0
 #
 # text_classification/trainer.py: Trainer for Text Classification model in Trainer Studio
 #
 # (C) Copyright 2024 Lithicsoft Organization
 # Author: Bui Nguyen Tan Sang <tansangbuinguyen52@gmail.com>
 #

import re
import string
import pickle
import os
import joblib
import zipfile
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from dotenv import load_dotenv

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Load environment variables
load_dotenv()

# Download necessary resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load hyperparameters from .env
TRAIN_X_PATH = os.getenv("TRAIN_X_PATH", ".\\datasets\\train_X.pkl")
TRAIN_Y_PATH = os.getenv("TRAIN_Y_PATH", ".\\datasets\\train_y.pkl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", ".\\outputs")
MAX_FEATURES = int(os.getenv("MAX_FEATURES", 7000))
NGRAM_RANGE = tuple(map(int, os.getenv("NGRAM_RANGE", "1,3").split(',')))
SOLVER = os.getenv("SOLVER", "saga")
MAX_ITER = int(os.getenv("MAX_ITER", 1500))
REGULARIZATION = float(os.getenv("REGULARIZATION", 2.0))
MULTI_CLASS = os.getenv("MULTI_CLASS", "multinomial")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# -----------------------------
# 1. Load Data
# -----------------------------
with open(TRAIN_X_PATH, "rb") as fx, open(TRAIN_Y_PATH, "rb") as fy:
    train_texts = pickle.load(fx)
    train_labels = pickle.load(fy)

# -----------------------------
# 2. Preprocess Text (Improved)
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"<[^>]+>", " ", text)  # Remove HTML
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization + Stopwords Removal
    text = " ".join(words)
    return text

train_texts_clean = [preprocess_text(text) for text in train_texts]

# -----------------------------
# 3. Feature Extraction (Improved TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    stop_words="english",
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_texts_clean)

# -----------------------------
# 4. Train Optimized Logistic Regression Model
# -----------------------------
clf = LogisticRegression(
    solver=SOLVER,
    max_iter=MAX_ITER,
    C=REGULARIZATION,
    multi_class=MULTI_CLASS,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
clf.fit(X_train, train_labels)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
train_preds = clf.predict(X_train)
train_f1 = f1_score(train_labels, train_preds, average="weighted")
train_acc = accuracy_score(train_labels, train_preds)

print("\n🚀 Model Evaluation on Training Data:")
print(f"✅ Accuracy: {train_acc:.4f}")
print(f"✅ Weighted F1-score: {train_f1:.4f}")
print("\n🔍 Detailed Classification Report:\n", classification_report(train_labels, train_preds))

# -----------------------------
# 6. Save Model & Vectorizer
# -----------------------------
joblib.dump(clf, f"{OUTPUT_DIR}\\logistic_regression_model.pkl")
joblib.dump(vectorizer, f"{OUTPUT_DIR}\\tfidf_vectorizer.pkl")

# Check file sizes
clf_size = os.path.getsize(f"{OUTPUT_DIR}\\logistic_regression_model.pkl") / (1024 * 1024)
vect_size = os.path.getsize(f"{OUTPUT_DIR}\\tfidf_vectorizer.pkl") / (1024 * 1024)
print(f"\n📦 Model File Sizes:")
print(f"   - Logistic Regression model: {clf_size:.2f} MB")
print(f"   - TF-IDF Vectorizer: {vect_size:.2f} MB")
