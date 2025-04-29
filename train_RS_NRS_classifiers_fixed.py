# -*- coding: utf-8 -*-
"""
Script Name: train_RS_NRS_classifiers_fixed.py
Description: This script trains multiple machine learning models (Naive Bayes, Logistic Regression, SVM, Random Forest)
             for the classification of Research Software (RS) and Non-Research Software (NRS) using predefined optimal
             TF-IDF feature extraction settings and model hyperparameters. It evaluates each model, saves the best performing
             model and vectorizer, generates confusion matrices for each classifier, and exports the test set for future use.

Author: B.M. Bruntink
Date: 2025-03-25

Dependencies:
    - pandas: For dataset loading and processing.
    - numpy: For numerical operations.
    - nltk: For text preprocessing (tokenization, stopwords, stemming).
    - scikit-learn: For machine learning models, TF-IDF vectorization, model evaluation, and saving.
    - matplotlib: For plotting confusion matrices.
    - joblib: For saving models and vectorizers.

Usage:
    python train_RS_NRS_classifiers_fixed.py
    This script loads 'combined_dataset.csv', trains classifiers with fixed settings, evaluates them,
    saves the best model ('best_model.pkl') and vectorizer ('best_vectorizer.pkl'), and saves test data
    for external evaluation.
"""
import pandas as pd
import os
import warnings
import json
import nltk
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import re


# Suppress warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# File paths
DATASET_FILE = "combined_dataset.csv"
BEST_MODEL_FILE = "best_model.pkl"
BEST_VECTORIZER_FILE = "best_vectorizer.pkl"
TEST_SET_FILE = "test_set.csv"  # File to save the test set

# Create directory for confusion matrices if not exists
CONFUSION_MATRIX_DIR = "confusion_matrices/"
os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)

# Load stopwords and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Preprocessing strategies per classifier
BEST_PREPROCESSING = {
    "Naive Bayes": "Lowercase Only",
    "Logistic Regression": "Lowercase Only",
    "SVM": "Lowercase + Stemming",
    "Random Forest": "Lowercase + Stopword Removal",
}

# Predefined best TF-IDF settings per classifier (from Table 12)
BEST_TFIDF_SETTINGS = {
    "Naive Bayes": {"max_df": 0.75, "min_df": 1, "ngram_range": (1, 1), "max_features": None},
    "Logistic Regression": {"max_df": 0.85, "min_df": 3, "ngram_range": (1, 2), "max_features": 5000},
    "SVM": {"max_df": 0.85, "min_df": 5, "ngram_range": (1, 2), "max_features": 10000},
    "Random Forest": {"max_df": 0.75, "min_df": 3, "ngram_range": (1, 1), "max_features": 5000},
}

# Predefined best model hyperparameters per classifier
BEST_MODEL_HYPERPARAMS = {
    "Naive Bayes": {"alpha": 0.1},
    "Logistic Regression": {"C": 100},
    "SVM": {"C": 1, "loss": "hinge"},
    "Random Forest": {"n_estimators": 100, "max_depth": None, "min_samples_split": 5},
}

def preprocess_text(text, method):
    """Applies preprocessing including hyperlink removal and optional steps."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)

    # Stopword removal
    if "Stopword Removal" in method:
        tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    if "Stemming" in method:
        tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


def load_and_preprocess_dataset():
    """Loads dataset, removes duplicates, filters short README files, and balances data."""
    try:
        df = pd.read_csv(DATASET_FILE)
        print(f"✅ Successfully loaded dataset: {DATASET_FILE}")

        df = df.drop_duplicates(subset=["Raw README Content"])
        df["Word Count"] = df["Cleaned README Content"].str.split().str.len()
        df = df[df["Word Count"] >= 50]

        # Balance dataset
        research_repos = df[df["Label"] == "Research"]
        non_research_repos = df[df["Label"] == "Non-Research"]
        min_count = min(len(research_repos), len(non_research_repos))
        balanced_df = pd.concat([
            research_repos.sample(min_count, random_state=42),
            non_research_repos.sample(min_count, random_state=42)
        ]).reset_index(drop=True)

        return balanced_df

    except FileNotFoundError:
        print(f"❌ Error: {DATASET_FILE} not found.")
        exit()

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains models using predefined best TF-IDF and hyperparameters, then selects the best model."""
    best_model = None
    best_model_name = None
    best_f1_score = 0
    best_vectorizer = None

    classifiers = {
        "Naive Bayes": MultinomialNB(**BEST_MODEL_HYPERPARAMS["Naive Bayes"]),
        "Logistic Regression": LogisticRegression(max_iter=1000, **BEST_MODEL_HYPERPARAMS["Logistic Regression"]),
        "SVM": LinearSVC(**BEST_MODEL_HYPERPARAMS["SVM"]),
        "Random Forest": RandomForestClassifier(random_state=42, **BEST_MODEL_HYPERPARAMS["Random Forest"]),
    }

    results = []

    for model_name, model in classifiers.items():
        print(f"\n🚀 Training {model_name} using predefined best TF-IDF settings and model hyperparameters...")

        # Apply best preprocessing
        X_train_processed = X_train["Raw README Content"].apply(lambda x: preprocess_text(x, BEST_PREPROCESSING[model_name]))
        X_test_processed = X_test["Raw README Content"].apply(lambda x: preprocess_text(x, BEST_PREPROCESSING[model_name]))

        # Apply best TF-IDF settings
        vectorizer = TfidfVectorizer(**BEST_TFIDF_SETTINGS[model_name])
        X_train_vectorized = vectorizer.fit_transform(X_train_processed)
        X_test_vectorized = vectorizer.transform(X_test_processed)

        # Train model
        model.fit(X_train_vectorized, y_train)

        # Evaluate on training and test set
        y_train_pred = model.predict(X_train_vectorized)
        y_test_pred = model.predict(X_test_vectorized)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average="macro")
        recall = recall_score(y_test, y_test_pred, average="macro")
        f1 = f1_score(y_test, y_test_pred, average="macro")

        results.append({
            "Model": model_name,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        print(f"📊 {model_name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # 🟦 Print and save confusion matrix
        cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_ if hasattr(model, "classes_") else np.unique(y_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_ if hasattr(model, "classes_") else np.unique(y_test))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.savefig(os.path.join(CONFUSION_MATRIX_DIR, f"{model_name}_confusion_matrix.png"))
        plt.show()

        # Select the best model
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = model
            best_model_name = model_name
            best_vectorizer = vectorizer

    # Save the best model and vectorizer
    joblib.dump(best_model, BEST_MODEL_FILE)
    joblib.dump(best_vectorizer, BEST_VECTORIZER_FILE)
    print(f"\n✅ Best Model: {best_model_name} saved to {BEST_MODEL_FILE}")

    # Save results as DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv("model_comparison_results.csv", index=False)

    # Save test set for later use
    X_test[["Brand Name", "Raw README Content"]].to_csv(TEST_SET_FILE, index=False)
    y_test.to_csv("test_labels.csv", index=False)
    print(f"\n✅ Test set saved to {TEST_SET_FILE} and test_labels.csv for future use.")

def main():
    df = load_and_preprocess_dataset()
    X = df[["Brand Name", "Raw README Content"]]
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
