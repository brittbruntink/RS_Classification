# -*- coding: utf-8 -*-
"""
Script Name: hyperparameter_search_RS_NRS.py
Description: This script performs hyperparameter search using GridSearchCV to optimize TF-IDF feature extraction settings
             and machine learning model hyperparameters (Naive Bayes, Logistic Regression, SVM, Random Forest) for the
             classification of Research Software (RS) and Non-Research Software (NRS). It evaluates models on a balanced
             and preprocessed dataset, selects the best configurations, and saves the best model, vectorizer, and
             hyperparameter settings.

Author: B.M. Bruntink
Date: 2025-03-03

Dependencies:
    - pandas: For dataset loading and processing.
    - numpy: For numerical operations.
    - nltk: For text preprocessing (tokenization, stopwords, stemming).
    - scikit-learn: For machine learning models, TF-IDF vectorization, hyperparameter search (GridSearchCV), and evaluation.
    - matplotlib: For plotting (optional, confusion matrices directory created).
    - seaborn: (imported but not used directly — optional).
    - joblib: For saving models and vectorizers.

Usage:
    python hyperparameter_search_RS_NRS.py
    This script loads 'combined_dataset.csv', performs hyperparameter search using GridSearchCV,
    saves the best model ('best_model.pkl'), best TF-IDF vectorizer ('best_vectorizer.pkl'),
    and exports the best hyperparameter settings to 'best_tfidf_and_model_hyperparams.xlsx'.
"""

import pandas as pd
import os
import warnings
import json
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Suppress warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# File paths
DATASET_FILE = "combined_dataset.csv"
RESULTS_FILE = "best_tfidf_and_model_hyperparams.xlsx"
BEST_MODEL_FILE = "best_model.pkl"
BEST_VECTORIZER_FILE = "best_vectorizer.pkl"
BEST_MODEL_HYPERPARAMS_FILE = "best_model_hyperparams.json"

# Create directory for confusion matrices
CONFUSION_MATRIX_DIR = "confusion_matrices/"
if not os.path.exists(CONFUSION_MATRIX_DIR):
    os.makedirs(CONFUSION_MATRIX_DIR)

# Load stopwords and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Model-specific preprocessing strategies
BEST_PREPROCESSING = {
    "Naive Bayes": "Lowercase Only",
    "Logistic Regression": "Lowercase Only",
    "SVM": "Lowercase + Stemming",
    "Random Forest": "Lowercase + Stopword Removal",
}

def preprocess_text(text, method):
    """Applies preprocessing based on the classifier's optimal settings."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)

    if "Stopword Removal" in method:
        tokens = [word for word in tokens if word not in stop_words]

    if "Stemming" in method:
        tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

# Load and preprocess dataset
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

# Train, find best TF-IDF settings and model hyperparameters
def train_and_evaluate_models(X_train, X_dev, X_test, y_train, y_dev, y_test):
    """Finds best TF-IDF and model hyperparameters, evaluates models, and saves the best-performing one."""
    results = []
    best_model = None
    best_model_name = None
    best_f1_score = 0
    best_model_hyperparams = {}  # ✅ Stores both TF-IDF settings AND model hyperparameters

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    # Define TF-IDF and model hyperparameter grid
    param_grids = {
        "Naive Bayes": {"classifier__alpha": [0.1, 0.5, 1.0, 2.0, 5.0]},
        "Logistic Regression": {"classifier__C": [0.1, 1, 10, 100]},
        "SVM": {"classifier__C": [0.1, 1, 10], "classifier__loss": ["hinge", "squared_hinge"]},
        "Random Forest": {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None, 10, 20], "classifier__min_samples_split": [2, 5]},
    }

    tfidf_param_grid = {
        "vectorizer__max_df": [0.75, 0.85, 0.95],
        "vectorizer__min_df": [1, 3, 5],
        "vectorizer__ngram_range": [(1, 1), (1,2), (1,3)],
        "vectorizer__max_features": [5000, 10000, None]
    }

    for model_name, model in classifiers.items():
        print(f"\n🚀 Training {model_name} using GridSearchCV...")

        # Apply best preprocessing
        X_train_processed = X_train["Raw README Content"].apply(lambda x: preprocess_text(x, BEST_PREPROCESSING[model_name]))
        X_dev_processed = X_dev["Raw README Content"].apply(lambda x: preprocess_text(x, BEST_PREPROCESSING[model_name]))
        X_test_processed = X_test["Raw README Content"].apply(lambda x: preprocess_text(x, BEST_PREPROCESSING[model_name]))

        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", model)
        ])

        # Combine TF-IDF and model hyperparameters
        full_param_grid = {**tfidf_param_grid, **param_grids[model_name]}

        grid_search = GridSearchCV(pipeline, full_param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train_processed, y_train)

        best_model_current = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_dev_accuracy = grid_search.best_score_

        print(f"✅ Best TF-IDF & Model settings for {model_name}: {best_params} (Dev Accuracy: {best_dev_accuracy:.4f})")

        # Store best parameters
        best_model_hyperparams[model_name] = best_params  # ✅ Stores TF-IDF & model hyperparameters

        # Evaluate on test set
        y_pred = best_model_current.predict(X_test_processed)
        f1 = f1_score(y_test, y_pred, average="macro")

        results.append({
            "Model": model_name,
            "Best Hyperparameters": best_params,
            "Dev Accuracy": best_dev_accuracy,
            "F1-Score": f1
        })

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = best_model_current
            best_model_name = model_name

    # Save best model, vectorizer, and hyperparameters
    joblib.dump(best_model, BEST_MODEL_FILE)
    joblib.dump(best_model.named_steps["vectorizer"], BEST_VECTORIZER_FILE)

    with open(BEST_MODEL_HYPERPARAMS_FILE, "w") as f:
        json.dump(best_model_hyperparams, f, indent=4)  # ✅ Save best TF-IDF & model settings

    df_results = pd.DataFrame(results)
    df_results.to_excel(RESULTS_FILE, index=False)

# ✅ Main function
def main():
    df = load_and_preprocess_dataset()
    X = df[["Brand Name", "Raw README Content"]]
    y = df["Label"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    train_and_evaluate_models(X_train, X_dev, X_test, y_train, y_dev, y_test)

if __name__ == "__main__":
    main()
