# -*- coding: utf-8 -*-
"""
Script Name: train_domain_classifier.py
Description: This script trains an academic-domain classifier on research paper abstracts. It loads two TSV files
             containing labeled abstracts, applies preprocessing (lowercasing, stopword removal, optional stemming),
             vectorizes text via TF-IDF (with hyperparameters tuned per model), trains a Random Forest classifier
             (the selected best model), evaluates its performance, and saves the trained model and vectorizer for
             downstream classification of research software domains.

Author: B.M. Bruntink
Date: 2025-03-25

Dependencies:
    - pandas: For data loading and manipulation.
    - numpy: For numerical operations.
    - nltk: For text preprocessing (tokenization, stopwords, stemming).
    - scikit-learn: For TF-IDF vectorization, model training, evaluation, and splitting.
    - joblib: For saving the trained model and vectorizer.

Usage:
    python train_domain_classifier.py
    This will:
      1. Load 'academic_papers.tsv' and 'academic_papers2.tsv'.
      2. Preprocess the 'Abstracts' column.
      3. Vectorize text and train a Random Forest classifier.
      4. Save the best model to 'best_domain_classifier.pkl' and the vectorizer to 'best_tfidf_vectorizer.pkl'.
      5. Output performance metrics to 'model_performance_metrics.csv'.
"""


import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# File paths
BEST_MODEL_FILE = "best_domain_classifier.pkl"
BEST_VECTORIZER_FILE = "best_tfidf_vectorizer.pkl"
RESULTS_FILE = "model_performance_metrics.csv"
CLASSIFIED_READMES_FILE = "classified_readmes.csv"

# Hardcoded best TF-IDF settings
BEST_TFIDF = {
    "Naive Bayes": {"max_df": 0.75, "min_df": 1, "ngram_range": (1, 1), "max_features": None},
    "Logistic Regression": {"max_df": 0.85, "min_df": 3, "ngram_range": (1, 2), "max_features": 5000},
    "SVM": {"max_df": 0.85, "min_df": 5, "ngram_range": (1, 2), "max_features": 10000},
    "Random Forest": {"max_df": 0.75, "min_df": 3, "ngram_range": (1, 1), "max_features": 5000},
}

# Hardcoded best model hyperparameters
BEST_MODEL_HYPERPARAMS = {
    "Naive Bayes": {"alpha": 0.1},
    "Logistic Regression": {"C": 100},
    "SVM": {"C": 1, "loss": "hinge"},
    "Random Forest": {"n_estimators": 100, "max_depth": None, "min_samples_split": 5},
}

# Preprocessing settings
BEST_PREPROCESSING = {
    "Naive Bayes": "Lowercase Only",
    "Logistic Regression": "Lowercase Only",
    "SVM": "Lowercase + Stemming",
    "Random Forest": "Lowercase + Stopword Removal",
}

# Preprocess text based on selected method
def preprocess_text(text, method):
    """Clean and preprocess text based on selected method."""
    if pd.isna(text):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    
    if "Stopword Removal" in method:
        tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    if "Stemming" in method:
        tokens = [PorterStemmer().stem(word) for word in tokens]
    
    return " ".join(tokens)

# Train and evaluate models
def train_and_evaluate_models(papers_df):
    """Train models using best hyperparameters and evaluate them."""
    best_model = None
    best_model_name = None
    best_accuracy = 0
    best_vectorizer = None
    results = []

    # Define models with best hyperparameters
    models = {
        "Naive Bayes": MultinomialNB(**BEST_MODEL_HYPERPARAMS["Naive Bayes"]),
        "Logistic Regression": LogisticRegression(max_iter=1000, **BEST_MODEL_HYPERPARAMS["Logistic Regression"]),
        "SVM": LinearSVC(**BEST_MODEL_HYPERPARAMS["SVM"]),
         "Random Forest": RandomForestClassifier(random_state=42, **BEST_MODEL_HYPERPARAMS["Random Forest"]),
    }

    for model_name, model in models.items():
        print(f"\n🚀 Training {model_name} using hardcoded best TF-IDF and model hyperparameters...")

        # Apply best preprocessing
        papers_df['Processed Abstracts'] = papers_df['Abstracts'].apply(lambda x: preprocess_text(x, BEST_PREPROCESSING[model_name]))

        # Apply best TF-IDF settings
        vectorizer = TfidfVectorizer(**BEST_TFIDF[model_name])
        X = vectorizer.fit_transform(papers_df['Processed Abstracts'])
        y = papers_df['Domain']

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Print all metrics
        print(f"{model_name} - Training Accuracy: {train_accuracy:.4f}")
        print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")
        print(f"{model_name} - Precision: {precision:.4f}")
        print(f"{model_name} - Recall: {recall:.4f}")
        print(f"{model_name} - F1-Score: {f1:.4f}")
        print(f"{model_name} - Confusion Matrix:\n{conf_matrix}")

        # Store results
        results.append({
            "Model": model_name,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Confusion Matrix": conf_matrix
        })

        # Select the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_name = model_name
            best_model = model
            best_vectorizer = vectorizer

    # Save best model and vectorizer for future use
    joblib.dump(best_model, BEST_MODEL_FILE)
    joblib.dump(best_vectorizer, BEST_VECTORIZER_FILE)
    print("\n💾 Best model and vectorizer saved!")

    # Save evaluation results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_FILE, index=False)
    print("\n📊 Model performance results saved to `model_performance_metrics.csv`.")

    return best_model, best_vectorizer, best_model_name

# Main function
def main():
    # Load datasets
    papers1_df = pd.read_csv("academic_papers.tsv", sep="\t", header=None, names=["Labels", "Domain", "Keywords", "Abstracts"])
    papers2_df = pd.read_csv("academic_papers2.tsv", sep="\t", header=None, names=["Labels", "Domain", "Keywords", "Abstracts"])
    papers_df = pd.concat([papers1_df, papers2_df], ignore_index=True)
    papers_df = papers_df[['Abstracts', 'Domain']]
    papers_df.dropna(inplace=True)

    best_model, best_vectorizer, best_model_name = train_and_evaluate_models(papers_df)

if __name__ == "__main__":
    main()
