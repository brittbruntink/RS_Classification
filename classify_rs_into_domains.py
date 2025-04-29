# -*- coding: utf-8 -*-
"""
Script Name: classify_rs_into_domains.py
Description: This script loads a pre‐trained academic‐domain classifier and its TF-IDF vectorizer,
             then classifies Research Software README files into academic domains. It applies
             text preprocessing, filters out short READMEs (<150 words), uses a decision‐function
             threshold to mark uncertain cases, and outputs both per‐repo domain predictions
             and overall classification statistics.

Author: B.M. Bruntink
Date: 2025-02-27

Dependencies:
    - pandas: For data loading and saving.
    - joblib: For loading the trained model and vectorizer.
    - nltk: For text tokenization and stopword removal.
    - scikit-learn: For TF-IDF transformation and decision‐function usage.
    - re: For regex‐based text cleaning.

Usage:
    python classify_rs_into_domains.py
    Outputs:
      - rs_classified_domains.csv   (Brand Name + Predicted Domain)
      - classification_statistics.csv  (Count & % per domain)
"""

import joblib
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# File paths
BEST_MODEL_FILE = "best_domain_classifier.pkl"
BEST_VECTORIZER_FILE = "best_tfidf_vectorizer.pkl"
INPUT_FILE = "rs_contents.csv"
OUTPUT_FILE = "rs_classified_domains.csv"
STATS_FILE = "classification_statistics.csv"

# Load stopwords
stop_words = set(stopwords.words("english"))

# Load best model and vectorizer
try:
    model = joblib.load(BEST_MODEL_FILE)
    vectorizer = joblib.load(BEST_VECTORIZER_FILE)
    print("✅ Best model and vectorizer loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("Trained model or vectorizer not found. Ensure they are saved correctly.")

# Preprocessing function
def preprocess_text(text):
    """Applies text preprocessing to README content."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# Load dataset
def load_rs_contents():
    """Loads and preprocesses research software README contents."""
    df = pd.read_csv(INPUT_FILE)
    df = df.rename(columns={"Cleaned README Content": "README"})
    
    # Remove README files with fewer than 150 words
    df["Word Count"] = df["README"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    df = df[df["Word Count"] >= 150]
    
    # Preprocess text
    df["Processed README"] = df["README"].apply(preprocess_text)
    
    return df

# Apply classification with confidence threshold using decision function
def classify_readmes(df):
    """Classifies README files into academic domains with a confidence threshold."""
    readme_tfidf = vectorizer.transform(df["Processed README"])
    decision_values = model.decision_function(readme_tfidf)  # Get decision function values
    
    # Define a threshold for uncertainty (e.g., close to 0 indicates uncertainty)
    threshold = 0.25  # You can adjust this threshold based on your experiments
    predicted_domains = []
    
    for value in decision_values:
        # Find the maximum decision value (distance from the decision boundary)
        max_value = max(value)  # The class with the largest decision value
        if abs(max_value) < threshold:  # Close to the decision boundary
            # If the decision function value is close to zero, classify as 'Uncertain'
            predicted_domains.append("Uncertain")
        else:
            predicted_class = model.classes_[value.argmax()]  # Get the predicted class for the largest decision value
            predicted_domains.append(predicted_class)
    
    df["Predicted Domain"] = predicted_domains
    return df

# Calculate classification statistics
def calculate_statistics(df):
    """Calculates statistics for the classification results."""
    domain_counts = df["Predicted Domain"].value_counts()
    total = len(df)
    domain_percentage = (domain_counts / total) * 100
    
    # Create a DataFrame with statistics
    stats = pd.DataFrame({
        "Domain": domain_counts.index,
        "Count": domain_counts.values,
        "Percentage": domain_percentage.values
    })
    
    return stats

# Main function
def main():
    df = load_rs_contents()
    df = classify_readmes(df)
    df[["Brand Name", "Predicted Domain"]].to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Classification completed! Results saved to {OUTPUT_FILE}.")
    
    # Calculate and save classification statistics
    stats = calculate_statistics(df)
    stats.to_csv(STATS_FILE, index=False)
    print(f"✅ Classification statistics saved to {STATS_FILE}.")
    print("\nClassification Statistics:\n", stats)

if __name__ == "__main__":
    main()

