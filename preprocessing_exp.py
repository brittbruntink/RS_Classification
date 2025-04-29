# -*- coding: utf-8 -*-
"""
Script Name: preprocessing_exp.py
Description: This script conducts a preprocessing experiment for the classification of Research Software (RS)
             and Non-Research Software (NRS). It evaluates the impact of different text preprocessing techniques
             (lowercasing, stopword removal, stemming, lemmatization) on classifier performance. For each preprocessing
             strategy, multiple machine learning models (Naive Bayes, Logistic Regression, SVM, Random Forest) are
             trained and optimized using TF-IDF feature extraction with GridSearchCV. The results are evaluated on
             development and test sets and saved for further analysis.

Author: B.M. Bruntink
Date: 2025-02-26

Dependencies:
    - pandas: For dataset loading and result storage.
    - numpy: For numerical operations.
    - nltk: For text preprocessing (tokenization, stopwords, stemming, lemmatization).
    - scikit-learn: For machine learning models, TF-IDF vectorization, GridSearchCV, and evaluation.
    - warnings: To suppress unnecessary output during training.

Usage:
    python preprocessing_exp.py
    This script loads 'combined_dataset.csv', evaluates different preprocessing techniques across classifiers,
    and saves the evaluation results to 'preprocessing_results.xlsx'.
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# File Paths
COMBINED_DATASET_FILE = "combined_dataset.csv"
TEST_DATA_FILE = "test_data.csv"
RESULTS_FILE = "preprocessing_results.xlsx"

# Initialize tools for preprocessing
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define preprocessing function
def preprocess_text(text, lowercase=True, remove_stopwords=False, apply_stemming=False, apply_lemmatization=False):
    """Applies various preprocessing steps to text based on specified options."""
    if not isinstance(text, str):
        return ""

    if lowercase:
        text = text.lower()

    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    if apply_stemming:
        tokens = [stemmer.stem(word) for word in tokens]

    if apply_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# Load dataset
def load_dataset():
    """Loads the dataset and ensures it has the required structure."""
    try:
        df = pd.read_csv(COMBINED_DATASET_FILE)
        print(f"✅ Successfully loaded dataset: {COMBINED_DATASET_FILE}")
        print(f"📊 Dataset contains {len(df)} entries.")

        # Ensure necessary columns exist
        if "Cleaned README Content" not in df.columns or "Label" not in df.columns:
            raise ValueError("Dataset is missing required columns: 'Cleaned README Content' or 'Label'.")

        return df
    except FileNotFoundError:
        print(f"❌ Error: {COMBINED_DATASET_FILE} not found. Please check the file path.")
        exit()
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit()

# Split dataset into train/dev/test
def preprocess_and_split_data(df):
    """Splits the dataset into training (80%), development (10%), and testing (10%) while ensuring stratification."""
    X = df["Cleaned README Content"]
    y = df["Label"]

    # Initial split: 80% train, 20% temp (dev + test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Second split: 10% dev, 10% test (from the temp 20%)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Save test dataset
    test_df = pd.DataFrame({"Cleaned README Content": X_test, "Label": y_test})
    test_df.to_csv(TEST_DATA_FILE, index=False, encoding="utf-8")
    print(f"📂 Test dataset saved to {TEST_DATA_FILE}")

    return X_train, X_dev, X_test, y_train, y_dev, y_test

# Evaluate preprocessing methods and store results
def evaluate_preprocessing_methods(X_train, X_dev, X_test, y_train, y_dev, y_test):
    """Evaluates different preprocessing methods and classifiers using train/dev/test split."""
    results = []
    preprocessing_methods = [
        ("Lowercase Only", {"lowercase": True, "remove_stopwords": False, "apply_stemming": False, "apply_lemmatization": False}),
        ("Remove Stopwords", {"lowercase": True, "remove_stopwords": True, "apply_stemming": False, "apply_lemmatization": False}),
        ("Stemming", {"lowercase": True, "remove_stopwords": False, "apply_stemming": True, "apply_lemmatization": False}),
        ("Lemmatization", {"lowercase": True, "remove_stopwords": False, "apply_stemming": False, "apply_lemmatization": True}),
        ("Lowercase + Stopword Removal", {"lowercase": True, "remove_stopwords": True, "apply_stemming": False, "apply_lemmatization": False}),
        ("Lowercase + Stemming", {"lowercase": True, "remove_stopwords": False, "apply_stemming": True, "apply_lemmatization": False}),
        ("Lowercase + Lemmatization", {"lowercase": True, "remove_stopwords": False, "apply_stemming": False, "apply_lemmatization": True}),
        ("Lowercase + Stopword Removal + Stemming", {"lowercase": True, "remove_stopwords": True, "apply_stemming": True, "apply_lemmatization": False}),
        ("Lowercase + Stopword Removal + Lemmatization", {"lowercase": True, "remove_stopwords": True, "apply_stemming": False, "apply_lemmatization": True})
    ]

    # Define TF-IDF parameter grid for optimization
    tfidf_param_grid = {
        "vectorizer__max_df": [0.75, 0.85, 0.95],
        "vectorizer__min_df": [1, 3, 5],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__max_features": [None, 5000, 10000]
    }

    # Define classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for method_name, method_params in preprocessing_methods:
        print(f"\n🔬 Evaluating preprocessing method: {method_name}")

        # Apply preprocessing
        X_train_processed = X_train.apply(lambda x: preprocess_text(x, **method_params))
        X_dev_processed = X_dev.apply(lambda x: preprocess_text(x, **method_params))
        X_test_processed = X_test.apply(lambda x: preprocess_text(x, **method_params))

        for model_name, model in classifiers.items():
            print(f"\n🛠 Training {model_name} with {method_name} preprocessing")

            # Create pipeline
            pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer()),
                ("classifier", model)
            ])

            # GridSearch on Dev Set
            grid_search = GridSearchCV(pipeline, tfidf_param_grid, cv=3, scoring="accuracy", n_jobs=-1)
            grid_search.fit(X_train_processed, y_train)

            # Best parameters and score
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            dev_accuracy = grid_search.best_score_

            print(f"✅ Best Dev Accuracy: {dev_accuracy:.4f} with params: {best_params}")

            # Final test evaluation
            y_pred = best_model.predict(X_test_processed)
            test_accuracy = accuracy_score(y_test, y_pred)

            print(f"📊 Test Accuracy: {test_accuracy:.4f}")
            print("📜 Classification Report:\n", classification_report(y_test, y_pred))

            # Store results
            results.append({
                "Preprocessing Method": method_name,
                "Model": model_name,
                "Best TF-IDF Params": best_params,
                "Dev Accuracy": dev_accuracy,
                "Test Accuracy": test_accuracy
            })

    # Save results to Excel
    df_results = pd.DataFrame(results)
    df_results.to_excel(RESULTS_FILE, index=False)
    print(f"\n📂 Results saved to {RESULTS_FILE}")

# Run the script
if __name__ == "__main__":
    df_combined = load_dataset()
    X_train, X_dev, X_test, y_train, y_dev, y_test = preprocess_and_split_data(df_combined)
    evaluate_preprocessing_methods(X_train, X_dev, X_test, y_train, y_dev, y_test)
