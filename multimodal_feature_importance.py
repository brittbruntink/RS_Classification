# -*- coding: utf-8 -*-
"""
Script Name: extract_multimodal_feature_importance.py
Description: Loads the trained multimodal classifier, TF-IDF vectorizer, and column transformer, then extracts
             and visualizes the top 20 most important text features and top 20 metadata features (license one-hot 
             encodings plus stargazers/forks) that drive the multimodal model’s decisions.

Author: B.M. Bruntink
Date: 2025-04-17

Dependencies:
    - joblib
    - numpy
    - pandas
    - matplotlib
    - scikit-learn
Usage:
    python extract_multimodal_feature_importance.py
"""


import sys
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack

# Suppress warnings
warnings.filterwarnings("ignore")

# File paths
DATASET_FILE = "combined_dataset_with_metadata_cleaned.csv"
BEST_MODEL_FILE = "best_model_multimodal.pkl"
BEST_VECTORIZER_FILE = "best_vectorizer_multimodal.pkl"
TEST_SET_FILE = "test_set.csv"  # File containing the test set brand names

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

# Predefined best TF-IDF settings per classifier
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

        # Ensure 'Raw README Content' is being used as a column
        df = df.drop_duplicates(subset=["Raw README Content"])
        df["Word Count"] = df["Cleaned README Content"].str.split().str.len()
        df = df[df["Word Count"] >= 50]

        # Balance dataset
        research_repos = df[df["Label"] == "Research"]
        non_research_repos = df[df["Label"] == "Non-Research"]
        min_count = min(len(research_repos), len(non_research_repos))
        balanced_df = pd.concat([research_repos.sample(min_count, random_state=42),
                                 non_research_repos.sample(min_count, random_state=42)]).reset_index(drop=True)

        return balanced_df

    except FileNotFoundError:
        print(f"❌ Error: {DATASET_FILE} not found.")
        sys.exit()

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    df['programming_language'] = df['programming_language'].fillna('No Programming Language')
    df['license'] = df['license'].fillna('No License')
    imputer = SimpleImputer(strategy='median')
    df[['stargazers_count', 'forks_count']] = imputer.fit_transform(df[['stargazers_count', 'forks_count']])
    return df

def get_test_set(df, test_set_df):
    """Match the repositories from test_set_df and filter the original df."""
    test_repos = test_set_df['Brand Name'].values
    test_set = df[df['Brand Name'].isin(test_repos)]

    # Ensure test set has exactly 181 instances
    print(f"Test set size before adjustments: {len(test_set)} instances")

    if len(test_set) != 181:
        print(f"❌ Warning: Test set contains {len(test_set)} instances, but 181 are expected.")
        if len(test_set) > 181:
            # Trim test set to 181 instances without removing duplicates
            test_set = test_set.head(181)
            print(f"✅ Test set trimmed to 181 instances.")
        elif len(test_set) < 181:
            # Raise an error and stop execution if test set is too small
            print(f"❌ Error: Test set contains only {len(test_set)} instances, but it should contain exactly 181.")
            sys.exit()  # Exit the script if the test set size is incorrect
    
    else:
        print(f"✅ Test set contains exactly 181 instances.")
    
    return test_set

def print_top_features(feature_importances, feature_names, model_name, top_n=20):
    """Prints top features with their importance values for better interpretability."""
    feature_importance_dict = {feature: importance for feature, importance in zip(feature_names, feature_importances)}
    
    # Sort features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Get top N features
    top_features = sorted_features[:top_n]

    # Print top features with their respective values
    print(f"\nTop {top_n} Features for {model_name}:")
    for feature, importance in top_features:
        print(f"Feature: {feature}, Importance: {importance:.4f}")

def analyze_top_features(model_name, model, column_transformer, top_n=20):
    """Analyze top features based on the model type and print them."""
    if model_name == "Random Forest":
        print(f"\nTop Features ({model_name}):")
        feature_importances = model.feature_importances_
        feature_names = column_transformer.get_feature_names_out()  # For transformed features
        print_top_features(feature_importances, feature_names, model_name)
        
    elif model_name == "Logistic Regression":
        print(f"\nTop Features ({model_name}):")
        coefficients = model.coef_[0]
        feature_names = column_transformer.get_feature_names_out()
        print_top_features(coefficients, feature_names, model_name)

    elif model_name == "SVM":
        print(f"\nTop Features ({model_name}):")
        coefficients = model.coef_[0]
        feature_names = column_transformer.get_feature_names_out()
        print_top_features(coefficients, feature_names, model_name)
    
    elif model_name == "Naive Bayes":
        print(f"\nTop Features ({model_name}):")
        log_likelihoods = model.feature_log_prob_[1]  # Class 1 (Research software) log likelihoods
        feature_names = column_transformer.get_feature_names_out()
        print_top_features(log_likelihoods, feature_names, model_name)

def train_and_evaluate_multimodal_models(X_train, X_test, y_train, y_test):
    """Trains multimodal models using README text content + additional features, and selects the best model."""
    classifiers = {
        "Naive Bayes": MultinomialNB(**BEST_MODEL_HYPERPARAMS["Naive Bayes"]),
        "Logistic Regression": LogisticRegression(max_iter=1000, **BEST_MODEL_HYPERPARAMS["Logistic Regression"]),
        "SVM": LinearSVC(**BEST_MODEL_HYPERPARAMS["SVM"]),
        "Random Forest": RandomForestClassifier(random_state=42, **BEST_MODEL_HYPERPARAMS["Random Forest"]),
    }

    results = []

    for model_name, model in classifiers.items():
        print(f"\n🚀 Training {model_name} using predefined best TF-IDF settings and model hyperparameters...")

        # Apply best preprocessing to the README content
        X_train_processed = X_train["Raw README Content"].apply(lambda x: preprocess_text(x, BEST_PREPROCESSING[model_name]))
        X_test_processed = X_test["Raw README Content"].apply(lambda x: preprocess_text(x, BEST_PREPROCESSING[model_name]))

        # Apply TF-IDF vectorizer to README text
        vectorizer = TfidfVectorizer(**BEST_TFIDF_SETTINGS[model_name])
        X_train_vectorized = vectorizer.fit_transform(X_train_processed)
        X_test_vectorized = vectorizer.transform(X_test_processed)

        # Now include additional features (stargazers_count, forks_count, license)
        feature_columns = ["stargazers_count", "forks_count", "license"]

        column_transformer = ColumnTransformer(
            transformers=[('license', OneHotEncoder(handle_unknown='ignore'), ['license'])],
            remainder='passthrough'
        )

        X_train_features = column_transformer.fit_transform(X_train[feature_columns])
        X_test_features = column_transformer.transform(X_test[feature_columns])

        # Combine the vectorized text and additional features
        X_train_combined = hstack([X_train_vectorized, X_train_features])
        X_test_combined = hstack([X_test_vectorized, X_test_features])

        # Train model
        model.fit(X_train_combined, y_train)

        # Evaluate on training and test set
        y_train_pred = model.predict(X_train_combined)
        y_test_pred = model.predict(X_test_combined)

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

        cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_ if hasattr(model, "classes_") else np.unique(y_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_ if hasattr(model, "classes_") else np.unique(y_test))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.savefig(os.path.join(CONFUSION_MATRIX_DIR, f"{model_name}_confusion_matrix.png"))
        plt.show()

        # Analyze Top Features for interpretability
        analyze_top_features(model_name, model, column_transformer)

    # Return results for further analysis or saving
    return results

def main():
    df = load_and_preprocess_dataset()
    df = handle_missing_values(df)  # Apply the missing value handling

    X = df[["Brand Name", "Raw README Content", "stargazers_count", "forks_count", "license"]]
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train and evaluate multimodal models
    results = train_and_evaluate_multimodal_models(X_train, X_test, y_train, y_test)

    # Optionally, save the results as CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("model_comparison_results_multimodal.csv", index=False)

if __name__ == "__main__":
    main()
