# -*- coding: utf-8 -*-
"""
Script Name: train_RS_NRS_classifiers_multimodal.py
Description: This script trains and evaluates multiple machine learning classifiers (Naive Bayes,
             Logistic Regression, SVM, Random Forest) for the classification of Research Software
             (RS) and Non-Research Software (NRS) using a multimodal approach. The model
             integrates README text features (TF-IDF) with additional GitHub repository metadata
             (stargazers count, forks count, and license information). The script selects the best
             performing classifier and saves it along with its vectorizer and metadata transformer.

Author: B.M. Bruntink
Date: 2025-04-17

Dependencies:
    - pandas: For data loading and processing.
    - numpy: For numerical operations.
    - nltk: For text preprocessing (tokenization, stopwords, stemming).
    - scikit-learn: For ML models, TF-IDF, feature combination, evaluation, and saving.
    - matplotlib: For plotting confusion matrices.
    - joblib: For persisting the trained model and transformers.
    - scipy: For stacking sparse matrices.

Usage:
    python train_RS_NRS_classifiers_multimodal.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Paths & filenames
DATASET_FILE               = "combined_dataset_with_metadata_cleaned.csv"
BEST_MODEL_FILE            = "best_model_multimodal.pkl"
BEST_VECTORIZER_FILE       = "best_vectorizer_multimodal.pkl"
METADATA_TRANSFORMER_FILE  = "metadata_transformer.pkl"   # <— newly added
TEST_SET_FILE              = "test_set.csv"
CONFUSION_MATRIX_DIR       = "confusion_matrices/"
os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)

# NLP tools
stop_words = set(stopwords.words("english"))
stemmer     = PorterStemmer()

# Best preprocessing & hyperparameters (from your experiments)
BEST_PREPROCESSING = {
    "Naive Bayes": "Lowercase Only",
    "Logistic Regression": "Lowercase Only",
    "SVM": "Lowercase + Stemming",
    "Random Forest": "Lowercase + Stopword Removal",
}
BEST_TFIDF_SETTINGS = {
    "Naive Bayes": {"max_df": .75, "min_df": 1, "ngram_range": (1,1), "max_features": None},
    "Logistic Regression": {"max_df": .85, "min_df": 3, "ngram_range": (1,2), "max_features": 5000},
    "SVM": {"max_df": .85, "min_df": 5, "ngram_range": (1,2), "max_features": 10000},
    "Random Forest": {"max_df": .75, "min_df": 3, "ngram_range": (1,1), "max_features": 5000},
}
BEST_MODEL_HYPERPARAMS = {
    "Naive Bayes": {"alpha": 0.1},
    "Logistic Regression": {"C": 100},
    "SVM": {"C":1, "loss":"hinge"},
    "Random Forest": {"n_estimators":100, "max_depth":None, "min_samples_split":5},
}

def preprocess_text(text, method):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    if "Stopword Removal" in method:
        tokens = [w for w in tokens if w not in stop_words]
    if "Stemming" in method:
        tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

def load_and_balance():
    df = pd.read_csv(DATASET_FILE).drop_duplicates(subset=["Raw README Content"])
    df["Word Count"] = df["Cleaned README Content"].str.split().str.len()
    df = df[df["Word Count"]>=50]
    rs  = df[df["Label"]=="Research"]
    nrs = df[df["Label"]=="Non-Research"]
    m   = min(len(rs), len(nrs))
    return pd.concat([rs.sample(m,random_state=42), nrs.sample(m,random_state=42)]).reset_index(drop=True)

def get_test_set(full_df):
    names = pd.read_csv(TEST_SET_FILE)["Brand Name"].values
    test  = full_df[full_df["Brand Name"].isin(names)]
    if len(test)!=181:
        print(f"⚠️ Test set size {len(test)} (expected 181)")
        if len(test)>181: test = test.head(181)
        else: sys.exit("❌ Too few test instances.")
    return test

def train_and_evaluate(train_df, test_df):
    feature_cols = ["stargazers_count","forks_count","license"]
    Xtr, ytr = train_df, train_df["Label"]
    Xte, yte = test_df,  test_df["Label"]

    best_f1 = -1
    best = None

    results = []
    for name, clf in [
        ("Naive Bayes",       MultinomialNB(**BEST_MODEL_HYPERPARAMS["Naive Bayes"])),
        ("Logistic Regression",LogisticRegression(max_iter=1000, **BEST_MODEL_HYPERPARAMS["Logistic Regression"])),
        ("SVM",                LinearSVC(**BEST_MODEL_HYPERPARAMS["SVM"])),
        ("Random Forest",      RandomForestClassifier(random_state=42, **BEST_MODEL_HYPERPARAMS["Random Forest"]))
    ]:
        print(f"\n🚀 Training {name}")
        # Text
        tr_txt = Xtr["Raw README Content"].apply(lambda t: preprocess_text(t, BEST_PREPROCESSING[name]))
        te_txt = Xte["Raw README Content"].apply(lambda t: preprocess_text(t, BEST_PREPROCESSING[name]))
        vec    = TfidfVectorizer(**BEST_TFIDF_SETTINGS[name])
        tr_vec = vec.fit_transform(tr_txt)
        te_vec = vec.transform(te_txt)

        # Metadata
        meta_tf = ColumnTransformer([
            ("license", OneHotEncoder(handle_unknown="ignore"), ["license"])
        ], remainder="passthrough")
        tr_meta = meta_tf.fit_transform(Xtr[feature_cols])
        te_meta = meta_tf.transform(Xte[feature_cols])

        # Combine
        trX = hstack([tr_vec, tr_meta])
        teX = hstack([te_vec, te_meta])

        # Fit & predict
        clf.fit(trX, ytr)
        pred = clf.predict(teX)
        f1   = f1_score(yte, pred, average="macro")
        print(f"{name} → F1: {f1:.4f}")
        results.append((name,f1))

        # Confusion matrix
        cm   = confusion_matrix(yte,pred,labels=clf.classes_)
        disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{name} Confusion Matrix")
        plt.savefig(os.path.join(CONFUSION_MATRIX_DIR, f"{name}_cm.png"))
        plt.close()

        # Track best
        if f1>best_f1:
            best_f1, best = f1, (clf, vec, meta_tf)

    # Save artifacts
    clf, vec, meta_tf = best
    joblib.dump(clf,      BEST_MODEL_FILE)
    joblib.dump(vec,      BEST_VECTORIZER_FILE)
    joblib.dump(meta_tf,  METADATA_TRANSFORMER_FILE)
    print(f"\n✅ Saved best model ({best[0].__class__.__name__}), vectorizer, and metadata transformer.")

    # Results summary
    pd.DataFrame(results, columns=["Model","F1"]).to_csv("multimodal_model_results.csv", index=False)

def main():
    df_all = load_and_balance()
    df_test= get_test_set(df_all)
    df_train = df_all[~df_all["Brand Name"].isin(df_test["Brand Name"])]
    train_and_evaluate(df_train, df_test)

if __name__=="__main__":
    main()


