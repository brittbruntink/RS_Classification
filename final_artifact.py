#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: final_artifact_multimodal.py
Description: This script classifies a GitHub repository as Research or Non-Research software
             using a multimodal Random Forest model (README text + metadata) and, if it is Research
             software, suggests an academic domain based on README content.
Author: B.M. Bruntink
Date: 2025-04-29

Dependencies:
    - joblib
    - requests
    - nltk
    - beautifulsoup4
    - pandas
    - scipy
Usage:
    python final_artifact_multimodal.py
    (then paste a GitHub repository URL when prompted)
"""

import os
import re
import sys
import joblib
import requests
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import hstack

# Ensure nltk resources are available
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Filepaths for our saved artifacts
RS_MODEL_FILE         = "best_model_multimodal.pkl"
TEXT_VECTORIZER_FILE  = "best_vectorizer_multimodal.pkl"
META_TRANSFORMER_FILE = "metadata_transformer.pkl"
DOMAIN_MODEL_FILE     = "domain_model.pkl"
DOMAIN_VECTORIZER_FILE= "domain_vectorizer.pkl"

# Load GitHub token from environment for metadata requests
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("⚠️  Warning: GITHUB_TOKEN env var not set; metadata requests may be rate-limited.")
GH_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Load our models and transformers
try:
    rs_model        = joblib.load(RS_MODEL_FILE)
    text_vectorizer = joblib.load(TEXT_VECTORIZER_FILE)
    meta_transform  = joblib.load(META_TRANSFORMER_FILE)
    domain_model    = joblib.load(DOMAIN_MODEL_FILE)
    domain_vectorizer = joblib.load(DOMAIN_VECTORIZER_FILE)
except Exception as e:
    print(f"❌ Error loading models or transformers: {e}")
    sys.exit(1)

# Map domain abbreviations to full names
DOMAIN_MAPPING = {
    "CS":         "Computer Science",
    "Civil":      "Civil Engineering",
    "EE":         "Electrical Engineering",
    "ME":         "Mechanical Engineering",
    "Medical":    "Medical Sciences",
    "Psychology":"Psychology",
    "Biochemistry":"Biochemistry",
    "Uncertain":  "Uncertain"
}

# Helpers for README URL inference
README_NAMES = ['README.md','README.rst','README.txt','README','readme.md','readme.rst','readme.txt']
BRANCHES     = ['main','master','develop','gh-pages']

def construct_raw_readme_url(repo_url: str) -> str:
    """
    Try standard GitHub raw URLs for README files on common branches.
    """
    m = re.match(r'https?://github\.com/([^/]+)/([^/]+)', repo_url)
    if not m:
        return ""
    owner, repo = m.group(1), m.group(2).rstrip('.git')
    for branch in BRANCHES:
        for name in README_NAMES:
            candidate = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{name}"
            r = requests.head(candidate, allow_redirects=True)
            if r.status_code == 200:
                return candidate
    return ""

def fetch_readme_text(raw_url: str) -> str:
    """Fetch the raw README content."""
    try:
        r = requests.get(raw_url, timeout=10)
        return r.text if r.status_code == 200 else ""
    except:
        return ""

def fetch_github_metadata(repo_url: str) -> dict:
    """
    Retrieve forks_count, stargazers_count, and license.name from GitHub API.
    """
    m = re.match(r'https?://github\.com/([^/]+)/([^/]+)', repo_url)
    if not m:
        return {"forks_count": 0, "stargazers_count": 0, "license": "No License"}
    owner, repo = m.group(1), m.group(2).rstrip('.git')
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        r = requests.get(api_url, headers=GH_HEADERS, timeout=10)
        data = r.json() if r.status_code == 200 else {}
        return {
            "forks_count":     data.get("forks_count", 0) or 0,
            "stargazers_count":data.get("stargazers_count", 0) or 0,
            "license":         (data.get("license") or {}).get("name", "No License") or "No License"
        }
    except:
        return {"forks_count":0, "stargazers_count":0, "license":"No License"}

def preprocess_text(text: str) -> str:
    """
    Clean HTML, remove URLs, non-alphanumeric, lowercase, remove stopwords.
    """
    # strip HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # remove URLs
    text = re.sub(r'http\S+', '', text)
    # keep letters/numbers/spaces
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text).lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def classify_repository(repo_url: str):
    """
    Main pipeline: fetch README + metadata, vectorize, combine, predict RS/NRS -> domain.
    """
    print(f"\n🔗 Processing repository: {repo_url}")
    raw_readme_url = construct_raw_readme_url(repo_url)
    if not raw_readme_url:
        print("❌ Could not locate a raw README URL.")
        return
    readme_text = fetch_readme_text(raw_readme_url)
    if not readme_text:
        print("❌ Failed to download README.")
        return

    meta = fetch_github_metadata(repo_url)
    clean_text = preprocess_text(readme_text)

    # Text features
    X_text = text_vectorizer.transform([clean_text])
    # Metadata features
    df_meta = pd.DataFrame([{
        "license": meta["license"],
        "stargazers_count": meta["stargazers_count"],
        "forks_count": meta["forks_count"]
    }])
    X_meta = meta_transform.transform(df_meta)

    # Combine
    X_comb = hstack([X_text, X_meta])
    rs_pred = rs_model.predict(X_comb)[0]

    print(f"✅ Classified as: {'Research Software' if rs_pred=='Research' else 'Non-Research Software'}")
    if rs_pred == "Research":
        # Domain prediction (text only)
        dom_vect = domain_vectorizer.transform([clean_text])
        dom_abbr = domain_model.predict(dom_vect)[0]
        print(f"📚 Suggested academic domain: {DOMAIN_MAPPING.get(dom_abbr,'Unknown')}")

if __name__ == "__main__":
    url = input("Enter a GitHub repository URL (e.g. https://github.com/user/repo): ").strip()
    classify_repository(url)
