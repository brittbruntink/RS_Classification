# -*- coding: utf-8 -*-
"""
Script Name: enrich_dataset_with_metadata.py
Description: This script enriches the research software dataset by retrieving repository metadata (programming language, stargazers count,
             forks count, license) from GitHub and GitLab APIs based on README links. It handles missing values through manual additions,
             default values, and median imputation, and saves a cleaned combined dataset ready for multimodal classification.

Author: B.M. Bruntink
Date: 2025-04-17

Dependencies:
    - pandas: For dataset manipulation and saving.
    - requests: For API calls to GitHub and GitLab.
    - scikit-learn: For simple imputation of missing numeric metadata.
    - nltk: For text normalization (used lightly for tokenization).

Usage:
    python enrich_dataset_with_metadata.py
    This script reads 'combined_dataset.csv', retrieves and fills metadata, and saves the result as 'combined_dataset_with_metadata_cleaned.csv'.
"""


import pandas as pd
import requests
from sklearn.impute import SimpleImputer

# Load your existing dataset
df = pd.read_csv("combined_dataset.csv")

# Initialize counters for missing values and imputations
missing_programming_language_count = 0
missing_license_count = 0
missing_stargazers_count_count = 0
missing_forks_count_count = 0

# Manually adding metadata for the missing repos
manual_metadata = {
    " PanTools": {
        "programming_language": "No Programming Language",
        "stargazers_count": 5,
        "forks_count": 6,
        "license": "GNU GPLv3"
    },
    "Rucio deployment for KM3NeT": {
        "programming_language": "No Programming Language",
        "stargazers_count": 0,
        "forks_count": 0,
        "license": "Apache License 2.0"
    },
    "DuMux": {
        "programming_language": "C++",
        "stargazers_count": 35,
        "forks_count": 14,
        "license": "GNU GPL version 3"
    },
    "Foot Progression Angle Estimation": {
        "programming_language": "MATLAB",
        "stargazers_count": 0,
        "forks_count": 0,
        "license": "CC-BY-NC-SA-1.0"
    }
}

# Function to retrieve metadata for GitHub repositories
def get_github_metadata(owner, repo_name, token):
    url = f"https://api.github.com/repos/{owner}/{repo_name}"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        metadata = {
            "programming_language": data.get('language', None),
            "stargazers_count": data.get('stargazers_count', None),
            "forks_count": data.get('forks_count', None),
            "license": data.get('license', {}).get('name', None) if data.get('license') else None,
        }
        return metadata
    else:
        return None

# Function to retrieve metadata for GitLab repositories
def get_gitlab_metadata(owner, repo_name, token):
    url = f"https://gitlab.com/api/v4/projects/{owner}%2F{repo_name}"
    headers = {"PRIVATE-TOKEN": token}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        metadata = {
            "programming_language": data.get('language', None),
            "stargazers_count": data.get('star_count', None),  # GitLab uses 'star_count' instead of 'stargazers_count'
            "forks_count": data.get('forks_count', None),
            "license": data.get('license', {}).get('name', None) if data.get('license') else None,
        }
        return metadata
    else:
        return None

# Function to parse the owner and repo name from the raw README URL (handles both GitHub and GitLab)
def parse_repo_url_from_readme_link(url):
    url = url.strip()

    # Check if the URL is a GitHub raw file URL
    if 'raw.githubusercontent.com' in url:
        parts = url.split('raw.githubusercontent.com/')[1].split('/')
        if len(parts) >= 3:
            owner = parts[0]
            repo_name = parts[1]
            return owner, repo_name, "github"
    # Check if the URL is a GitLab raw file URL
    elif 'gitlab.com' in url:
        parts = url.split('gitlab.com/')[1].split('/')
        if len(parts) >= 2:
            owner = parts[0]
            repo_name = parts[1]
            return owner, repo_name, "gitlab"
    
    return None, None, None

# Add new metadata columns to your DataFrame (without contributors_count)
metadata_columns = ["programming_language", "stargazers_count", "forks_count", "license"]
for column in metadata_columns:
    df[column] = None  # Initialize the metadata columns

# Replace 'your_github_token_here' and 'your_gitlab_token_here' with actual tokens
github_token = "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"
gitlab_token = "glpat-gywy8zoCzUPa2DWUqGZ7"

# Retrieve metadata for each repository based on the README link
for index, row in df.iterrows():
    readme_url = row['README Link']
    owner, repo_name, platform = parse_repo_url_from_readme_link(readme_url)
    
    # Manually add metadata for repositories that failed to retrieve metadata
    brand_name = row['Brand Name']
    if brand_name in manual_metadata:
        metadata = manual_metadata[brand_name]
        for column, value in metadata.items():
            df.at[index, column] = value
        print(f"Manually added metadata for {brand_name}")

    elif owner and repo_name:
        if platform == "github":
            metadata = get_github_metadata(owner, repo_name, github_token)
        elif platform == "gitlab":
            metadata = get_gitlab_metadata(owner, repo_name, gitlab_token)
        
        if metadata:
            for column, value in metadata.items():
                df.at[index, column] = value
    else:
        print(f"Skipping invalid README Link for {row['Brand Name']}")

# Handle missing values in metadata and count missing values
# Track missing data for 'programming_language' and 'license'
missing_programming_language_count = df['programming_language'].isna().sum()
missing_license_count = df['license'].isna().sum()

# Fill missing values for 'programming_language' and 'license' with 'No Programming Language' and 'No License'
df['programming_language'] = df['programming_language'].fillna('No Programming Language')
df['license'] = df['license'].fillna('No License')

# Handle missing numerical data ('stargazers_count', 'forks_count')
missing_stargazers_count_count = df['stargazers_count'].isna().sum()
missing_forks_count_count = df['forks_count'].isna().sum()

# Impute missing values for 'stargazers_count' and 'forks_count' with the median
imputer = SimpleImputer(strategy='median')
df[['stargazers_count', 'forks_count']] = imputer.fit_transform(df[['stargazers_count', 'forks_count']])

# Print the number of times imputation was used
print(f"Missing 'programming_language' values filled: {missing_programming_language_count}")
print(f"Missing 'license' values filled: {missing_license_count}")
print(f"Missing 'stargazers_count' values imputed: {missing_stargazers_count_count}")
print(f"Missing 'forks_count' values imputed: {missing_forks_count_count}")

# Save the updated DataFrame with metadata
df.to_csv("combined_dataset_with_metadata_cleaned.csv", index=False)

# Print statistics on the metadata
def collect_metadata_statistics(df):
    metadata_stats = {
        "Total Repositories": len(df),
        "Missing programming_language": df["programming_language"].isna().sum(),
        "Missing license": df["license"].isna().sum(),
        "Missing stargazers_count": df["stargazers_count"].isna().sum(),
        "Missing forks_count": df["forks_count"].isna().sum(),
    }

    # Calculate percentage of missing values
    metadata_stats["Missing programming_language (%)"] = (metadata_stats["Missing programming_language"] / len(df)) * 100
    metadata_stats["Missing license (%)"] = (metadata_stats["Missing license"] / len(df)) * 100
    metadata_stats["Missing stargazers_count (%)"] = (metadata_stats["Missing stargazers_count"] / len(df)) * 100
    metadata_stats["Missing forks_count (%)"] = (metadata_stats["Missing forks_count"] / len(df)) * 100

    print(f"Metadata Statistics: {metadata_stats}")

# Collect and print metadata statistics
collect_metadata_statistics(df)
