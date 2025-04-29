# -*- coding: utf-8 -*-
"""
Script Name: retrieve_RSD_contents.py
Description: This script retrieves and cleans README contents from GitHub and GitLab repositories
             by fetching the raw content from the provided URLs, cleaning it (removing HTML tags,
             and normalizing whitespace), and filtering based on a minimum word count. The script
             processes a dataset of repositories, reports statistics on successfully fetched
             README contents, and saves the cleaned content to an output CSV file.

Author: B.M. Bruntink
Date: 2025-02-20

Dependencies:
    - requests: A library for making HTTP requests to GitHub and GitLab to fetch README contents.
    - pandas: A library for handling and saving datasets in CSV format.
    - nltk: A library used to download the necessary NLTK resources for text processing.
    - BeautifulSoup (from bs4): A library used to parse and clean HTML content from README files.

Usage:
    python retrieve_RSD_contents.py
    This will process the 'repository_readme_links.csv' input file, clean and filter the README contents, 
    and save the results to 'rs_contents.csv'.
"""
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk

# Download necessary NLTK resources
nltk.download('punkt')

# Minimum word threshold for including README content
MIN_WORD_COUNT = 50  # Adjust if needed

def clean_readme_content(content):
    """
    Cleans the README content by removing HTML tags and normalizing whitespace.

    Args:
        content (str): The raw content of the README file.

    Returns:
        str: The cleaned content with HTML tags removed and whitespace normalized.
    """
    try:
        # Remove HTML tags if present
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        # Normalize whitespace (no stopword removal, no lemmatization, no stemming)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error cleaning content: {e}")
        return None

def fetch_readme_content(url):
    """
    Fetches the raw content of a README file from GitHub or GitLab.

    Args:
        url (str): The URL of the README file.

    Returns:
        str: The raw content of the README file if successful, None otherwise.
    """
    if not url:
        return None
    try:
        headers = {}
        if 'gitlab.com' in url:
            headers = {"PRIVATE-TOKEN": "your_gitlab_token_if_needed"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching README from {url}: {e}")
        return None

def retrieve_readme_contents(input_file, output_file):
    """
    Retrieves and cleans README contents, reports statistics, and saves the final dataset.

    Args:
        input_file (str): The input CSV file containing repository metadata, including README links.
        output_file (str): The output CSV file where cleaned README contents will be saved.
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"File '{input_file}' not found.")
        return

    initial_count = len(df)
    print(f"Initial repositories processed: {initial_count}")

    # Count repositories with a non-empty README Link.
    readme_link_exists = df['README Link'].apply(lambda x: isinstance(x, str) and x.strip() != "")
    count_readme_link_exists = readme_link_exists.sum()
    print(f"Repositories with non-empty README Link: {count_readme_link_exists}")

    # Initialize columns for the fetched content.
    df['Raw README Content'] = None
    df['Cleaned README Content'] = None

    success_count = 0
    for index, row in df.iterrows():
        url = row.get('README Link')
        if pd.notna(url) and isinstance(url, str) and url.strip():
            raw_content = fetch_readme_content(url)
            if raw_content:
                cleaned_content = clean_readme_content(raw_content)
                # Check if the cleaned content meets the minimum word count requirement.
                if cleaned_content and len(cleaned_content.split()) >= MIN_WORD_COUNT:
                    df.at[index, 'Raw README Content'] = raw_content
                    df.at[index, 'Cleaned README Content'] = cleaned_content
                    success_count += 1

    # Filter out repositories that lack valid cleaned README content.
    final_df = df.dropna(subset=["Cleaned README Content"])
    final_df = final_df[final_df["Cleaned README Content"].str.strip() != ""]

    dropped_count = initial_count - len(final_df)
    print(f"Successfully fetched README files: {success_count}")
    print(f"Repositories dropped due to insufficient README content: {dropped_count}")
    print(f"Final dataset size: {len(final_df)} repositories (after filtering).")

    final_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ README contents saved to {output_file}.")

def main():
    """
    Main function to run the entire process of retrieving, cleaning, and saving README contents.
    """
    input_file = 'repository_readme_links.csv'  # Input file containing the README links
    output_file = 'rs_contents.csv'             # Output file for the cleaned README contents
    retrieve_readme_contents(input_file, output_file)

if __name__ == "__main__":
    main()
