# -*- coding: utf-8 -*-
"""
Script Name: RS_domain_benchmark_chat.py
Description: Benchmarks GPT-4 on assigning academic domains to research-software README files.
             Reads `rs_contents.csv`, sends each README to GPT-4 with your list of target domains,
             maps GPT’s reply to your standardized codes (CS, Civil, EE, ME, Medical, Psychology,
             Biochemistry, Uncertain), and writes out `chatgpt_domain_results.csv`.

Author: B.M. Bruntink
Date: 2025-02-27

Dependencies:
    - openai
    - pandas
    - nltk
    - re, time, os
Usage:
    export OPENAI_API_KEY="<your key>"
    python RS_domain_benchmark_chat.py
"""
import openai
import pandas as pd
import time
import re
import os
from openai import OpenAIError, RateLimitError, APIConnectionError
import nltk

# Download necessary NLTK resources
nltk.download("punkt")

# OpenAI API key

api_key = "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"
if not api_key:
    raise ValueError("OpenAI API key not found.")
client = openai.Client(api_key=api_key)

def preprocess_text(text):
    """Applies 'Keep Only Alphabetic' preprocessing before passing to ChatGPT."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def classify_with_chatgpt(readme_content, max_length=500):
    """Classifies a README file using OpenAI GPT model into academic domains."""
    preprocessed_content = preprocess_text(readme_content)
    truncated_content = preprocessed_content[:max_length] if isinstance(preprocessed_content, str) else ""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": 
                    "You are a classifier that categorizes software README files into academic domains. "
                    "Classify the following README into one of the following academic domains: Computer Science, Civil Engineering, Electrical Engineering, Mechanical Engineering, Medical Sciences, Psychology, Biochemistry. "
                    "If the domain is unclear or ambiguous, respond with 'Uncertain'."},
                {"role": "user", "content": f"Classify the following README into an academic domain:\n\n{truncated_content}"}
            ],
            max_tokens=100,
            temperature=0
        )
        result = response.choices[0].message.content.strip()

        # Normalize results to match domain names
        if result.lower() in ["computer science", "cs"]:
            return "CS"
        elif result.lower() in ["civil engineering", "civil"]:
            return "Civil"
        elif result.lower() in ["electrical engineering", "ee"]:
            return "EE"
        elif result.lower() in ["mechanical engineering", "me"]:
            return "ME"
        elif result.lower() in ["medical sciences", "medical"]:
            return "Medical"
        elif result.lower() in ["psychology"]:
            return "Psychology"
        elif result.lower() in ["biochemistry", "bio"]:
            return "Biochemistry"
        else:
            return "Uncertain"
    except RateLimitError as e:
        print(f"Rate limit reached: {e}")
        delay = int(e.response.headers.get("Retry-After", 5))
        print(f"Retrying after {delay} seconds...")
        time.sleep(delay)
        return classify_with_chatgpt(readme_content, max_length)
    except APIConnectionError as e:
        print(f"API connection error: {e}")
        return None
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_in_batches(df, batch_size, output_file):
    """Processes the dataset in manageable batches."""
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        print(f"Processing batch {batch_num + 1}/{total_batches}")
        batch = df.iloc[batch_num * batch_size: (batch_num + 1) * batch_size]

        for index, row in batch.iterrows():
            prediction = classify_with_chatgpt(row['Raw README Content'])
            results.append({
                "Brand Name": row['Brand Name'],  # Include Brand Name in the output
                "Raw README Content": row['Raw README Content'],
                "ChatGPT Prediction": prediction
            })

        pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved progress to {output_file}")
        time.sleep(5)  # Brief delay to prevent rate limits

    return results

def main():
    test_data_path = "rs_contents.csv"  # Your dataset
    output_file_chatgpt = "chatgpt_classified_results.csv"
    batch_size = 5

    try:
        df = pd.read_csv(test_data_path)

        required_columns = ['Raw README Content', 'Brand Name']  # Ensure these columns exist
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")

        print(f"Classifying {len(df)} README files using ChatGPT...")

        # Process the dataset with ChatGPT in batches
        process_in_batches(df, batch_size, output_file_chatgpt)

        print(f"Classification complete. Results saved to {output_file_chatgpt}.")
    except FileNotFoundError:
        print(f"File '{test_data_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main()
