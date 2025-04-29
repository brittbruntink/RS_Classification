# -*- coding: utf-8 -*-
"""
Script Name: combine_RS_NRS_datasets.py
Description: 
    Merge research-software and non-research-software README datasets into a single 
    'combined_dataset.csv' with a proper Label column, ready for model training.

Author: B.M. Bruntink
Date: 2025-04-29
"""
import pandas as pd

# File paths
RESEARCH_FILE = "rs_contents.csv"
NON_RESEARCH_FILE = "random_non_research_software_contents.csv"
OUTPUT_FILE = "combined_dataset.csv"

# Load datasets
research_df = pd.read_csv(RESEARCH_FILE)
non_research_df = pd.read_csv(NON_RESEARCH_FILE)

# Assign labels
research_df["Label"] = "Research"
non_research_df["Label"] = "Non-Research"

# Align columns
common_columns = list(set(research_df.columns) & set(non_research_df.columns))
research_df = research_df[common_columns]
non_research_df = non_research_df[common_columns]

# Merge and save
combined_df = pd.concat([research_df, non_research_df], ignore_index=True)
print(f"🔹 Research: {len(research_df)} | Non-Research: {len(non_research_df)} | Total: {len(combined_df)}")
combined_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"✅ Saved merged dataset to {OUTPUT_FILE}")
