# Research Software Classifier

**MSc Thesis by B. M. Bruntink**  
MSc Business Informatics, Utrecht University

A code & data repository for automatically identifying **research software** (RS) vs **non-research software** (NRS) on GitHub, and suggesting academic domain labels.

---

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Installation](#installation)  
4. [Data Files](#data-files)  
5. [Scripts & Notebooks](#scripts--notebooks)  
6. [Running the Artifact](#running-the-artifact)  
7. [Reproducing Experiments](#reproducing-experiments)  
8. [Environment](#environment)  
9. [License](#license)  

---

## Overview

Research software powers modern science but lacks standard discovery & classification. This project:

- Builds a balanced, labeled dataset of RS vs NRS from GitHub.  
- Benchmarks traditional ML classifiers (Naïve Bayes, SVM, Logistic Regression, Random Forest) on README text.  
- Extends classification with a **multimodal** approach (README + GitHub metadata).  
- Trains a separate text-classifier to assign RS to academic domains.  
- Compares performance against ChatGPT baselines.  
- Delivers a stand-alone artifact (`final_artifact.py`) for on-the-fly classification.

---

## Repository Structure

```text
.
├── data/
│   ├── academic_papers.tsv
│   ├── academic_papers2.tsv
│   ├── combined_dataset.csv
│   ├── combined_dataset_with_metadata_cleaned.csv
│   ├── manual_readme_file.csv
│   ├── repository_readme_links.csv
│   └── random_non_research_software_contents.csv
│
├── models/
│   ├── best_model_multimodal.pkl
│   ├── best_vectorizer_multimodal.pkl
│   ├── domain_model.pkl
│   ├── domain_vectorizer.pkl
│   └── metadata_transformer.pkl
│
├── scripts/
│   ├── retrieve_RSD_contents.py
│   ├── retrieve_links_RSD.py
│   ├── retrieve_NRS_contents.py
│   ├── combine_RS_NRS_datasets.py
│   ├── enrich_dataset_with_metadata.py
│   ├── preprocessing_exp.py
│   ├── train_RS_NRS_classifiers_fixed.py
│   ├── hyperparameter_search_RS_NRS.py
│   ├── train_RS_NRS_classifiers_multimodal.py
│   ├── train_domain_classifiers.py
│   ├── extract_multimodal_feature_importance.py
│   ├── RS_NRS_benchmark_chat.py
│   ├── RS_domain_benchmark_chat.py
│   ├── classify_rs_into_domains.py
│   └── final_artifact.py
│
├── results/
│   ├── preprocessing_results.xlsx
│   ├── model_comparison_results_multimodal.csv
│   ├── model_performance_metrics.csv
│   └── confusion_matrices/
│
├── LICENSE
└── README.md






