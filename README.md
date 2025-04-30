# Research Software Classification Toolkit

*For the MSc thesis of B. M. Bruntink (5169968), MSc Business Informatics, Utrecht University*

---

## 📖 Overview

This repository contains all code, data, and models developed for  
**“A Multimodal Machine Learning Approach for Automated Research Software Classification”**.

It supports:

1. **RS vs. NRS classification**  
   - Traditional (text-only) and multimodal (text + metadata) pipelines  
2. **Domain classification** of research software into academic disciplines  
3. A **final artifact** that, given a GitHub README URL,  
   1. fetches the README,  
   2. retrieves repo metadata,  
   3. classifies as Research vs. Non-Research,  
   4. suggests an academic domain if RS.

---

## 📦 Bill of Materials

| File / Script                                | Purpose                                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Data & Metadata**                          |                                                                                               |
| academic_papers.tsv                          | Paper abstracts + labels, part 1 for domain model training                                    |
| academic_papers2.tsv                         | Paper abstracts + labels, part 2                                                             |
| manual_readme_file.csv                       | Hand-curated overrides for missing README links                                              |
| **Data retrieval & preprocessing**           |                                                                                               |
| retrieve_RSD_contents.py                     | Harvest RSD entries and infer raw README URLs                                                |
| retrieve_links_RSD.py                        | Infer & merge GitHub/GitLab README URLs                                                      |
| retrieve_NRS_contents.py                     | Sample non-research repos from GitHub and fetch their README                                 |
| combine_RS_NRS_datasets.py                   | Merge research & non-research README datasets                                                |
| enrich_dataset_with_metadata.py              | Fetch GitHub/GitLab metadata (language, stars, forks, license) and clean it                  |
| **Exploratory preprocessing**                |                                                                                               |
| preprocessing_exp.py                         | Compare lowercase, stopwords, stemming, etc.                                                 |
| **Model training & evaluation**              |                                                                                               |
| hyperparameter_search_RS_NRS.py              | Grid-search for RS-vs-NRS TF-IDF & model params                                              |
| train_RS_NRS_classifiers_fixed.py            | Train RS-vs-NRS (text-only) with final hyperparams                                           |
| train_RS_NRS_classifiers_mm.py                | Train RS-vs-NRS (multimodal) and save best model/vectorizer                                  |
| train_domain_classifiers.py                  | Train domain classifier on paper abstracts                                                   |
| **Benchmarks vs. ChatGPT**                   |                                                                                               |
| RS_NRS_benchmark_chat.py                     | Compare RS-vs-NRS against GPT-4                                                              |
| RS_domain_benchmark_chat.py                  | Compare domain classifier against GPT-4                                                      |
| **Feature analysis**                         |                                                                                               |
| multimodal_feature_importance.py              | Plot top-k multimodal features (text + metadata)                                             |
| **Final artifact**                           |                                                                                               |
| final_artifact.py                            | CLI tool: input README URL → classify RS/NRS + suggest domain                                |
| **Models & Transformers**                    |                                                                                               |
| best_model_multimodal.pkl                    | Saved RandomForest multimodal RS-vs-NRS model                                                |
| best_vectorizer_multimodal.pkl               | Corresponding TF-IDF vectorizer                                                              |
| domain_model.pkl                             | Saved domain classifier (e.g. SVM)                                                           |
| domain_vectorizer.pkl                        | Domain TF-IDF vectorizer                                                                     |
| metadata_transformer.pkl                     | ColumnTransformer for license & numeric metadata                                             |
| **Results & Reports**                        |                                                                                               |
| combined_dataset.csv                         | Merged README dataset (RS + NRS)                                                             |
| combined_dataset_with_metadata_cleaned.csv   | Merged dataset enriched with metadata                                                        |
| preprocessing_results.xlsx                   | Detailed preprocessing experiment results                                                    |
| **Utilities & Misc**                         |                                                                                               |
| repository_readme_links.csv                  | Raw links from RSD                                                                           |

> _See [Releases](https://github.com/brittbruntink/RS_Classification/releases) for large assets._

---

## 🚀 Getting Started

1. **Clone this repo**  
   ```bash
   git clone https://github.com/brittbruntink/RS_Classification
   cd your-repo

2. ** Install dependencies
   pip install -r requirements.txt

3. Run the final artifact
   python final_artifact.py

   You will be prompted for a GitHub README URL. The script will print:

  - **Research vs. Non-Research**
  - **Suggested academic domain** (if classified as Research Software)

## 📖 Usage Example

```bash
$ python final_artifact.py
Please enter the GitHub repository README URL: https://github.com/uw-comphys/opencmp/blob/main/README.md

✅ Repository 'opencmp' is classified as Research Software.
✅ Suggested Academic Domain: Mechanical Engineering

## 🧑‍💻 Author

**B. M. Bruntink**  
MSc Business Informatics, Utrecht University
