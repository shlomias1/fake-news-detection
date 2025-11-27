
-----

# Fake News Detection

This repository contains code for detecting fake news using machine learning and natural language processing (NLP) techniques. The project leverages multiple datasets and a modular code structure to preprocess, analyze, and classify news content.

-----

## Project Structure

```
fake-news-detection/
│
├── data/                   # Dataset files (ignored in Git)
├── artifacts_simple/       # Trained models, vectorizers, and metrics (e.g., tfidf.pkl)
├── logs/                   # Training logs
├── models/                 # Trained models or saved checkpoints
├── src/                    # Core source code
│   └── app.py              # FastAPI + UI for prediction and explanations
│   └── model_runner/              
│       └── tfidf_classifier.py # Training script for TF-IDF model
│       └── miniTransformer.py  # Training script for MiniLM/Late Fusion models
│   └── ...
├── utils/                  # Utility modules (e.g., text preprocessing)
│   └── feature_utils.py
│   └── plloting.py
│   └── preprocessing_utils.py
│   └── text_utils.py
├── predictions/            # Logs of predictions (JSONL)
├── .venv/                  # Virtual environment (ignored)
├── main.py                 # Main entry script (deprecated/legacy)
├── config.py               # Configuration variables
├── data_io.py              # Functions to load/save data
├── preprocessing.py        # **[Step 1] Initial data consolidation and cleaning**
├── processing.py           # **[Step 2] Feature engineering pipeline**
├── EDA.ipynb               # Exploratory Data Analysis notebook
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

-----

## Features

  * **Modular Data Pipeline (Polars/Pandas):**
      * **`preprocessing.py`**: Initial consolidation of 10+ raw datasets into a unified format (`fake_news_combined_dataset.csv`).
      * **`processing.py`**: A dedicated feature engineering pipeline (`feat_pipeline`) to generate statistical and textual features (`df_feat.csv`), including **Token Ratio (TTR)**, **Stopword Ratio**, **Punctuation counts**, and **Jaccard Similarity** between title and text.
  * **Model Training Pipelines:** Separate scripts for training the core models:
      * `tfidf_classifier.py` (for the TF-IDF-based model).
      * `miniTransformer.py` (for the Sentence Transformer and Late Fusion models).
  * **Web UI for Prediction (FastAPI):**
      * Paste article text, classify using the trained model.
      * Choose explainer (**LIME** or **SHAP**).
      * View token-level contribution weights.
      * Optionally auto-translate before classification.
  * **Prediction Logging:** Every prediction is appended to `predictions/preds.jsonl` (Timestamp, input text, result, explanation info).
  * Modular architecture (`utils/`, `config.py`) and easy dependency management (`requirements.txt`).

-----

## Getting Started

### 1\. Clone the repository

```bash
git clone https://github.com/shlomias1/fake-news-detection.git
cd fake-news-detection
```

### 2\. Create and activate a virtual environment

```bash
python -m venv .venv
# Activate on Windows (PowerShell):
.venv\Scripts\Activate
# Or on Linux/macOS:
source .venv/bin/activate
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

### 4\. Full Training Pipeline

The full training process involves three main stages:

#### 4.1. Data Preparation and Feature Engineering

Run the main function in `preprocessing.py` to consolidate datasets, and then run `processing.py` to generate the final feature file (`data/df_feat.csv`).

```bash
python preprocessing.py 
python processing.py 
```

#### 4.2. Train Base Models

Run the dedicated scripts to train the initial models.

```bash
# Trains the TF-IDF model and saves it to artifacts_simple/sgd_logloss.pkl
python src/tfidf_classifier.py

# Trains the MiniLM Embeddings model and the Meta-Classifier (Late Fusion)
python src/miniTransformer.py
```

### 4.3. Run FastAPI UI

Once models are trained and saved in `artifacts_simple/`, start the web service.

```bash
python src/app.py
```

Then open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000) to test predictions.

-----

## Datasets

Due to size limitations, dataset files (e.g., `.csv`, `.npy`) are not included in this repository.
Please download them manually and place them under the `data/` directory. Example datasets used:

  * [FakeNewsDataset](https://www.kaggle.com/datasets/mrisdal/fake-news)
  * [PHEME dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_rumour_detection_and_veracity_classification/6392072)
  * [BanFakeNews](https://github.com/AmanPriyanshu/BanFakeNews)

-----

## Contact

Built by [shlomias1](https://github.com/shlomias1)

-----
