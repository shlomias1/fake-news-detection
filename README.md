
---

# Fake News Detection

This repository contains code for detecting fake news using machine learning and natural language processing (NLP) techniques. The project leverages multiple datasets and modular code structure to preprocess, analyze, and classify news content.

---

## Project Structure

```
fake-news-detection/
│
├── data/                   # Dataset files (ignored in Git)
├── models/                 # Trained models or saved checkpoints
├── utils/                  # Utility modules (e.g., text preprocessing)
│   └── feature_utils.py
│   └── plloting.py
│   └── preprocessing_utils.py
│   └── text_utils.py
├── predictions/            # Logs of predictions (JSONL)
├── .venv/                  # Virtual environment (ignored)
├── main.py                 # Main script to run training or evaluation
├── app.py                  # FastAPI + UI for prediction and token-level explanations
├── config.py               # Configuration variables
├── data_io.py              # Functions to load/save data
├── preprocessing.py        # Data preprocessing logic
├── processing.py           # Main processing pipeline
├── EDA.ipynb               # Exploratory Data Analysis notebook
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Features

* Preprocessing pipelines for structured and unstructured datasets
* Modular architecture (e.g., `utils/`, `config.py`)
* Jupyter notebook for exploratory data analysis
* Support for multiple datasets: PHEME, FakeNewsDataset, BanFakeNews, and more
* `.gitignore` configured to avoid uploading heavy datasets
* Ready for model training and evaluation pipelines
* **Web UI for prediction**:

  * Paste article text
  * Choose explainer (LIME or SHAP)
  * Optionally auto-translate before classification
  * View token-level contribution weights
* **Logging predictions**: Every prediction is appended to `predictions/preds.jsonl` with:

  * Input text
  * Result (`FAKE` / `REAL`, probability, token weights, translation info)
  * Timestamp

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shlomias1/fake-news-detection.git
cd fake-news-detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Or on Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies

> If `requirements.txt` doesn't exist yet, create it with your current environment or use `pip freeze > requirements.txt`

```bash
pip install -r requirements.txt
```

### 4. Run the project

**Option 1: Run FastAPI UI**

```bash
python app.py
```

Then open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000) to test predictions.

**Option 2: Run main script**

```bash
python main.py
```

Or explore the dataset in the notebook:

```bash
jupyter notebook EDA.ipynb
```

---

## Datasets

Due to size limitations, dataset files (e.g., `.csv`, `.npy`) are not included in this repository.
Please download them manually and place them under the `data/` directory. Example datasets used:

* [FakeNewsDataset](https://www.kaggle.com/datasets/mrisdal/fake-news)
* [PHEME dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_rumour_detection_and_veracity_classification/6392072)
* [BanFakeNews](https://github.com/AmanPriyanshu/BanFakeNews)

---

## Contact

Built by [shlomias1](https://github.com/shlomias1)

---
רוצה שאעשה את זה?
