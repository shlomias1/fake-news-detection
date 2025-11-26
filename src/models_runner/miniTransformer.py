import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# ===================== Paths & Settings =====================
DATA_PATH = Path("/home/shlomias/fake_news_detection/data/df_feat.csv")
TEXT_COL = "text_ns_text"
LABEL_COL = "label"

ARTIFACTS_DIR = Path(os.getenv("FND_MODEL_DIR",
                               "/home/shlomias/fake_news_detection/artifacts_simple"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# use existing TF-IDF vectorizer
TFIDF_PATH = ARTIFACTS_DIR / "tfidf.pkl"
# use existing TF-IDF classifier (SGD)
TFIDF_EXISTING_CLF_PATH = ARTIFACTS_DIR / "sgd_logloss.pkl"

# נבנה תת-תיקייה למודל ה-Late Fusion
LF_DIR = ARTIFACTS_DIR / "late_fusion"
LF_DIR.mkdir(parents=True, exist_ok=True)

EMB_CLF_PATH  = LF_DIR / "clf_emb_sgd.pkl"
META_CLF_PATH = LF_DIR / "clf_meta_logreg.pkl"

TRAIN_EMB_PATH = LF_DIR / "train_embeddings.npy"
TEST_EMB_PATH  = LF_DIR / "test_embeddings.npy"

TEST_METRICS_PATH = LF_DIR / "test_metrics.json"

MINILM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE_EMB = 64

TRAIN_PATH = DATA_PATH.with_name("df_feat_train.csv")
TEST_PATH  = DATA_PATH.with_name("df_feat_test.csv")


# ===================== Train–Test Split =====================
if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    print("Creating train/test files...")
    df = pd.read_csv(DATA_PATH, usecols=[TEXT_COL, LABEL_COL]).dropna()
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[LABEL_COL],
        random_state=42,
    )
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
else:
    print("Using existing train/test CSV files")

train_df = pd.read_csv(TRAIN_PATH).dropna()
test_df  = pd.read_csv(TEST_PATH).dropna()

train_texts = train_df[TEXT_COL].astype(str).tolist()
test_texts  = test_df[TEXT_COL].astype(str).tolist()

train_y = train_df[LABEL_COL].astype(int).values
test_y  = test_df[LABEL_COL].astype(int).values


# ===================== Load existing TF-IDF + classifier =====================
if not TFIDF_PATH.exists():
    raise FileNotFoundError(f"TF-IDF vectorizer not found at {TFIDF_PATH}")

print(f"Loading existing TF-IDF vectorizer from: {TFIDF_PATH}")
tfidf: TfidfVectorizer = joblib.load(TFIDF_PATH)

if not TFIDF_EXISTING_CLF_PATH.exists():
    raise FileNotFoundError(
        f"Existing TF-IDF classifier not found at {TFIDF_EXISTING_CLF_PATH}.\n"
        f"Update TFIDF_EXISTING_CLF_PATH in the script to point to your working model file."
    )

print(f"Loading existing TF-IDF classifier from: {TFIDF_EXISTING_CLF_PATH}")
clf_tfidf: SGDClassifier = joblib.load(TFIDF_EXISTING_CLF_PATH)

print("Transforming TF-IDF (train/test)...")
X_train_tfidf = tfidf.transform(train_texts).astype(np.float32)
X_test_tfidf  = tfidf.transform(test_texts).astype(np.float32)

print("Getting TF-IDF probabilities...")
train_prob_tfidf = clf_tfidf.predict_proba(X_train_tfidf)[:, 1]
test_prob_tfidf  = clf_tfidf.predict_proba(X_test_tfidf)[:, 1]


# ===================== Embeddings (MiniLM) =====================
print(f"Loading MiniLM model: {MINILM_MODEL_NAME}")
sbert = SentenceTransformer(MINILM_MODEL_NAME)

if TRAIN_EMB_PATH.exists() and TEST_EMB_PATH.exists():
    print("Loading saved embeddings...")
    X_train_emb = np.load(TRAIN_EMB_PATH)
    X_test_emb  = np.load(TEST_EMB_PATH)
else:
    print("Encoding MiniLM embeddings (train)...")
    X_train_emb = sbert.encode(
        train_texts,
        batch_size=BATCH_SIZE_EMB,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Encoding MiniLM embeddings (test)...")
    X_test_emb = sbert.encode(
        test_texts,
        batch_size=BATCH_SIZE_EMB,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(TRAIN_EMB_PATH, X_train_emb)
    np.save(TEST_EMB_PATH, X_test_emb)
    print(f"Saved embeddings to:\n  {TRAIN_EMB_PATH}\n  {TEST_EMB_PATH}")


# ===================== Model 2: Embeddings + SGD =====================
if EMB_CLF_PATH.exists():
    print(f"Loading existing embeddings classifier: {EMB_CLF_PATH}")
    clf_emb = joblib.load(EMB_CLF_PATH)
else:
    print("Training embeddings classifier (SGD)...")
    clf_emb = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )
    clf_emb.fit(X_train_emb, train_y)
    joblib.dump(clf_emb, EMB_CLF_PATH)
    print(f"Saved embeddings classifier to: {EMB_CLF_PATH}")

print("Getting embedding-based probabilities...")
train_prob_emb = clf_emb.predict_proba(X_train_emb)[:, 1]
test_prob_emb  = clf_emb.predict_proba(X_test_emb)[:, 1]


# ===================== Meta-Classifier (Late Fusion) =====================
train_meta_X = np.vstack([train_prob_tfidf, train_prob_emb]).T
test_meta_X  = np.vstack([test_prob_tfidf, test_prob_emb]).T

if META_CLF_PATH.exists():
    print(f"Loading existing meta-classifier: {META_CLF_PATH}")
    meta_clf = joblib.load(META_CLF_PATH)
else:
    print("Training meta-classifier (LogisticRegression)...")
    meta_clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    meta_clf.fit(train_meta_X, train_y)
    joblib.dump(meta_clf, META_CLF_PATH)
    print(f"Saved meta-classifier to: {META_CLF_PATH}")

print("Predicting with meta-classifier...")
test_meta_pred = meta_clf.predict(test_meta_X)

test_meta_acc = accuracy_score(test_y, test_meta_pred)
test_meta_f1  = f1_score(test_y, test_meta_pred, average="binary")

print("\n=== Late Fusion (Meta Model) ===")
print(f"TEST Accuracy: {test_meta_acc:.4f}")
print(f"TEST F1:       {test_meta_f1:.4f}")

# מדדי בסיס למודלים לבד
base_tfidf_pred = (test_prob_tfidf >= 0.5).astype(int)
base_emb_pred   = (test_prob_emb >= 0.5).astype(int)

base_tfidf_acc = accuracy_score(test_y, base_tfidf_pred)
base_tfidf_f1  = f1_score(test_y, base_tfidf_pred, average="binary")

base_emb_acc = accuracy_score(test_y, base_emb_pred)
base_emb_f1  = f1_score(test_y, base_emb_pred, average="binary")

print("\n=== Base Models ===")
print(f"TF-IDF model    → Acc: {base_tfidf_acc:.4f}, F1: {base_tfidf_f1:.4f}")
print(f"Embeddings model→ Acc: {base_emb_acc:.4f}, F1: {base_emb_f1:.4f}")

# ===================== Save Metrics =====================
metrics = {
    "late_fusion": {
        "test_accuracy": float(test_meta_acc),
        "test_f1": float(test_meta_f1),
    },
    "tfidf_model": {
        "test_accuracy": float(base_tfidf_acc),
        "test_f1": float(base_tfidf_f1),
    },
    "emb_model": {
        "test_accuracy": float(base_emb_acc),
        "test_f1": float(base_emb_f1),
    },
}

with open(TEST_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved metrics to: {TEST_METRICS_PATH}")
print("\nArtifacts:")
print(f"  TF-IDF vectorizer: {TFIDF_PATH}")
print(f"  Existing TF-IDF classifier: {TFIDF_EXISTING_CLF_PATH}")
print(f"  Embeddings classifier: {EMB_CLF_PATH}")
print(f"  Meta-classifier: {META_CLF_PATH}")
print(f"  Train embeddings: {TRAIN_EMB_PATH}")
print(f"  Test embeddings:  {TEST_EMB_PATH}")
