import os
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, hstack

# ==== File Settings ====
DATA_PATH = Path("/home/shlomias/fake_news_detection/data/df_feat.csv")
TEXT_COL = "text_ns_text"
LABEL_COL = "label"

ARTIFACTS_DIR = Path(os.getenv("FND_MODEL_DIR", "/home/shlomias/fake_news_detection/artifacts_simple"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_PATH = ARTIFACTS_DIR / "tfidf.pkl"
COMBINED_MODEL_PATH = ARTIFACTS_DIR / "miniTransformer/sgd_combined_no_chunks.pkl"
TEST_METRICS_PATH = ARTIFACTS_DIR / "miniTransformer/test_metrics_no_chunks.json"

TRAIN_EMB_PATH = ARTIFACTS_DIR / "miniTransformer/train_embeddings.npy"
TEST_EMB_PATH = ARTIFACTS_DIR / "miniTransformer/test_embeddings.npy"

# ==== parameters ====
BATCH_SIZE_EMB = 64
MINILM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TRAIN_PATH = DATA_PATH.with_name("df_feat_train.csv")
TEST_PATH = DATA_PATH.with_name("df_feat_test.csv")

# ==== Train–Test Split (only if needed) ====
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

# ==== Load TF-IDF ====
print(f"Loading TF-IDF: {TFIDF_PATH}")
tfidf = joblib.load(TFIDF_PATH)

# ==== Load SBERT ====
print(f"Loading MiniLM model: {MINILM_MODEL_NAME}")
sbert = SentenceTransformer(MINILM_MODEL_NAME)

# ==== Load Train Data ====
train_df = pd.read_csv(TRAIN_PATH).dropna()
train_texts = train_df[TEXT_COL].astype(str).tolist()
train_y = train_df[LABEL_COL].astype(int).values

# ==== TF-IDF Train ====
print("Transforming TF-IDF (train)...")
X_train_tfidf = tfidf.transform(train_texts)

# ==== Embedding Train ====
if TRAIN_EMB_PATH.exists():
    print("Loading saved train embeddings...")
    X_train_emb = np.load(TRAIN_EMB_PATH)
else:
    print("Encoding MiniLM embeddings (train)...")
    X_train_emb = sbert.encode(
        train_texts,
        batch_size=BATCH_SIZE_EMB,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(TRAIN_EMB_PATH, X_train_emb)

# Convert embeddings to sparse matrix
X_train_emb_sparse = csr_matrix(X_train_emb)

# Force int32 indices (critical!)
X_train_emb_sparse.indices = X_train_emb_sparse.indices.astype(np.int32)
X_train_emb_sparse.indptr = X_train_emb_sparse.indptr.astype(np.int32)

# Also enforce int32 for TF-IDF indices
X_train_tfidf.indices = X_train_tfidf.indices.astype(np.int32)
X_train_tfidf.indptr = X_train_tfidf.indptr.astype(np.int32)

# ==== Combine Features ====
X_train_combined = hstack([X_train_tfidf, X_train_emb_sparse], format='csr')

# ==== Train Classifier ====
clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-5,
    max_iter=1000,
    tol=1e-3
)

print("Training classifier on full dataset...")
clf.fit(X_train_combined, train_y)

# ==== TEST SET ====
test_df = pd.read_csv(TEST_PATH).dropna()
test_texts = test_df[TEXT_COL].astype(str).tolist()
test_y = test_df[LABEL_COL].astype(int).values

print("TF-IDF (test)...")
X_test_tfidf = tfidf.transform(test_texts)

print("Embedding (test)...")
X_test_emb = sbert.encode(
    test_texts,
    batch_size=BATCH_SIZE_EMB,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
np.save(TEST_EMB_PATH, X_test_emb)

# Create sparse matrix from embeddings
X_test_emb_sparse = csr_matrix(X_test_emb)

# === Fix int64 → int32 for TF-IDF ===
X_test_tfidf.indices = X_test_tfidf.indices.astype(np.int32)
X_test_tfidf.indptr = X_test_tfidf.indptr.astype(np.int32)

# === Fix int64 → int32 for embeddings sparse ===
X_test_emb_sparse.indices = X_test_emb_sparse.indices.astype(np.int32)
X_test_emb_sparse.indptr = X_test_emb_sparse.indptr.astype(np.int32)

# === Combine features (must use format='csr') ===
X_test_combined = hstack([X_test_tfidf, X_test_emb_sparse], format='csr')

test_pred = clf.predict(X_test_combined)

test_acc = accuracy_score(test_y, test_pred)
test_f1 = f1_score(test_y, test_pred, average="binary")

print(f"\nTEST Accuracy: {test_acc:.4f}")
print(f"TEST F1: {test_f1:.4f}")

# ==== Save Model + Metrics ====
joblib.dump(clf, COMBINED_MODEL_PATH)
print(f"Saved model to: {COMBINED_MODEL_PATH}")

with open(TEST_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump({"test_accuracy": test_acc, "test_f1": test_f1}, f, indent=2)
print(f"Saved test metrics to: {TEST_METRICS_PATH}")

print("\nSaved embeddings:")
print(f"  Train: {TRAIN_EMB_PATH}")
print(f"  Test:  {TEST_EMB_PATH}")