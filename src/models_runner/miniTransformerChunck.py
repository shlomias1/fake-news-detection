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
DATA_PATH = r"/home/shlomias/fake_news_detection/data/df_feat.csv"      
TEXT_COL = "text_ns_text"                       
LABEL_COL = "label"                     

ARTIFACTS_DIR = Path(os.getenv("FND_MODEL_DIR", "/home/shlomias/fake_news_detection/artifacts_simple"))
ARTIFACTS_DIR.mkdir(parents=True,exist_ok=True)
TFIDF_PATH = ARTIFACTS_DIR / "tfidf.pkl"
COMBINED_MODEL_PATH = ARTIFACTS_DIR / "sgd_combined.pkl"
CHUNK_METRICS_PATH = ARTIFACTS_DIR / "chunk_metrics.json"
TEST_METRICS_PATH = ARTIFACTS_DIR / "test_metrics.json"

TRAIN_PATH = DATA_PATH.with_name("df_feat_train.csv")
TEST_PATH = DATA_PATH.with_name("df_feat_test.csv")

# ==== parameters ====
CHUNK_SIZE = 10_000
BATCH_SIZE_EMB = 64
MINILM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VALIDATION_CHUNKS = 1  

# ==== preper train and test data ====
if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    print(f"Loading foll data from {DATA_PATH} for train/ test split...")
    df = pd.read_csv(DATA_PATH,usecols=[TEXT_COL, LABEL_COL])
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    train_df, test_df = train_test_split(
        df,
        test_size = 0.2,
        stratify = fd[LABEL_COL],
        random_state = 42,
    )
    train_df = train_df.sample(frac=1.0, random_state = 42)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved train to {TRAIN_PATH}")
    print(f"Saved test to {TEST_PATH}")
else:
    print(f"using existing train / test files:\n Train: {TRAIN_PATH}\n Test: {TEST_PATH}")

# ==== Loading existing TF-IDF ====
print(f"Loading TF-IDF vectorizer from {TFIDF_PATH} ...")
tfidf = joblib.load(TFIDF_PATH)

# ==== Loading MiniLM/SBERT ====
print(f"Loading MiniLM model: {MINILM_MODEL_NAME} ...")
sbert = SentenceTransformer(MINILM_MODEL_NAME)

# ==== Classic model ====
clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-5,
    max_iter=1,       # partial_fit loop
    tol=None
)

first_fit = True
label_encoder = LabelEncoder()
classes_ = np.array([0,1])
first_chunk = True

# ==== Variables for validation ====
val_tfidf = None
val_emb = None
val_y = None

# List for storing metrics for each chunk
chunk_metrics = []

# ==== Chunk Loop ====
for i, chunk in enumerate(pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE)):
    print(f"\n=== Train Chunk {i} ===")

    chunk = chunk.dropna(subset=[TEXT_COL, LABEL_COL])
    if chunk.empty:
        print("  Empty chunk, skipping.")
        continue

    texts = chunk[TEXT_COL].astype(str).tolist()
    y_raw = chunk[LABEL_COL].tolist()

    # יצירת y
    y = np.array(y_raw, dtype= int)

    # ---- 1. TF-IDF ----
    print("  Transforming TF-IDF...")
    X_tfidf = tfidf.transform(texts)

    # ---- 2. MiniLM Embeddings ----
    print("  Encoding MiniLM embeddings...")
    X_emb = sbert.encode(
        texts,
        batch_size=BATCH_SIZE_EMB,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    X_emb_sparse = csr_matrix(X_emb)

    # ---- 3. אם זה צ'אנק validation – רק לשמור, לא לאמן ----
    if i < VALIDATION_CHUNKS and val_tfidf is None:
        print("  Storing this chunk as validation set.")
        val_tfidf = X_tfidf
        val_emb = X_emb_sparse
        val_y = y
        # לא מאמנים על צ'אנק ה-validation
        continue

    # ---- 4. חיבור פיצ'רים ואימון ----
    X_combined = hstack([X_tfidf, X_emb_sparse])

    print("  Training SGDClassifier (partial_fit)...")
    if first_chunk:
        clf.partial_fit(X_combined, y, classes=classes_)
        first_chunk = False
    else:
        clf.partial_fit(X_combined, y)

    # ---- 5. הערכה על סט ה-validation ----
    if val_tfidf is not None:
        X_val_combined = hstack([val_tfidf, val_emb])
        y_pred = clf.predict(X_val_combined)

        acc = accuracy_score(val_y, y_pred)
        f1 = f1_score(val_y, y_pred, average="binary")

        print(f"  Validation Accuracy: {acc:.4f}, F1: {f1:.4f}")

        # שמירת המטריקות בצורת רשומה
        chunk_metrics.append({
            "chunk_index": int(i),
            "val_accuracy": float(acc),
            "val_f1": float(f1)
        })

# ==== End of training – saving model, LabelEncoder, and metrics ====

print(f"\n=== Evaluating on TEST set: {TEST_PATH} ===")

test_df = pd.read_csv(TEST_PATH)
test_df = test_df.dropna(subset=[TEXT_COL, LABEL_COL])

test_texts = test_df[TEXT_COL].astype(str).tolist()
test_y = test_df[LABEL_COL].astype(int).values

print("  TF-IDF on test...")
X_test_tfidf = tfidf.transform(test_texts)

print("  MiniLM embeddings on test...")
X_test_emb = sbert.encode(
    test_texts,
    batch_size=BATCH_SIZE_EMB,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
X_test_emb_sparse = csr_matrix(X_test_emb)

X_test_combined = hstack([X_test_tfidf, X_test_emb_sparse])

test_pred = clf.predict(X_test_combined)
test_acc = accuracy_score(test_y, test_pred)
test_f1 = f1_score(test_y, test_pred, average="binary", zero_division=0)

print(f"\nTEST Accuracy: {test_acc:.4f}, TEST F1: {test_f1:.4f}")

# =========================================================
# שלב 5: שמירת המודל והמדדים
# =========================================================
print(f"\nSaving combined SGD model to {COMBINED_MODEL_PATH}")
joblib.dump(clf, COMBINED_MODEL_PATH)

print(f"Saving per-chunk validation metrics to {CHUNK_METRICS_PATH}")
with open(CHUNK_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(chunk_metrics, f, ensure_ascii=False, indent=2)

print(f"Saving final TEST metrics to {TEST_METRICS_PATH}")
with open(TEST_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {"test_accuracy": float(test_acc), "test_f1": float(test_f1)},
        f,
        ensure_ascii=False,
        indent=2
    )