import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# ===================== Config & Paths =====================
# הגדרות מתוך המודל הראשי (יש לוודא עקביות)
TEXT_COL = "text_ns_text"
LABEL_COL = "label"

# נתיב תיקיית הארטיפקטים (Artifacts)
ARTIFACTS_DIR = Path(os.getenv("FND_MODEL_DIR", "artifacts_simple"))

# נתיבי מודלי בסיס קיימים (שלא נאמן מחדש)
TFIDF_PATH = ARTIFACTS_DIR / "tfidf.pkl"
TFIDF_EXISTING_CLF_PATH = ARTIFACTS_DIR / "sgd_logloss.pkl"

LF_DIR = ARTIFACTS_DIR / "late_fusion"
EMB_CLF_PATH = LF_DIR / "clf_emb_sgd.pkl"
META_CLF_PATH = LF_DIR / "clf_meta_logreg.pkl"

MINILM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE_EMB = 64

# נתיב לקובץ הנתונים החדש (שאילתות משתמשים מתויגות)
NEW_DATA_PATH = Path(
    "predictions") / "preds.jsonl" # הנתיב היחסי מתיקיית הבסיס
# ----------------------------------------------------------


def load_new_data(new_data_path: Path) -> pd.DataFrame:
    """טוען את הנתונים החדשים מקובץ jsonl."""
    if not new_data_path.exists():
        print(f"🛑 שגיאה: קובץ נתונים חדש לא נמצא ב- {new_data_path}")
        return None
    
    print(f"טוען נתונים חדשים מ: {new_data_path}")
    
    # טעינת נתונים מקובץ jsonl
    records = []
    with open(new_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"אזהרה: שורה שגויה ב-JSONL: {line.strip()}")
                continue
    
    df = pd.DataFrame(records)
    
    # ודא שהעמודות הנדרשות קיימות ושאין ערכים חסרים
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        print(f"🛑 שגיאה: נתונים חסרים עמודות {TEXT_COL} או {LABEL_COL}")
        return None
    
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    
    if len(df) == 0:
        print("🛑 אין נתונים תקינים לאימון מחדש.")
        return None
        
    print(f"👍 נטענו {len(df)} דוגמאות חדשות.")
    return df


def retrain_meta_classifier():
    """מבצע אימון מחדש של ה-Meta-Classifier על נתונים חדשים."""
    
    # 1. טעינת נתונים חדשים
    new_df = load_new_data(NEW_DATA_PATH)
    if new_df is None:
        return
    
    new_texts = new_df[TEXT_COL].astype(str).tolist()
    new_y = new_df[LABEL_COL].values
    
    # 2. טעינת מודלי בסיס
    print("טוען מודלי בסיס קיימים...")
    try:
        tfidf = joblib.load(TFIDF_PATH)
        clf_tfidf = joblib.load(TFIDF_EXISTING_CLF_PATH)
        clf_emb = joblib.load(EMB_CLF_PATH)
        sbert = SentenceTransformer(MINILM_MODEL_NAME)
    except FileNotFoundError as e:
        print(f"🛑 שגיאת טעינת מודל בסיס: {e}")
        print("ודא שהמודלים הקיימים (tfidf.pkl, sgd_logloss.pkl, clf_emb_sgd.pkl) נמצאים במקום.")
        return

    # 3. יצירת מאפיינים חדשים (Embeddings ו-TF-IDF)
    print("יצירת מאפייני TF-IDF חדשים...")
    X_new_tfidf = tfidf.transform(new_texts).astype(np.float32)
    
    print("יצירת מאפייני MiniLM Embeddings חדשים...")
    X_new_emb = sbert.encode(
        new_texts,
        batch_size=BATCH_SIZE_EMB,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # 4. קבלת הסתברויות מודלי בסיס (Meta-Features)
    print("קבלת הסתברויות ממודלי בסיס...")
    new_prob_tfidf = clf_tfidf.predict_proba(X_new_tfidf)[:, 1]
    new_prob_emb = clf_emb.predict_proba(X_new_emb)[:, 1]
    
    # 5. בניית מערך האימון ל-Meta-Classifier
    new_meta_X = np.vstack([new_prob_tfidf, new_prob_emb]).T
    
    # 6. אימון ה-Meta-Classifier מחדש (או המשך אימון)
    print("אימון מחדש (או המשך אימון) של ה-Meta-Classifier...")
    
    # טוען את המודל הקיים אם קיים
    if META_CLF_PATH.exists():
        meta_clf = joblib.load(META_CLF_PATH)
    else:
        # אם לא קיים - מתחיל אימון מהתחלה
        meta_clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

    # אימון על הנתונים החדשים
    meta_clf.fit(new_meta_X, new_y)
    
    # 7. שמירת המודל המעודכן
    joblib.dump(meta_clf, META_CLF_PATH)
    print(f"✅ Meta-Classifier מעודכן נשמר ל- {META_CLF_PATH}")
    
    # בדיקת ביצועים בסיסית (אופציונלי)
    new_pred = meta_clf.predict(new_meta_X)
    acc = accuracy_score(new_y, new_pred)
    f1 = f1_score(new_y, new_pred, average="binary")
    print(f"\nמדדי אימון על הנתונים החדשים: Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    (LF_DIR / "retrain_success.txt").write_text(f"Retrained on {len(new_df)} samples at {pd.Timestamp.now()}")
    print("--- הסקריפט הסתיים בהצלחה ---")


if __name__ == "__main__":
    retrain_meta_classifier()