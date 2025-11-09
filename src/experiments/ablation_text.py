# experiments/ablation_text.py
from __future__ import annotations
import os, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import polars as pl
import joblib
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve

FAKE_LABEL = 0  # {'fake': 0, 'real': 1}
REAL_LABEL = 1

def proba_fake(clf, X) -> np.ndarray:
    """החזר p(fake) לפי classes_ של המודל (אל תניח שהעמודה היא 0/1)."""
    idx = int(np.where(clf.classes_ == FAKE_LABEL)[0][0])
    return clf.predict_proba(X)[:, idx]

# ====== Stopwords (ניסיונות טעינה) ======
try:
    from config import STOP_ALL  # אם יש לך קובץ config עם עברית/אנגלית וכו'
except Exception:
    STOP_ALL = set()  # fallback: בלי הסרת stopwords

# ====== קריאת הדאטה ======
def load_df(df_path: str) -> pl.DataFrame:
    return pl.read_parquet(df_path) if str(df_path).endswith(".parquet") else pl.read_csv(df_path)

# ====== בניית טקסט לפי מצב ה-Ablation ======
def build_texts_for_mode(df: pl.DataFrame, mode: str) -> List[str]:
    mode = mode.lower()
    n = df.height

    def safe_utf8(col: str) -> pl.Series:
        return df.get_column(col).fill_null("").cast(pl.Utf8) if col in df.columns else pl.Series([""]*n, dtype=pl.Utf8)

    if mode == "title":
        return safe_utf8("title").to_list()

    if mode == "text":
        return safe_utf8("text").to_list()

    if mode == "titleplus_text_ns":
        # אם כבר קיים בעיבוד פיצ'רים
        if "titleplus_text_ns" in df.columns:
            return df.get_column("titleplus_text_ns").cast(pl.Utf8).fill_null("").to_list()
        # אחרת נבנה מ-title/text עם הסרת stopwords (פשוטה)
        t = safe_utf8("title").to_list()
        x = safe_utf8("text").to_list()
        out = []
        for ti, xi in zip(t, x):
            toks = [w for w in (ti + " " + xi).split() if w not in STOP_ALL]
            out.append(" ".join(toks))
        return out

    if mode in ("title_text_no_stop", "title+text_no_stop"):
        # נסה להשתמש בטוקנים הנקיים אם יש:
        if "title_tokens_ns" in df.columns and "text_tokens_ns" in df.columns:
            tt = df.get_column("title_tokens_ns").to_list()
            tx = df.get_column("text_tokens_ns").to_list()
            # עמודות אלו לרוב הן List[str]
            out = []
            for a, b in zip(tt, tx):
                a = a or []; b = b or []
                out.append(" ".join(a + b))
            return out
        # fallback: הסרה ע"י STOP_ALL מהטקסט הגולמי
        t = safe_utf8("title").to_list()
        x = safe_utf8("text").to_list()
        out = []
        for ti, xi in zip(t, x):
            toks = [w for w in (ti + " " + xi).split() if w not in STOP_ALL]
            out.append(" ".join(toks))
        return out

    # ברירת מחדל: title + text (ללא הסרת stopwords)
    t = safe_utf8("title").to_list()
    x = safe_utf8("text").to_list()
    return [(ti + " [SEP] " + xi) for ti, xi in zip(t, x)]

# ====== בניית וקטורייזרים + פיצ'רים מספריים (אופציונלי) ======
def vectorize_texts(train_texts: List[str], val_texts: List[str],
                    max_features_word: Optional[int]=None,
                    max_features_char: Optional[int]=None):
    v_word = TfidfVectorizer(
        analyzer="word", ngram_range=(1,2), min_df=5, sublinear_tf=True, max_features=max_features_word
    )
    v_char = TfidfVectorizer(
        analyzer="char", ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=max_features_char
    )
    Xtr_w = v_word.fit_transform(train_texts)
    Xva_w = v_word.transform(val_texts)
    Xtr_c = v_char.fit_transform(train_texts)
    Xva_c = v_char.transform(val_texts)
    return hstack([Xtr_w, Xtr_c], format="csr"), hstack([Xva_w, Xva_c], format="csr"), v_word, v_char

# ====== מודל בסיס + כיול הסתברויות ======
def make_model(seed: int = 42):
    base = LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=seed)
    clf  = CalibratedClassifierCV(base, cv=3)  # Platt/Isotonic בהתאם ל-sklearn
    return clf

# ====== מדדים ======
def metrics_from_proba(p_fake: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pos = (y == FAKE_LABEL).astype(int)  # חיובי=FAKE
    pr_auc = float(average_precision_score(y_pos, p_fake))
    roc = float(roc_auc_score(y_pos, p_fake)) if len(np.unique(y_pos)) == 2 else float("nan")
    # F1@best_threshold (לפי סריקה על PR curve)
    prec, rec, thr = precision_recall_curve(y_pos, p_fake)
    f1s = 2 * (prec * rec) / np.clip(prec + rec, 1e-12, None)
    f1_best = float(np.nanmax(f1s)) if f1s.size else float("nan")
    return {"pr_auc": pr_auc, "roc_auc": roc, "f1_best": f1_best}

# ====== הרצת ablation אחת עם Kfold ======
def run_single_ablation(df: pl.DataFrame, mode: str, n_splits: int = 5, seed: int = 42) -> Dict:
    texts = build_texts_for_mode(df, mode)
    y = df.get_column("label").cast(pl.Int64).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows = []
    for k, (tr_idx, va_idx) in enumerate(skf.split(texts, y), start=1):
        Xtr_txt = [texts[i] for i in tr_idx]
        Xva_txt = [texts[i] for i in va_idx]
        ytr = y[tr_idx]; yva = y[va_idx]

        Xtr, Xva, v_word, v_char = vectorize_texts(Xtr_txt, Xva_txt)
        clf = make_model(seed=seed)
        clf.fit(Xtr, ytr)

        p_fake_va = proba_fake(clf, Xva)
        m = metrics_from_proba(p_fake_va, yva)
        m.update({"fold": k, "n_val": int(len(va_idx))})
        fold_rows.append(m)

    # ממוצע/סטיית תקן
    agg = {}
    for key in ("pr_auc","roc_auc","f1_best"):
        vals = [r[key] for r in fold_rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        agg[key+"_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[key+"_std"]  = float(np.std(vals)) if vals else float("nan")

    return {"mode": mode, "folds": fold_rows, "summary": agg}

# ====== הרצת כל ה-Ablations ושמירה ======
def run_all_ablations(
    df_path: str,
    out_dir: str | Path,
    modes: List[str] = ("title","text","titleplus_text_ns","title_text_no_stop"),
    n_splits: int = 5,
    seed: int = 42
) -> Dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_df(df_path)

    results = []
    for mode in modes:
        print(f"[ablation] Running mode={mode} ...")
        res = run_single_ablation(df, mode, n_splits=n_splits, seed=seed)
        results.append(res)

    # טבלת סיכום
    import pandas as pd
    rows = []
    for r in results:
        s = r["summary"]
        rows.append({
            "mode": r["mode"],
            "pr_auc_mean": s["pr_auc_mean"], "pr_auc_std": s["pr_auc_std"],
            "roc_auc_mean": s["roc_auc_mean"], "roc_auc_std": s["roc_auc_std"],
            "f1_best_mean": s["f1_best_mean"], "f1_best_std": s["f1_best_std"],
        })
    df_sum = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    df_sum.to_csv(out_dir / "ablation_summary.csv", index=False)

    # שמירת קבצי JSON עם פירוט קפלים לכל מצב
    with open(out_dir / "ablation_details.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[ablation] Saved summary → {out_dir/'ablation_summary.csv'}")
    print(f"[ablation] Saved details → {out_dir/'ablation_details.json'}")
    return {"out_dir": str(out_dir), "results": results}