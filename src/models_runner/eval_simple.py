# models_runner/eval_simple.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple, List
import numpy as np
import polars as pl
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    average_precision_score, roc_auc_score
)

# מיפוי המחלקות שלך
FAKE_LABEL = 0   # {'fake': 0, 'real': 1}
REAL_LABEL = 1

# לוג ידידותי: גם print וגם _create_log אם קיים
def _log(msg: str):
    print(msg)
    try:
        from utils.logger import _create_log
        _create_log(msg)
    except Exception:
        pass

def _load_df_texts_labels(df_path: str | Path) -> Tuple[List[str], np.ndarray, pl.DataFrame]:
    df = pl.read_parquet(df_path) if str(df_path).endswith(".parquet") else pl.read_csv(df_path)
    for c in ("titleplus_text_ns", "title_ns_text", "text_ns_text"):
        if c in df.columns:
            texts = df[c].cast(pl.Utf8).fill_null("").to_list()
            break
    else:
        t = df.get_column("title").fill_null("").cast(pl.Utf8) if "title" in df.columns else pl.Series([""]*df.height)
        x = df.get_column("text").fill_null("").cast(pl.Utf8)  if "text"  in df.columns else pl.Series([""]*df.height)
        texts = (t + pl.lit(" [SEP] ") + x).to_list()
    y = df.get_column("label").cast(pl.Int64).to_numpy()
    return texts, y, df

def _proba_fake(clf, X) -> np.ndarray:
    """מחזיר p(fake) לפי classes_ של המודל (בטוח גם אם הסדר שונה)."""
    idx = int(np.where(clf.classes_ == FAKE_LABEL)[0][0])
    return clf.predict_proba(X)[:, idx]

# ---------------------------------------------------------
# 1) הערכת מודל בסיסי (בלי Abstention)
# ---------------------------------------------------------
def eval_plain(
    df_path: str | Path,
    model_dir: str | Path,
    val_frac: float = 0.2,
    seed: int = 42
) -> Dict:
    model_dir = Path(model_dir)
    vect = joblib.load(model_dir / "tfidf.pkl")
    clf  = joblib.load(model_dir / "sgd_logloss.pkl")

    texts, y, _ = _load_df_texts_labels(df_path)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(sss.split(texts, y))
    Xva_txt = [texts[i] for i in va_idx]
    y_va    = y[va_idx]

    X_va = vect.transform(Xva_txt)
    p_fk = _proba_fake(clf, X_va)     # p(fake) \in [0,1]

    # חיזוי בינארי: p>0.5 => FAKE(=0), אחרת REAL(=1)
    y_hat = np.where(p_fk > 0.5, FAKE_LABEL, REAL_LABEL)

    acc = float(accuracy_score(y_va, y_hat))
    f1_fake = float(f1_score((y_va==FAKE_LABEL).astype(int),
                             (y_hat==FAKE_LABEL).astype(int), zero_division=0))
    pr_auc = float(average_precision_score((y_va==FAKE_LABEL).astype(int), p_fk))
    try:
        roc_auc = float(roc_auc_score((y_va==FAKE_LABEL).astype(int), p_fk))
    except Exception:
        roc_auc = None

    res = {
        "n_val": int(len(y_va)),
        "accuracy": acc,
        "f1_fake@0.5": f1_fake,
        "pr_auc_fake": pr_auc,
        "roc_auc_fake": roc_auc,
    }
    _log(f"[EVAL plain] n={res['n_val']}  acc={acc:.4f}  F1_fake@0.5={f1_fake:.4f}  PR-AUC={pr_auc:.4f}" +
         (f"  ROC-AUC={roc_auc:.4f}" if roc_auc is not None else ""))
    return res

# ---------------------------------------------------------
# 2) הערכה עם Abstention לפי ספים פר-קונטקסט
#    משתמש ב-threshold_bandit לבניית הקונטקסט
# ---------------------------------------------------------
def eval_with_thresholds(
    df_path: str | Path,
    model_dir: str | Path,
    thresholds_json: str | Path,
    context_keys: Iterable[str] = ("source","category","len_bin","punc_bin","clickbait"),
    val_frac: float = 0.2,
    seed: int = 42
) -> Dict:
    from models_runner.threshold_bandit import build_context, Thresholds  # משתמש בכלים שכבר כתבת

    model_dir = Path(model_dir)
    vect = joblib.load(model_dir / "tfidf.pkl")
    clf  = joblib.load(model_dir / "sgd_logloss.pkl")

    import json
    with open(thresholds_json, "r", encoding="utf-8") as f:
        th = json.load(f)
    th_map_json  = th["thresholds_map"]
    th_global_js = th["global_thresholds"]

    thresholds_map = {k: Thresholds(**v) for k,v in th_map_json.items()}
    global_th = Thresholds(**th_global_js)

    texts, y, df = _load_df_texts_labels(df_path)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(sss.split(texts, y))
    Xva_txt = [texts[i] for i in va_idx]
    y_va    = y[va_idx]
    df_va = (
        df.with_row_count("_rn")
          .filter(pl.col("_rn").is_in(pl.Series(va_idx, dtype=pl.UInt32)))
          .drop("_rn")
    )

    X_va = vect.transform(Xva_txt)
    p_fk = _proba_fake(clf, X_va)

    # בונים קונטקסטים ויישום הספים
    ctx = build_context(df_va, keys=context_keys)

    y_hat = np.full(len(p_fk), -1, dtype=int)
    abst  = np.zeros(len(p_fk), dtype=bool)
    for i, prob in enumerate(p_fk):
        th = thresholds_map.get(ctx[i], global_th)
        if prob < th.tau_low:
            y_hat[i] = REAL_LABEL
        elif prob > th.tau_high:
            y_hat[i] = FAKE_LABEL
        else:
            y_hat[i] = -1
            abst[i] = True

    covered = ~abst
    coverage = float(covered.mean())
    acc_on_covered = float(accuracy_score(y_va[covered], y_hat[covered])) if covered.any() else 0.0
    effective_accuracy = acc_on_covered * coverage  # אם מחשיבים abstain כשגיאה

    # F1 על covered (אופציונלי)
    if covered.any():
        f1_fake_cov = float(f1_score((y_va[covered]==FAKE_LABEL).astype(int),
                                     (y_hat[covered]==FAKE_LABEL).astype(int), zero_division=0))
    else:
        f1_fake_cov = 0.0

    res = {
        "n_val": int(len(y_va)),
        "coverage": coverage,
        "accuracy_on_covered": acc_on_covered,
        "effective_accuracy": effective_accuracy,
        "f1_fake_on_covered": f1_fake_cov,
        "abstain_count": int(abst.sum()),
        "fp": int(((y_hat == FAKE_LABEL) & (y_va == REAL_LABEL)).sum()),
        "fn": int(((y_hat == REAL_LABEL) & (y_va == FAKE_LABEL)).sum()),
    }
    _log("[EVAL abst] "
         f"n={res['n_val']}  coverage={coverage:.3f}  "
         f"acc_on_covered={acc_on_covered:.4f}  effective_acc={effective_accuracy:.4f}  "
         f"F1_fake(covered)={f1_fake_cov:.4f}  abst={res['abstain_count']}")
    return res