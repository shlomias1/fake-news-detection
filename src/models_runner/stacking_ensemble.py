# models_runner/stacking_ensemble.py
from __future__ import annotations
import os, json, gc, math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from utils.logger import _create_log

import numpy as np
import polars as pl
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score

# XGBoost CPU
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost לא מותקן. התקן עם: pip install xgboost") from e

FAKE_LABEL = 0  # mapping: {'fake':0,'real':1}
REAL_LABEL = 1

# ---------- לוג קטן שמכבד _create_log אם קיים ----------
def _log(msg: str):
    print(msg)
    try:
        # אם יש לך פונקציה גלובלית _create_log בפרויקט – נקרא לה
        from main import _create_log  # אם לא קיים/יש מעגליות, ה-except יתפוס
        _create_log(msg)
    except Exception:
        pass

# ---------- עזר ----------
def _ensure_text_column(df: pl.DataFrame) -> List[str]:
    for c in ("titleplus_text_ns", "title_ns_text", "text_ns_text"):
        if c in df.columns:
            return df[c].cast(pl.Utf8).fill_null("").to_list()
    t = df.get_column("title").fill_null("").cast(pl.Utf8) if "title" in df.columns else pl.Series([""]*df.height)
    x = df.get_column("text").fill_null("").cast(pl.Utf8)  if "text"  in df.columns else pl.Series([""]*df.height)
    return (t + pl.lit(" [SEP] ") + x).to_list()

def _extract_numeric_feature_matrix(df: pl.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
    num_cols = []
    for c, dt in zip(df.columns, df.dtypes):
        if c == "label":
            continue
        sdt = str(dt)
        if sdt.startswith("Int") or sdt.startswith("Float"):
            num_cols.append(c)
    if not num_cols:
        return None, []
    Xnum = df.select(num_cols).to_numpy()
    return Xnum, num_cols

def _proba_fake_from_calibrated(clf, X) -> np.ndarray:
    """החזרת הסתברות למחלקת FAKE (0) מתוך CalibratedClassifierCV / כל מודל עם predict_proba."""
    idx = int(np.where(clf.classes_ == FAKE_LABEL)[0][0])
    return clf.predict_proba(X)[:, idx]

def _pr_auc_fake(p_fake: np.ndarray, y: np.ndarray) -> float:
    y_pos = (y == FAKE_LABEL).astype(int)
    return float(average_precision_score(y_pos, p_fake))

# ---------- הגדרות ----------
@dataclass
class StackingCfg:
    # TF-IDF char לסיווג SVM
    max_char_features: int = 150_000
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    # TF-IDF word + SVD ל-XGB
    max_word_features: int = 100_000
    word_ngram_max: int = 2
    svd_dim: int = 300
    # Transformer embeddings (קלים, על CPU)
    transformer_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # XGB
    xgb_estimators: int = 400
    xgb_max_depth: int = 6
    xgb_lr: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.8
    # CV/כללי
    n_splits: int = 5
    seed: int = 42

# ---------- הטמעת טרנספורמר קל לאמבדינגים ----------
class TransformerEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def _lazy_init(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise ImportError("נדרש sentence-transformers. התקן עם: pip install sentence-transformers") from e
            self.model = SentenceTransformer(self.model_name, device="cpu")  # CPU

    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        self._lazy_init()
        # normalize_embeddings=True מחזיר וקטורים מנורמלים, טוב ללוגיסטי
        return self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        ).astype(np.float32)

# ---------- אימון סטאקינג עם OOF ----------
def train_stacking_ensemble(
    df_path: str | Path,
    out_dir: str | Path,
    cfg: StackingCfg = StackingCfg()
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(df_path) if str(df_path).endswith(".parquet") else pl.read_csv(df_path)
    texts = _ensure_text_column(df)
    y = df.get_column("label").cast(pl.Int64).to_numpy()
    Xnum, num_cols = _extract_numeric_feature_matrix(df)

    n = len(y)
    oof_svm = np.zeros(n, dtype=np.float32)
    oof_xgb = np.zeros(n, dtype=np.float32)
    oof_trf = np.zeros(n, dtype=np.float32)

    # נכין אמבדינגים פעם אחת
    _log("[stack] Encoding transformer embeddings (CPU)...")
    embedder = TransformerEmbedder(cfg.transformer_model)
    X_trf_all = embedder.encode(texts, batch_size=256)  # [N, D_emb]

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    fold = 0
    for tr_idx, va_idx in skf.split(texts, y):
        fold += 1
        _log(f"[stack][fold {fold}] start")

        # ---------- SVM על TF-IDF char + כיול להסתברויות ----------
        v_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(cfg.char_ngram_min, cfg.char_ngram_max),
            min_df=5,
            sublinear_tf=True,
            max_features=cfg.max_char_features,
            dtype=np.float32,
        )
        Xtr_char = v_char.fit_transform([texts[i] for i in tr_idx])
        Xva_char = v_char.transform([texts[i] for i in va_idx])

        base_svm = LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=cfg.seed)
        svm_cal = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
        svm_cal.fit(Xtr_char, y[tr_idx])
        oof_svm[va_idx] = _proba_fake_from_calibrated(svm_cal, Xva_char).astype(np.float32)

        # ---------- XGB על SVD(word) + פיצ'רים מספריים ----------
        v_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, cfg.word_ngram_max),
            min_df=5,
            sublinear_tf=True,
            max_features=cfg.max_word_features,
            dtype=np.float32,
        )
        Xtr_w = v_word.fit_transform([texts[i] for i in tr_idx])
        Xva_w = v_word.transform([texts[i] for i in va_idx])

        svd = TruncatedSVD(n_components=cfg.svd_dim, random_state=cfg.seed)
        Xtr_svd = svd.fit_transform(Xtr_w).astype(np.float32)
        Xva_svd = svd.transform(Xva_w).astype(np.float32)

        if Xnum is not None and len(num_cols) > 0:
            scaler = MaxAbsScaler(copy=False)
            Xtr_num = scaler.fit_transform(Xnum[tr_idx]).astype(np.float32)
            Xva_num = scaler.transform(Xnum[va_idx]).astype(np.float32)
            Xtr_xgb = np.hstack([Xtr_svd, Xtr_num]).astype(np.float32)
            Xva_xgb = np.hstack([Xva_svd, Xva_num]).astype(np.float32)
        else:
            scaler = None
            Xtr_xgb = Xtr_svd
            Xva_xgb = Xva_svd

        xgb = XGBClassifier(
            n_estimators=cfg.xgb_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_lr,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=4,
            random_state=cfg.seed,
        )
        xgb.fit(Xtr_xgb, y[tr_idx])
        oof_xgb[va_idx] = xgb.predict_proba(Xva_xgb)[:, int(np.where(xgb.classes_==FAKE_LABEL)[0][0])].astype(np.float32)

        # ---------- “Transformer” = לוגיסטי על האמבדינגים ----------
        clf_trf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=4, random_state=cfg.seed)
        clf_trf.fit(X_trf_all[tr_idx], y[tr_idx])
        proba_trf = clf_trf.predict_proba(X_trf_all[va_idx])
        oof_trf[va_idx] = proba_trf[:, int(np.where(clf_trf.classes_==FAKE_LABEL)[0][0])].astype(np.float32)

        # ניקוי ביניים
        del Xtr_char, Xva_char, Xtr_w, Xva_w, Xtr_svd, Xva_svd, Xtr_xgb, Xva_xgb
        gc.collect()
        _log(f"[stack][fold {fold}] done")

    # ---------- מטה-מודל על OOF ----------
    Z = np.vstack([oof_svm, oof_xgb, oof_trf]).T  # [N, 3]
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=4, random_state=cfg.seed)
    meta.fit(Z, y)
    pr_auc_oof = _pr_auc_fake(meta.predict_proba(Z)[:, int(np.where(meta.classes_==FAKE_LABEL)[0][0])], y)
    _log(f"[stack] meta OOF PR-AUC={pr_auc_oof:.4f}")

    # ---------- אימון סופי על כל הדאטה לשימוש בפרודקשן ----------
    # SVM char
    v_char_full = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(cfg.char_ngram_min, cfg.char_ngram_max),
        min_df=5,
        sublinear_tf=True,
        max_features=cfg.max_char_features,
        dtype=np.float32,
    )
    X_char_full = v_char_full.fit_transform(texts)
    svm_full = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=cfg.seed),
                                      method="sigmoid", cv=5)
    svm_full.fit(X_char_full, y)

    # XGB (SVD + features) – fit על הכל
    v_word_full = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, cfg.word_ngram_max),
        min_df=5,
        sublinear_tf=True,
        max_features=cfg.max_word_features,
        dtype=np.float32,
    )
    X_word_full = v_word_full.fit_transform(texts)
    svd_full = TruncatedSVD(n_components=cfg.svd_dim, random_state=cfg.seed)
    X_svd_full = svd_full.fit_transform(X_word_full).astype(np.float32)

    if Xnum is not None and len(num_cols) > 0:
        scaler_full = MaxAbsScaler(copy=False)
        Xnum_full = scaler_full.fit_transform(Xnum).astype(np.float32)
        X_xgb_full = np.hstack([X_svd_full, Xnum_full]).astype(np.float32)
    else:
        scaler_full = None
        X_xgb_full = X_svd_full

    xgb_full = XGBClassifier(
        n_estimators=cfg.xgb_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_lr,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=4,
        random_state=cfg.seed,
    )
    xgb_full.fit(X_xgb_full, y)

    # Transformer head על כל הדאטה
    X_trf_full = X_trf_all
    trf_full = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=4, random_state=cfg.seed)
    trf_full.fit(X_trf_full, y)

    # שמירת ארטיפקטים
    joblib.dump(v_char_full, out_dir / "tfidf_char.pkl")
    joblib.dump(svm_full,   out_dir / "svm_char_calibrated.pkl")

    joblib.dump(v_word_full, out_dir / "tfidf_word.pkl")
    joblib.dump(svd_full,    out_dir / "svd.pkl")
    if scaler_full is not None:
        joblib.dump({"scaler": scaler_full, "num_cols": num_cols}, out_dir / "num_scaler.pkl")
    joblib.dump(xgb_full, out_dir / "xgb.pkl")

    # אמבדder נשאר נטען לפי שם – אין מה “לשמור”
    with open(out_dir / "transformer.json", "w", encoding="utf-8") as f:
        json.dump({"model": cfg.transformer_model}, f, ensure_ascii=False, indent=2)
    joblib.dump(trf_full, out_dir / "trf_head.pkl")

    joblib.dump(meta, out_dir / "meta_logreg.pkl")

    # מטא־דאטה
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "label_mapping": {"fake": 0, "real": 1},
            "oof_pr_auc": pr_auc_oof,
            "cfg": vars(cfg),
            "base_features": {
                "tfidf_char": {"ngram": [cfg.char_ngram_min, cfg.char_ngram_max], "max_features": cfg.max_char_features},
                "tfidf_word": {"ngram": [1, cfg.word_ngram_max], "max_features": cfg.max_word_features},
                "svd_dim": cfg.svd_dim,
                "num_cols": num_cols
            }
        }, f, ensure_ascii=False, indent=2)

    _log(f"[stack] saved artifacts to: {out_dir}")
    return {"out_dir": str(out_dir), "oof_pr_auc": pr_auc_oof}

# ---------- חיזוי ----------
def predict_article(
    artifacts_dir: str | Path,
    title: Optional[str],
    text: Optional[str]
) -> Dict:
    artifacts_dir = Path(artifacts_dir)

    # טען ארטיפקטים
    v_char = joblib.load(artifacts_dir / "tfidf_char.pkl")
    svm    = joblib.load(artifacts_dir / "svm_char_calibrated.pkl")

    v_word = joblib.load(artifacts_dir / "tfidf_word.pkl")
    svd    = joblib.load(artifacts_dir / "svd.pkl")
    try:
        dnum = joblib.load(artifacts_dir / "num_scaler.pkl")
        scaler = dnum["scaler"]; num_cols = dnum["num_cols"]
    except Exception:
        scaler = None; num_cols = []
    xgb    = joblib.load(artifacts_dir / "xgb.pkl")

    with open(artifacts_dir / "transformer.json","r",encoding="utf-8") as f:
        trf_name = json.load(f)["model"]
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(trf_name, device="cpu")
    trf_head = joblib.load(artifacts_dir / "trf_head.pkl")

    meta   = joblib.load(artifacts_dir / "meta_logreg.pkl")

    # בניית טקסט
    raw = (title or "") + " [SEP] " + (text or "")

    # SVM char
    Xc = v_char.transform([raw])
    p_svm = float(_proba_fake_from_calibrated(svm, Xc)[0])

    # XGB: word->SVD + optional num
    Xw = v_word.transform([raw])
    Xsvd = svd.transform(Xw).astype(np.float32)
    if scaler is not None and num_cols:
        # אין לך פיצ'רים הנדסיים במעבר יחיד, נכניס 0ים (או תעביר dict חיצוני אם תרצה)
        Xnum = scaler.transform(np.zeros((1, len(num_cols)), dtype=np.float32)).astype(np.float32)
        Xxgb = np.hstack([Xsvd, Xnum]).astype(np.float32)
    else:
        Xxgb = Xsvd
    p_xgb = float(xgb.predict_proba(Xxgb)[:, int(np.where(xgb.classes_==FAKE_LABEL)[0][0])][0])

    # Transformer
    v = embedder.encode([raw], batch_size=1, normalize_embeddings=True)
    p_trf = float(trf_head.predict_proba(v)[:, int(np.where(trf_head.classes_==FAKE_LABEL)[0][0])][0])

    # Meta
    z = np.array([[p_svm, p_xgb, p_trf]], dtype=np.float32)
    p_final = float(meta.predict_proba(z)[:, int(np.where(meta.classes_==FAKE_LABEL)[0][0])][0])

    # תיוג (ללא ספי abstention כאן; אפשר לשלב מה-Threshold Bandit שלך)
    label = "FAKE" if p_final >= 0.5 else "REAL"
    return {"prob_fake": p_final, "label": label, "base": {"svm": p_svm, "xgb": p_xgb, "trf": p_trf}}
