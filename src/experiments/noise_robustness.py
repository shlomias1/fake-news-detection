#noise_robustness.py
from __future__ import annotations
import re, random, json, gc
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import polars as pl
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from utils.logger import _create_log

FAKE_LABEL = 0   # {'fake': 0, 'real': 1}
REAL_LABEL = 1

# ---- logger קטן שמנסה לקרוא ל-_create_log אם יש ----
def _log(msg: str):
    print(msg)
    try:
        from main import _create_log
        _create_log(msg)
    except Exception:
        pass

# ==========================
#     עזרי דאטה
# ==========================
def _ensure_texts(df: pl.DataFrame) -> List[str]:
    for c in ("titleplus_text_ns", "title_ns_text", "text_ns_text"):
        if c in df.columns:
            return df[c].cast(pl.Utf8).fill_null("").to_list()
    # נפילה נעימה: title + text
    t = df.get_column("title").fill_null("").cast(pl.Utf8) if "title" in df.columns else pl.Series([""]*df.height)
    x = df.get_column("text").fill_null("").cast(pl.Utf8)  if "text"  in df.columns else pl.Series([""]*df.height)
    return (t + pl.lit(" [SEP] ") + x).to_list()

def _load_df(df_path: str | Path) -> Tuple[List[str], np.ndarray, pl.DataFrame]:
    df = pl.read_parquet(df_path) if str(df_path).endswith(".parquet") else pl.read_csv(df_path)
    texts = _ensure_texts(df)
    y = df.get_column("label").cast(pl.Int64).to_numpy()
    return texts, y, df

def _pr_auc_fake(p_fake: np.ndarray, y: np.ndarray) -> float:
    y_pos = (y == FAKE_LABEL).astype(int)
    return float(average_precision_score(y_pos, p_fake))

def _f1_fake(p_fake: np.ndarray, y: np.ndarray, thr: float = 0.5) -> float:
    y_pred = (p_fake >= thr).astype(int)  # 1=FAKE? לא! FAKE=0 → נטפל בזה מיד:
    # נעשה מיפוי: 1→FAKE_LABEL(=0), 0→REAL_LABEL(=1)
    # כדי לחשב F1 למחלקת FAKE, ניצור תחזית ב-0/1 עם 1 עבור FAKE
    y_pred_fakepos = (p_fake >= thr).astype(int)     # 1 אומר "FAKE"
    y_true_fakepos = (y == FAKE_LABEL).astype(int)   # 1 אומר "FAKE"
    _, _, f1, _ = precision_recall_fscore_support(y_true_fakepos, y_pred_fakepos, average="binary", zero_division=0)
    return float(f1)

# ==========================
#     רעשים/אוגמנטציות
# ==========================
EMOJIS = ["🙂","🔥","❗","❓","😮","🤔","📢","💥"]

def noise_keyboard_typos(s: str, p: float = 0.02) -> str:
    # השמטה/כפל/החלפה של תו אחד-אחד בהסתברות p
    out = []
    abc = "אבגדהוזחטיךכלםמןנסעףפץצקרשתabcdefghijklmnopqrstuvwxyz0123456789"
    for ch in s:
        r = random.random()
        if r < p/3:
            continue                    # מחיקה
        out.append(ch)
        if p/3 <= r < 2*p/3:
            out.append(ch)              # כפילות
        elif 2*p/3 <= r < p:
            out[-1] = random.choice(abc)  # החלפה
    return "".join(out)

def noise_char_swap(s: str, p: float = 0.02) -> str:
    # החלפת שכנים (transposition) בהסתברות p לכל צמד
    s = list(s)
    i = 0
    while i < len(s) - 1:
        if random.random() < p:
            s[i], s[i+1] = s[i+1], s[i]
            i += 2
        else:
            i += 1
    return "".join(s)

def noise_punct_burst(s: str, p: float = 0.2) -> str:
    s = re.sub(r"!", lambda m: "!"*random.choice([1,2,3]) if random.random()<p else "!", s)
    s = re.sub(r"\?", lambda m: "?"*random.choice([1,2,3]) if random.random()<p else "?", s)
    s = re.sub(r"\.\.\.", lambda m: "."*random.choice([3,6,9]) if random.random()<p else m.group(0), s)
    return s

def noise_emoji(s: str, p: float = 0.2) -> str:
    s2 = s
    if random.random() < p:
        s2 = random.choice(EMOJIS) + " " + s2
    if random.random() < p:
        s2 = s2 + " " + random.choice(EMOJIS)
    return s2

def noise_spell_correct_en(s: str, p: float = 0.5) -> str:
    """
    'תיקון איות' (באנגלית בלבד, אופציונלי).
    אם pyspellchecker לא מותקן – נחזיר את הטקסט כמות שהוא.
    intensity=p → הסתברות לתקן את המילה האנגלית.
    """
    try:
        from spellchecker import SpellChecker
        sp = SpellChecker(distance=1)  # קליל ומהיר
    except Exception:
        return s

    def fix_word(w: str) -> str:
        if not re.search(r"[A-Za-z]", w):
            return w
        if random.random() > p:
            return w
        c = sp.correction(w)
        return c if c is not None else w

    return " ".join(fix_word(w) for w in s.split())

NOISE_FUNCS = {
    "keyboard_typos": noise_keyboard_typos,
    "char_swap":      noise_char_swap,
    "punct_burst":    noise_punct_burst,
    "emoji":          noise_emoji,
    "spell_correct":  noise_spell_correct_en,  # אופציונלי, עובד על מילים לטיניות
}

def apply_noise_batch(texts: List[str], noise: str, intensity: float) -> List[str]:
    fn = NOISE_FUNCS[noise]
    return [fn(t, intensity) if noise != "spell_correct" else fn(t, intensity) for t in texts]

# ==========================
#   מודל: טעינה או אימון
# ==========================
def _load_existing_model(art_dir: Path):
    """
    מנסה לטעון מודל קיים בשני פורמטים:
    1) מה-aug bandit: tfidf.pkl + sgd_logloss.pkl
    2) מה-GA: tfidf_char.pkl + svm_calibrated.pkl
    מחזיר (vectorizer, clf, proba_func)
    """
    if (art_dir / "tfidf.pkl").exists() and (art_dir / "sgd_logloss.pkl").exists():
        vect = joblib.load(art_dir / "tfidf.pkl")
        clf  = joblib.load(art_dir / "sgd_logloss.pkl")
        def proba(X): return clf.predict_proba(X)[:, 1]  # עמודת FAKE? נדאג למפה:
        # נוודא שעמודת FAKE היא אינדקס 0:
        try:
            idx = int(np.where(clf.classes_ == FAKE_LABEL)[0][0])
            def proba(X): return clf.predict_proba(X)[:, idx]
        except Exception:
            pass
        return vect, clf, proba

    if (art_dir / "tfidf_char.pkl").exists() and (art_dir / "svm_calibrated.pkl").exists():
        vect = joblib.load(art_dir / "tfidf_char.pkl")
        clf  = joblib.load(art_dir / "svm_calibrated.pkl")
        def proba(X):
            idx = int(np.where(clf.classes_ == FAKE_LABEL)[0][0])
            return clf.predict_proba(X)[:, idx]
        return vect, clf, proba

    return None, None, None

def _train_baseline(texts_tr: List[str], y_tr: np.ndarray):
    """
    Baseline חזק לרעשים תו-אופיים: TF-IDF char_wb + LinearSVC (כיול sigmoid).
    """
    vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=5, sublinear_tf=True, max_features=150_000, dtype=np.float32)
    Xtr  = vect.fit_transform(texts_tr)
    base = LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=42)
    clf  = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(Xtr, y_tr)

    def proba(X):
        idx = int(np.where(clf.classes_ == FAKE_LABEL)[0][0])
        return clf.predict_proba(X)[:, idx]

    return vect, clf, proba

# ==========================
#     הרצת ניסוי הרעש
# ==========================
def run_noise_robustness(
    df_path: str | Path,
    out_dir: str | Path,
    artifacts_dir: Optional[str | Path] = None,   # אם יש מודל קיים – נשתמש בו
    intensities: Dict[str, List[float]] = None,
    test_fraction: float = 0.2,
    seed: int = 42
) -> Dict:
    """
    מחזיר dict עם תוצאות, ושומר CSV/JSON ל-out_dir.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    texts, y, df = _load_df(df_path)

    if intensities is None:
        intensities = {
            "keyboard_typos": [0.01, 0.03, 0.06],
            "char_swap":      [0.01, 0.03, 0.06],
            "punct_burst":    [0.10, 0.20, 0.40],
            "emoji":          [0.10, 0.20, 0.40],
            "spell_correct":  [0.50],    # הסתברות תיקון למילה לטינית (אם מותקן)
        }

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    tr_idx, va_idx = next(sss.split(texts, y))
    Xtr_txt = [texts[i] for i in tr_idx]; ytr = y[tr_idx]
    Xva_txt = [texts[i] for i in va_idx]; yva = y[va_idx]

    # טען מודל קיים או אימן בסיסי
    vect = clf = proba = None
    if artifacts_dir is not None:
        vect, clf, proba = _load_existing_model(Path(artifacts_dir))
        if vect is not None:
            _log(f"[noise] loaded model from: {artifacts_dir}")
    if vect is None:
        _log("[noise] training baseline char TF-IDF + LinearSVC (calibrated)...")
        vect, clf, proba = _train_baseline(Xtr_txt, ytr)

    # בסיס נקי (clean)
    Xva_clean = vect.transform(Xva_txt)
    p_clean   = proba(Xva_clean)
    pr_auc_clean = _pr_auc_fake(p_clean, yva)
    f1_clean     = _f1_fake(p_clean, yva, thr=0.5)

    rows = []
    rows.append({
        "noise": "clean", "intensity": 0.0,
        "pr_auc": pr_auc_clean,
        "f1_fake@0.5": f1_clean,
        "drop_pr_auc": 0.0,
        "drop_f1": 0.0
    })
    _log(f"[noise] clean: PR-AUC={pr_auc_clean:.4f}  F1_FAKE@0.5={f1_clean:.4f}")

    # לכל רעש/עוצמה
    for noise_name, vals in intensities.items():
        for alpha in vals:
            Xno_txt = apply_noise_batch(Xva_txt, noise_name, alpha)
            Xno     = vect.transform(Xno_txt)
            p_no    = proba(Xno)
            pr_auc  = _pr_auc_fake(p_no, yva)
            f1_no   = _f1_fake(p_no, yva, thr=0.5)

            rows.append({
                "noise": noise_name, "intensity": float(alpha),
                "pr_auc": pr_auc,
                "f1_fake@0.5": f1_no,
                "drop_pr_auc": float(pr_auc - pr_auc_clean),
                "drop_f1": float(f1_no - f1_clean),
            })
            _log(f"[noise] {noise_name:12s} α={alpha:>4}:  PR-AUC={pr_auc:.4f} (Δ{pr_auc-pr_auc_clean:+.4f}) | F1={f1_no:.4f} (Δ{f1_no-f1_clean:+.4f})")

            gc.collect()

    # שמירה
    import pandas as pd
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_dir / "noise_robustness_results.csv", index=False)

    # דוגמאות before/after (20 שורות אקראיות מהוולידציה לרעש האחרון שרץ)
    rng = np.random.default_rng(seed)
    samp_idx = rng.choice(len(Xva_txt), size=min(20, len(Xva_txt)), replace=False)
    samples = pd.DataFrame({
        "orig": [Xva_txt[i] for i in samp_idx],
        "label": [int(yva[i]) for i in samp_idx],
    })
    # נעשה דוגמה עם keyboard_typos@0.06
    demo_noise, demo_alpha = "keyboard_typos", 0.06
    samples["noised"] = apply_noise_batch([Xva_txt[i] for i in samp_idx], demo_noise, demo_alpha)
    samples.to_csv(out_dir / "samples_clean_vs_noised.csv", index=False)

    meta = {
        "label_mapping": {"fake": 0, "real": 1},
        "artifacts_used": str(artifacts_dir) if artifacts_dir else None,
        "baseline_clean": {"pr_auc": pr_auc_clean, "f1_fake@0.5": f1_clean},
        "intensities": intensities,
    }
    with open(out_dir / "meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    _log(f"[noise] saved results to: {out_dir}")
    return {"out_dir": str(out_dir), "baseline": meta["baseline_clean"], "results": rows}