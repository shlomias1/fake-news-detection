from __future__ import annotations
import re, random, json, os, textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import polars as pl
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import scipy.sparse as sp

# הגדרת Backend למניעת שגיאות בתצוגה בשרתים ללא מסך
import matplotlib
matplotlib.use('Agg')

def _to_csr_int32(X):
    X = X.tocsr()
    if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
        X = sp.csr_matrix(
                (X.data, X.indices.astype(np.int32), X.indptr.astype(np.int32)),
                shape=X.shape)
    return X

# =========================
# 0) Label constants
# =========================
FAKE_LABEL = 0  # {'fake': 0, 'real': 1}
REAL_LABEL = 1

# =========================
# 1) Text Augmentation
# =========================
EMOJIS = ["🙂","🔥","❗","❓","😮","🤔","📢","💥"]
HEBREW_NIKKUD = re.compile(r"[\u0591-\u05C7]")

def aug_keyboard_typos(s: str, p: float=0.03) -> str:
    out = []
    abc = "אבגדהוזחטיךכלםמןנסעףפץצקרשתabcdefghijklmnopqrstuvwxyz0123456789"
    for ch in s:
        r = random.random()
        if r < p/3:
            continue
        out.append(ch)
        if p/3 <= r < 2*p/3:
            out.append(ch)
        elif 2*p/3 <= r < p:
            out[-1] = random.choice(abc)
    return "".join(out)

def aug_punct_burst(s: str, p: float=0.2) -> str:
    s = re.sub(r"!", lambda m: "!"*random.choice([1,2,3]) if random.random()<p else "!", s)
    s = re.sub(r"\?", lambda m: "?"*random.choice([1,2,3]) if random.random()<p else "?", s)
    s = re.sub(r"\.\.\.", lambda m: "."*random.choice([3,6,9]) if random.random()<p else m.group(0), s)
    return s

def aug_nikud_space(s: str, p_space: float=0.05) -> str:
    s = HEBREW_NIKKUD.sub("", s)
    s = re.sub(r"\s", lambda m: m.group(0)*(2 if random.random()<p_space else 1), s)
    return s

TOP_AUGS = {
    "keyboard_typos": lambda t: aug_keyboard_typos(t, p=0.03),
    "punct_burst":    lambda t: aug_punct_burst(t, p=0.2),
    "nikud_space":    lambda t: aug_nikud_space(t, p_space=0.05),
}

def apply_mixed_aug(texts: List[str], frac: float=0.7) -> List[str]:
    keys = list(TOP_AUGS.keys())
    out = []
    for t in texts:
        if random.random() < frac:
            k = random.choice(keys)
            out.append(TOP_AUGS[k](t))
        else:
            out.append(t)
    return out

# =========================
# 2) Load data (only text)
# =========================
def load_text_label_only(df_path: str) -> Tuple[List[str], np.ndarray]:
    df = pl.read_parquet(df_path) if df_path.endswith(".parquet") else pl.read_csv(df_path)
    col_candidates = ["text_ns_text", "text"]
    for c in col_candidates:
        if c in df.columns:
            texts = df[c].cast(pl.Utf8).fill_null("").to_list()
            break
    else:
        raise ValueError("לא נמצא עמודת טקסט ('text_ns_text' או 'text').")
    y = df.get_column("label").cast(pl.Int64).to_numpy()  # 0=fake, 1=real
    return texts, y

# =========================
# 3) External Knowledge (LLM)
# =========================
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

@dataclass
class ExternalKnowledgeCfg:
    enabled: bool = False
    max_chars: int = 800
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    timeout_s: int = 30
    system_prompt: str = (
        "You are helping by providing concise and neutral factual context. "
        "Please return only background/definitions/concepts relevant to the topic of the article, without opinions or truth-telling."
    )

def _make_client(cfg: ExternalKnowledgeCfg):
    if not cfg.enabled or OpenAI is None:
        return None
    key = cfg.api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

def fetch_topic_background(article_text: str, cfg: ExternalKnowledgeCfg) -> str:
    if not cfg.enabled:
        return ""
    client = _make_client(cfg)
    if client is None:
        return ""

    user_prompt = textwrap.dedent(f"""
    The following text is a news article or text post.
    Briefly summarize (up to ~{cfg.max_chars} characters) Neutral factual context:
    General concepts/background/events worth knowing to understand the topic.
    Do not declare whether the content is true or false, and do not include personally identifiable information.

    Text:
    \"\"\"{(article_text or '')[:4000]}\"\"\" # Safe truncation
    """).strip()

    try:
        resp = client.chat.completions.create(  # Fixed: changed from responses.create to chat.completions.create
            model=cfg.model,
            messages=[
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=cfg.timeout_s,
        )
        out = (resp.choices[0].message.content or "").strip()
        if len(out) > cfg.max_chars:
            out = out[: cfg.max_chars].rstrip() + "…"
        return out
    except Exception:
        return ""

# =========================
# 4) Training function
# =========================
@dataclass
class TrainCfg:
    max_features: int = 200_000
    aug_frac: float = 0.7
    seed: int = 42
    test_fraction: float = 0.2
    batch_size: int = 50_000
    max_train_docs: Optional[int] = None


def train_simple_aug_text(
    df_path: str,
    out_dir: str | Path,
    *,
    ext_cfg: ExternalKnowledgeCfg = ExternalKnowledgeCfg(enabled=False),
    cfg: TrainCfg = TrainCfg(),
) -> Dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    random.seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    texts, y = load_text_label_only(df_path)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_fraction, random_state=cfg.seed)
    tr_idx, va_idx = next(sss.split(texts, y))
    Xtr_txt = [texts[i] for i in tr_idx]; ytr = y[tr_idx]
    Xva_txt = [texts[i] for i in va_idx]; yva = y[va_idx]

    # External Knowledge Injection (Stub/Real)
    if ext_cfg.enabled:
        # Note: calling the real function here, assuming it's feasible for dataset size
        # or use a pre-computed cache in production.
        def add_bg(t): return t + " || " + fetch_topic_background(t, ext_cfg)[:ext_cfg.max_chars]
        Xtr_txt = [add_bg(t) for t in Xtr_txt]
        Xva_txt = [add_bg(t) for t in Xva_txt]

    if cfg.max_train_docs is not None and cfg.max_train_docs < len(Xtr_txt):
        Xtr_txt = Xtr_txt[:cfg.max_train_docs]
        ytr = ytr[:cfg.max_train_docs]

    # Vectorizer
    vect = TfidfVectorizer(
        ngram_range=(1,2), min_df=5, sublinear_tf=True,
        max_features=cfg.max_features, dtype=np.float32
    )
    print("Fitting Vectorizer...")
    vect.fit(Xtr_txt)

    # Pre-transform Validation set for monitoring
    print("Transforming Validation Set...")
    Xva = vect.transform(Xva_txt)
    Xva = _to_csr_int32(Xva)

    clf = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=cfg.seed)
    classes = np.array([0, 1], dtype=int)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    wmap = {int(c): float(w) for c, w in zip(classes, cw)}
    
    n = len(Xtr_txt)
    bs = max(1, int(cfg.batch_size))
    
    # --- Training Loop with Monitoring ---
    train_history = {"batch": [], "val_accuracy": []}
    
    start = 0
    batch_idx = 0
    
    while start < n:
        end = min(n, start + bs)
        print(f"Training batch {batch_idx+1}: samples {start} to {end}...")
        
        # Prepare Batch
        current_texts = Xtr_txt[start:end]
        X_batch = vect.transform(current_texts)
        X_batch_aug_txt = apply_mixed_aug(current_texts, frac=cfg.aug_frac)
        X_batch_aug = vect.transform(X_batch_aug_txt)
        
        Xb = sp.vstack([X_batch, X_batch_aug], format="csr")
        yb = np.concatenate([ytr[start:end], ytr[start:end]])
        Xb = _to_csr_int32(Xb)
        sw = np.array([wmap[int(yy)] for yy in yb], dtype=np.float32)
        
        clf.partial_fit(Xb, yb, classes=classes, sample_weight=sw)
        
        # Monitor Accuracy
        val_pred = clf.predict(Xva)
        acc_curr = accuracy_score(yva, val_pred)
        train_history["batch"].append(batch_idx + 1)
        train_history["val_accuracy"].append(acc_curr)
        print(f" -> Batch {batch_idx+1} Val Accuracy: {acc_curr:.4f}")
        
        start = end
        batch_idx += 1

    # Final Predictions
    proba = clf.predict_proba(Xva)
    fake_idx = list(clf.classes_).index(0)
    p_va = proba[:, fake_idx] # Prob of being FAKE

    y_fake_true = (yva == 0).astype(int)
    y_fake_pred = (p_va >= 0.5).astype(int)
    y_cls_pred  = np.where(p_va >= 0.5, 0, 1)

    # Metrics
    acc   = accuracy_score(yva, y_cls_pred)
    prauc = average_precision_score(y_fake_true, p_va)
    roc   = roc_auc_score(y_fake_true, p_va)
    f1fk  = f1_score(y_fake_true, y_fake_pred, zero_division=0)

    metrics = {
        "n_val": int(len(yva)),
        "accuracy": float(acc),
        "pr_auc_fake": float(prauc),
        "roc_auc_fake": float(roc),
        "f1_fake@0.5": float(f1fk),
    }

    # =========================
    # Generate Plots
    # =========================
    
    # 1. Training Progress (Learning Curve)
    plt.figure(figsize=(10, 6))
    plt.plot(train_history["batch"], train_history["val_accuracy"], marker='o', linestyle='-')
    plt.title("Training Progress: Validation Accuracy per Batch")
    plt.xlabel("Batch Number")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.savefig(out_dir / "training_progress.png")
    plt.close()

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_fake_true, p_va)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(out_dir / "roc_curve.png")
    plt.close()

    # 3. Confusion Matrix
    cm = confusion_matrix(yva, y_cls_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake (0)', 'Real (1)'], 
                yticklabels=['Fake (0)', 'Real (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    # Save Models
    joblib.dump(vect, out_dir / "tfidf.pkl")
    joblib.dump(clf,  out_dir / "sgd_logloss.pkl")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"out_dir": str(out_dir), "metrics": metrics}

# =========================
# 5) Predict with explanation
# =========================
def predict_with_explanation(
    text: str,
    model_dir: str | Path,
    *,
    explain: str = "lime",   # "lime" | "shap" | "none"
    top_k: int = 10,
    ext_cfg: ExternalKnowledgeCfg = ExternalKnowledgeCfg(enabled=False),
) -> Dict:
    model_dir = Path(model_dir)
    vect = joblib.load(model_dir / "tfidf.pkl")
    clf  = joblib.load(model_dir / "sgd_logloss.pkl")

    raw = text
    if ext_cfg.enabled:
        # Stub logic replaced with actual logic call if needed, 
        # or assuming fetch_topic_background is available.
        ctx = fetch_topic_background(text, ext_cfg)
        if ctx:
             raw += " || " + ctx

    X  = vect.transform([raw])
    X  = _to_csr_int32(X)
    proba = clf.predict_proba(X)
    fake_idx = list(clf.classes_).index(0)
    p_fake = float(proba[:, fake_idx][0])
    label = "FAKE" if p_fake >= 0.5 else "REAL"

    out = {"prob_fake": p_fake, "label": label}

    if explain == "lime":
        from lime.lime_text import LimeTextExplainer

        def clf_proba(text_list):
            Xb = vect.transform(text_list)
            Xb = _to_csr_int32(Xb)
            return clf.predict_proba(Xb)

        explainer = LimeTextExplainer(class_names=["FAKE(0)", "REAL(1)"], split_expression=r"\W+")
        exp = explainer.explain_instance(raw, classifier_fn=clf_proba, num_features=top_k)
        out["explanation"] = {"method":"lime", "weights": [{"token":t, "weight":float(w)} for t,w in exp.as_list()]}

    elif explain == "shap":
        import shap
        import polars as pl
        # Note: Paths here should ideally be configurable or relative
        DF_PATH = "/home/shlomias/fake_news_detection/data/df_feat.csv" 
        if os.path.exists(DF_PATH):
            df = pl.read_parquet(DF_PATH) if DF_PATH.endswith(".parquet") else pl.read_csv(DF_PATH)
            text_col = "text_ns_text" if "text_ns_text" in df.columns else "text"
            all_texts = [t or "" for t in df.get_column(text_col).cast(pl.Utf8).fill_null("").to_list()]
            k = min(100, len(all_texts))
            background_texts = random.sample(all_texts, k=k)
            X_bg = vect.transform(background_texts)
            X_bg = _to_csr_int32(X_bg)
            expl = shap.LinearExplainer(clf, X_bg, feature_perturbation="interventional")
            sv_all = expl.shap_values(X)
            idx = list(clf.classes_).index(0)
            sv = sv_all[idx]
            arr = sv.toarray().ravel() if hasattr(sv, "toarray") else np.ravel(sv)
            feats = vect.get_feature_names_out()
            order = np.argsort(np.abs(arr))[::-1][:top_k]
            out["explanation"] = {"method":"shap",
                                  "weights":[{"feature":feats[i], "shap":float(arr[i])} for i in order]}
        else:
             out["explanation"] = {"error": "Background data for SHAP not found."}

    return out

# ==========================================
# הוסף את זה לסוף הקובץ td_idf_classifier.py
# ==========================================

if __name__ == "__main__":
    # הגדרות נתיבים
    DATA_FILE = "/home/shlomias/fake_news_detection/data/df_feat.csv"
    OUTPUT_DIR = "/home/shlomias/fake_news_detection/artifacts_simple"
    
    print("Starting TF-IDF Training Pipeline...")
    
    # הרצת האימון
    train_simple_aug_text(
        df_path=DATA_FILE,
        out_dir=OUTPUT_DIR,
        cfg=TrainCfg(
            max_features=200_000,
            aug_frac=0.7,
            batch_size=50_000,
            seed=42
        )
    )
    
    print("TF-IDF Pipeline Completed.")
