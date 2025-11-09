# models_runner/main_model.py
from __future__ import annotations
import re, random, json, os, textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import polars as pl
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight
import scipy.sparse as sp
import os

def _to_csr_int32(X):
    X = X.tocsr()
    if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
        X = sp.csr_matrix(
                (X.data, X.indices.astype(np.int32), X.indptr.astype(np.int32)), 
                shape=X.shape)
    return X

# =========================
# 0) קבועי תוויות
# =========================
FAKE_LABEL = 0  # {'fake': 0, 'real': 1}
REAL_LABEL = 1

# =========================
# 1) Top Augmantation
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
# 2) loqd data (only text)
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
# 3) using LLM for external knowledge
# =========================
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

@dataclass
class ExternalKnowledgeCfg:
    enabled: bool = False
    max_chars: int = 800          
    model: str = "gpt-4o-mini"    # OpenAI model name
    api_key: Optional[str] = None # OPENAI_API_KEY
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
        resp = client.responses.create(
            model=cfg.model,
            input=[
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=cfg.timeout_s,
        )
        out = (resp.output_text or "").strip()
        if len(out) > cfg.max_chars:
            out = out[: cfg.max_chars].rstrip() + "…"
        return out
    except Exception:
        return ""

def attach_external_context(text: str, cfg: ExternalKnowledgeCfg) -> str:
    if not cfg.enabled:
        return text
    ctx = fetch_topic_background(text, cfg)
    return text if not ctx else f"{text}\n\n[EXT_KNOWLEDGE]\n{ctx}"

# =========================
# 4) training function
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

    # stub
    if ext_cfg.enabled:
        def add_bg(t): return t + " || " + fetch_topic_background_stub(t, ext_cfg)[:ext_cfg.max_chars]
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
    vect.fit(Xtr_txt)  

    clf = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=cfg.seed)
    classes = np.array([0, 1], dtype=int)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    wmap = {int(c): float(w) for c, w in zip(classes, cw)}  
    n = len(Xtr_txt)
    bs = max(1, int(cfg.batch_size))

    # warm start 
    start = 0
    end = min(n, bs)
    X_batch = vect.transform(Xtr_txt[start:end])
    X_batch_aug_txt = apply_mixed_aug(Xtr_txt[start:end], frac=cfg.aug_frac)
    X_batch_aug = vect.transform(X_batch_aug_txt)
    Xb = sp.vstack([X_batch, X_batch_aug], format="csr")
    yb = np.concatenate([ytr[start:end], ytr[start:end]])
    Xb = _to_csr_int32(Xb)
    sw = np.array([wmap[int(yy)] for yy in yb], dtype=np.float32) 
    clf.partial_fit(Xb, yb, classes=classes, sample_weight=sw) 

    start = end
    while start < n:
        end = min(n, start + bs)
        X_batch = vect.transform(Xtr_txt[start:end])
        X_batch_aug_txt = apply_mixed_aug(Xtr_txt[start:end], frac=cfg.aug_frac)
        X_batch_aug = vect.transform(X_batch_aug_txt)
        Xb = sp.vstack([X_batch, X_batch_aug], format="csr")
        yb = np.concatenate([ytr[start:end], ytr[start:end]])
        Xb = _to_csr_int32(Xb)
        sw = np.array([wmap[int(yy)] for yy in yb], dtype=np.float32) 
        clf.partial_fit(Xb, yb, sample_weight=sw) 
        start = end

    Xva = vect.transform(Xva_txt)
    Xva = _to_csr_int32(Xva)
    proba = clf.predict_proba(Xva)
    fake_idx = list(clf.classes_).index(0)
    p_va = proba[:, fake_idx]

    y_fake_true = (yva == 0).astype(int)
    y_fake_pred = (p_va >= 0.5).astype(int)
    y_cls_pred  = np.where(p_va >= 0.5, 0, 1)

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

    joblib.dump(vect, out_dir / "tfidf.pkl")
    joblib.dump(clf,  out_dir / "sgd_logloss.pkl")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"out_dir": str(out_dir), "metrics": metrics}

# =========================
# 5) predict with explanation (LIME/ SHAP)
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
        raw += " || " + fetch_topic_background_stub(text, ext_cfg)

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
        bg = ["", "חדשות", "דיווח", "כתבה", "פרסום"]
        X_bg = vect.transform(bg)
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
    return out