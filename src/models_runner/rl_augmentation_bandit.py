# rl_augmentation_bandit.py
import re, random 
import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.sparse import vstack
import polars as pl
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from datetime import datetime
import joblib, json
import pandas as pd
from utils.logger import _create_log

# ------------- 1) טעינת דאטה -------------
def load_text_label(df_path: str) -> Tuple[List[str], np.ndarray]:
    df = pl.read_parquet(df_path) if df_path.endswith(".parquet") else pl.read_csv(df_path)
    col_candidates = ["titleplus_text_ns", "title_ns_text", "text_ns_text"]
    for c in col_candidates:
        if c in df.columns:
            texts = df[c].to_list()
            break
    else:
        t = df["title"].fill_null("").cast(pl.Utf8)
        x = df["text"].fill_null("").cast(pl.Utf8)
        texts = (t + pl.lit(" [SEP] ") + x).to_list()
    y = df["label"].to_numpy()
    return texts, y

# ------------- 2) אוגמנטציות (זרועות) -------------
EMOJIS = ["🙂", "🔥", "❗", "❓", "😮", "🤔", "📢", "💥"]
HOMO = { 
    "O":"0", "o":"0", "I":"1", "l":"1", "B":"8", "S":"5", "s":"5", "E":"3", "e":"3"
}
HEBREW_NIKKUD = re.compile(r"[\u0591-\u05C7]")

def aug_keyboard_typos(s: str, p: float=0.03) -> str:
    out = []
    for ch in s:
        r = random.random()
        if r < p/3:
            continue  
        out.append(ch)
        if p/3 <= r < 2*p/3:
            out.append(ch)  
        elif 2*p/3 <= r < p:
            repl = random.choice("אבגדהוזחטיךכלםמןנסעףפץצקרשתabcdefghijklmnopqrstuvwxyz0123456789")
            out[-1] = repl
    return "".join(out)

def aug_punct_burst(s: str, p: float=0.2) -> str:
    s = re.sub(r"!", lambda m: "!"*random.choice([1,2,3]) if random.random()<p else "!", s)
    s = re.sub(r"\?", lambda m: "?"*random.choice([1,2,3]) if random.random()<p else "?", s)
    s = re.sub(r"\.\.\.", lambda m: "."*random.choice([3,6,9]) if random.random()<p else m.group(0), s)
    return s

def aug_emoji(s: str, p: float=0.2) -> str:
    if random.random() < p:
        return random.choice(EMOJIS) + " " + s
    if random.random() < p:
        return s + " " + random.choice(EMOJIS)
    return s

def aug_homoglyphs(s: str, p: float=0.05) -> str:
    chars = []
    for ch in s:
        if ch in HOMO and random.random() < p:
            chars.append(HOMO[ch])
        else:
            chars.append(ch)
    return "".join(chars)

def aug_nikud_space(s: str, p_add: float=0.03, p_noise_space: float=0.05) -> str:
    s = HEBREW_NIKKUD.sub("", s)
    s = re.sub(r"\s", lambda m: m.group(0)*random.choice([1,2]) if random.random()<p_noise_space else m.group(0), s)
    return s

AUGS: Dict[str, Callable[[str], str]] = {
    "keyboard_typos": aug_keyboard_typos,
    "punct_burst":    aug_punct_burst,
    "emoji":          aug_emoji,
    "homoglyphs":     aug_homoglyphs,
    "nikud_space":    aug_nikud_space,
}

def apply_aug_batch(texts: List[str], aug_name: str, frac: float=0.7) -> List[str]:
    fn = AUGS[aug_name]
    out = []
    for t in texts:
        if random.random() < frac:
            out.append(fn(t))
        else:
            out.append(t)
    return out

# ------------- 3) בנדיט ε-greedy -------------
@dataclass
class EpsGreedyBandit:
    arms: List[str]
    eps: float = 0.2
    alpha: float = 0.2 

    def __post_init__(self):
        self.values = {a: 0.0 for a in self.arms}
        self.counts = {a: 0 for a in self.arms}

    def choose(self) -> str:
        if random.random() < self.eps:
            return random.choice(self.arms)
        best = max(self.values.values())
        cands = [a for a,v in self.values.items() if abs(v-best) < 1e-12]
        return random.choice(cands)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        self.values[arm] = (1-self.alpha)*self.values[arm] + self.alpha*reward

# ------------- 4) אימון אינקרמנטלי עם Aug RL -------------
def run_aug_bandit_rl(
    df_path: str,
    max_features: int = 200_000,
    batch_size: int = 4096,
    mini_epochs: int = 30,
    aug_frac: float = 0.7,
    eps_start: float = 0.3,
    eps_end: float = 0.05,
    random_state: int = 42
):
    random.seed(random_state); np.random.seed(random_state)

    texts, y = load_text_label(df_path)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    tr_idx, va_idx = next(sss.split(texts, y))
    Xtr_txt = [texts[i] for i in tr_idx]; ytr = y[tr_idx]
    Xva_txt = [texts[i] for i in va_idx]; yva = y[va_idx]

    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=ytr)
    class_weight = {c: w for c, w in zip(classes, cw)}

    vect = TfidfVectorizer(ngram_range=(1,2), min_df=5, sublinear_tf=True, max_features=max_features)
    Xtr_all = vect.fit_transform(Xtr_txt)
    Xva_all = vect.transform(Xva_txt)

    classes = np.array([0,1])
    clf = SGDClassifier(loss="log_loss", class_weight=class_weight, alpha=1e-5, random_state=random_state)

    warm_n = min(20000, Xtr_all.shape[0])
    warm_idx = slice(0, warm_n)
    warm_sw = np.array([class_weight[int(yy)] for yy in ytr[warm_idx]], dtype=np.float32)
    clf.partial_fit(Xtr_all[warm_idx], ytr[warm_idx], classes=classes, sample_weight=warm_sw)

    # bandit
    arms = list(AUGS.keys())
    bandit = EpsGreedyBandit(arms=arms, eps=eps_start)

    # mini-epochs loop
    history = []
    for t in range(mini_epochs):
        bandit.eps = eps_start + (eps_end - eps_start) * (t / max(1, mini_epochs-1))

        arm = bandit.choose()

        idx = np.random.choice(len(Xtr_txt), size=min(batch_size, len(Xtr_txt)), replace=False)
        batch_txt = [Xtr_txt[i] for i in idx]
        batch_y   = ytr[idx]

        aug_txt = apply_aug_batch(batch_txt, arm, frac=aug_frac)

        X_batch = vect.transform(aug_txt)
        batch_sw = np.array([class_weight[int(yy)] for yy in batch_y], dtype=np.float32)
        clf.partial_fit(X_batch, batch_y, classes=classes, sample_weight=batch_sw)

        p_va = clf.predict_proba(Xva_all)[:,1]
        pr_auc = float(average_precision_score(yva, p_va))
        bandit.update(arm, pr_auc)
        history.append({
            "mini_epoch": t+1,
            "arm": arm,
            "reward_pr_auc": pr_auc,
            "eps": bandit.eps,
            "arm_value": bandit.values[arm],
            "counts": bandit.counts[arm],
        })
        if (t+1) % 5 == 0:
            print(f"[{t+1:02d}] arm={arm:12s} PR-AUC={pr_auc:.4f}  eps={bandit.eps:.2f}")

    return {
        "vectorizer": vect,
        "model": clf,
        "bandit_values": bandit.values,
        "bandit_counts": bandit.counts,
        "history": history,
    }

# ------------- 5) שימוש לאחר האימון -------------
def predict_article(title: str, text: str, vect: TfidfVectorizer, clf: SGDClassifier,
                    tau_low=0.3, tau_high=0.7):
    raw = (title or "") + " [SEP] " + (text or "")
    X = vect.transform([raw])
    p = float(clf.predict_proba(X)[0,1])
    if p < tau_low:  return {"label":"REAL", "prob_fake":p}
    if p > tau_high: return {"label":"FAKE", "prob_fake":p}
    return {"label":"ABSTAIN", "prob_fake":p}

def save_artifacts(result: dict, base_dir: str | Path) -> Path:
    base = Path(base_dir)
    out = base / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["vectorizer"], out / "tfidf.pkl")
    joblib.dump(result["model"],     out / "sgd_logloss.pkl")
    pd.DataFrame(result["history"]).to_csv(out / "history.csv", index=False)
    with open(out / "arms.json", "w", encoding="utf-8") as f:
        json.dump(
            {"values": result["bandit_values"], "counts": result["bandit_counts"]},
            f, ensure_ascii=False, indent=2
        )
    print(f"[save_artifacts] Saved to: {out}")
    return out
