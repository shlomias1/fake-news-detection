# src/service/predictor.py
from __future__ import annotations
from pathlib import Path
import json, datetime as dt, re
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

FAKE_LABEL = 0  # {'fake': 0, 'real': 1}
REAL_LABEL = 1
CLICKBAIT_RE = re.compile(r"(לא תאמינו|הלם|סנסציה|חשיפה|תדהמה|לחצו|מדהים)", re.I)

def _len_tokens(s: str) -> int:
    return len((s or "").split())

def _exclaim_qm_ratio(s: str) -> float:
    t = (s or "")
    return (t.count("!") + t.count("?")) / max(1, _len_tokens(t))

def _clickbait_hits(title: str) -> int:
    return len(CLICKBAIT_RE.findall(title or ""))

def _bin(v, edges, labels):
    i = 0
    while i < len(edges) and v >= edges[i]:
        i += 1
    return labels[min(i, len(labels)-1)]

def build_context_single(rec: dict) -> str:
    source   = rec.get("source")   or "UNK"
    category = rec.get("category") or "UNK"
    text     = rec.get("text")     or ""
    title    = rec.get("title")    or ""

    len_bin  = _bin(_len_tokens(text), [50,150,400], ["S","M","L","XL"])
    punc_bin = _bin(_exclaim_qm_ratio(text), [0.005,0.015,0.04], ["P0","P1","P2","P3"])
    clickbait= "CB1" if _clickbait_hits(title) > 0 else "CB0"

    hour_bin, dow_bin, age_bin = "H0","D0","A3"
    ts = rec.get("date_published")
    try:
        if ts:
            dtobj = dt.datetime.fromisoformat(str(ts).replace("Z","+00:00")) if "T" in str(ts) else dt.datetime.fromisoformat(str(ts))
            hour_bin = _bin(dtobj.hour, [6,12,18], ["H0","H1","H2","H3"])
            dow_bin  = _bin(dtobj.weekday(), [1,5], ["D0","D1","D2"])
            age_days = (dt.datetime.now(dtobj.tzinfo) - dtobj).days
            age_bin  = _bin(age_days, [3,7,30], ["A0","A1","A2","A3"])
    except Exception:
        pass

    parts = [
        f"SOU={source}", f"CAT={category}", f"LEN={len_bin}",
        f"PUN={punc_bin}", f"CLI={clickbait}",
        f"HOU={hour_bin}", f"DOW={dow_bin}", f"AGE={age_bin}",
    ]
    return "|".join(parts)

def _proba_fake_from_clf(clf, X) -> np.ndarray:
    """עמיד למיפוי: מוצא את אינדקס המחלקה FAKE_LABEL=0 ב- clf.classes_."""
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    fake_idx = classes.index(FAKE_LABEL)
    return proba[:, fake_idx]

class FakeNewsPredictor:
    def __init__(
        self,
        model_dir: str | Path,          # artifacts_aug/run_YYYYMMDD-HHMMSS
        thresholds_json: str | Path,    # artifacts_thresholds/thresholds_per_context.json
        tau_fallback: tuple[float,float] = (0.3, 0.7)
    ):
        model_dir = Path(model_dir)
        self.vect = joblib.load(model_dir / "tfidf.pkl")
        self.clf  = joblib.load(model_dir / "sgd_logloss.pkl")

        with open(thresholds_json, "r", encoding="utf-8") as f:
            th = json.load(f)
        self.map_ctx = {k: (v["tau_low"], v["tau_high"]) for k, v in th["thresholds_map"].items()}
        g = th.get("global_thresholds", {})
        self.global_tau = (float(g.get("tau_low", tau_fallback[0])), float(g.get("tau_high", tau_fallback[1])))

    def predict_one(self, *, title: str, text: str, meta: dict | None = None) -> dict:
        raw = (title or "") + " [SEP] " + (text or "")
        X = self.vect.transform([raw])
        p = float(_proba_fake_from_clf(self.clf, X)[0])

        rec = {"title": title, "text": text}
        if meta: rec.update(meta)
        ctx = build_context_single(rec)
        lo, hi = self.map_ctx.get(ctx, self.global_tau)

        if p < lo:   label = "REAL"     # 1
        elif p > hi: label = "FAKE"     # 0
        else:        label = "ABSTAIN"

        return {"prob_fake": p, "label": label, "context": ctx, "tau_used": {"low": lo, "high": hi}}