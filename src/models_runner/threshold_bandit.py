#threshold_bandit.py
from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from utils.logger import _create_log

FAKE_LABEL = 0   # אצלך: fake=0
REAL_LABEL = 1   # אצלך: real=1

def proba_fake(model, X) -> np.ndarray:
    """החזר p(fake) לפי classes_ של המודל (לא מניח שהעמודה היא 0/1)."""
    idx = int(np.where(model.classes_ == FAKE_LABEL)[0][0])
    return model.predict_proba(X)[:, idx]

@dataclass
class Costs:
    R_correct: float = 1.0
    C_fp: float = 5.0
    C_fn: float = 2.0
    C_abstain: float = 0.5

@dataclass
class Thresholds:
    tau_low: float
    tau_high: float

# ==========================================================
#   1) קונטקסטים: בנייה מדאטה עם באקטינג ותיקוני חסרים
# ==========================================================
def _bin_edges(values: np.ndarray, edges: List[float], labels: List[str]) -> List[str]:
    ix = np.digitize(values, bins=np.array(edges, dtype=float), right=False)  # 0..len(edges)
    # map to labels
    out = []
    for i in ix:
        i = int(i)
        if i < 0: i = 0
        if i >= len(labels): i = len(labels)-1
        out.append(labels[i])
    return out

def _age_days_from_dt(dt: pl.Series, ref) -> np.ndarray:
    n = len(dt)
    if ref is None:
        return np.full(n, 9999, dtype=int)
    try:
        ref_series = pl.Series(name="ref", values=[ref] * n, dtype=pl.Datetime)
        return (ref_series - dt).dt.total_days().fill_null(9999).to_numpy()
    except Exception:
        pass
    out = []
    for t in dt.to_list():
        if t is None:
            out.append(9999)
        else:
            try:
                delta = ref - t
                out.append(int(delta.total_seconds() // 86400))
            except Exception:
                out.append(9999)
    return np.asarray(out, dtype=int)

def _ensure_datetime(col: pl.Series) -> pl.Series:
    try:
        if col.dtype == pl.Datetime or col.dtype == pl.Date:
            return col.cast(pl.Datetime)
    except Exception:
        pass
    s = col.cast(pl.Utf8)
    try:
        return s.str.to_datetime(strict=False, infer_datetime_format=True)
    except Exception:
        try:
            return s.str.strptime(pl.Datetime, strict=False)
        except Exception:
            return pl.Series(name=col.name, values=[None] * len(s), dtype=pl.Datetime)

def build_context(
    df: pl.DataFrame,
    *,
    keys: Iterable[str] = ("source","category","len_bin","punc_bin","clickbait","hour_bin","dow_bin","age_bin"),
    today_ts: Optional[np.datetime64] = None
) -> List[str]:

    n = df.height
    source = (df.get_column("source").fill_null("UNK").cast(pl.Utf8) if "source" in df.columns
              else pl.Series(["UNK"]*n, dtype=pl.Utf8, name="source"))
    category = (df.get_column("category").fill_null("UNK").cast(pl.Utf8) if "category" in df.columns
                else pl.Series(["UNK"]*n, dtype=pl.Utf8, name="category"))

    if "text_n_tokens" in df.columns:
        length_vals = df.get_column("text_n_tokens").fill_null(0).cast(pl.Int64).to_numpy()
    else:
        if "text" in df.columns:
            length_vals = df.get_column("text").fill_null("").cast(pl.Utf8).str.lengths().to_numpy()
        else:
            length_vals = np.zeros(n, dtype=int)
    len_bin = _bin_edges(length_vals, edges=[50, 150, 400], labels=["S","M","L","XL"])

    if "text_exclaim_qm_per_token" in df.columns:
        punc_ratio = df.get_column("text_exclaim_qm_per_token").fill_null(0.0).cast(pl.Float64).to_numpy()
    else:
        punc_ratio = np.zeros(n, dtype=float)
    punc_bin = _bin_edges(punc_ratio, edges=[0.005, 0.015, 0.04], labels=["P0","P1","P2","P3"])

    clickbait = ((df.get_column("title_clickbait_hits").fill_null(0).cast(pl.Int64) > 0).to_numpy()
                 if "title_clickbait_hits" in df.columns else np.zeros(n, dtype=bool))
    clickbait_bin = np.where(clickbait, "CB1", "CB0").tolist()

    if "date_published" in df.columns:
        dt = _ensure_datetime(df.get_column("date_published"))
    else:
        dt = pl.Series(values=[None]*n, dtype=pl.Datetime, name="date_published")

    hour = dt.dt.hour().fill_null(-1).cast(pl.Int64).to_numpy()
    hour_bin = _bin_edges(hour, edges=[6,12,18], labels=["H0","H1","H2","H3"])

    dow = dt.dt.weekday().fill_null(-1).cast(pl.Int64).to_numpy()
    dow_bin = _bin_edges(dow, edges=[1,5], labels=["D0","D1","D2"])  # 0=Mon→D0, 1..5→D1, 6→D2

    if today_ts is None:
        ref = dt.max()  
    else:
        ref = today_ts

    if ref is not None:
        ref_series = pl.Series(name="ref", values=[ref] * n, dtype=pl.Datetime)
        age_days = _age_days_from_dt(dt, ref)
    else:
        age_days = np.full(n, 9999, dtype=int)
    age_bin = _bin_edges(age_days, edges=[3,7,30], labels=["A0","A1","A2","A3"])

    cols = {
        "source": source.to_list(),
        "category": category.to_list(),
        "len_bin": len_bin,
        "punc_bin": punc_bin,
        "clickbait": clickbait_bin,
        "hour_bin": hour_bin,
        "dow_bin": dow_bin,
        "age_bin": age_bin,
    }
    keys = list(keys)
    ctx = []
    for i in range(n):
        parts = []
        for k in keys:
            v = cols[k][i]
            parts.append(f"{k[:3].upper()}={v}")
        ctx.append("|".join(parts))
    return ctx

def reward_vectorized(p: np.ndarray, y: np.ndarray, lo: float, hi: float, costs: Costs) -> float:
    """
    p = p(fake); y: 0=fake, 1=real.
    """
    abst = (p >= lo) & (p <= hi)
    pred_fake = (p > hi)        # חזוי fake → label 0
    pred_real = (p < lo)        # חזוי real → label 1

    correct = (pred_fake & (y == FAKE_LABEL)) | (pred_real & (y == REAL_LABEL))
    fp = (pred_fake & (y == REAL_LABEL))  # חזינו fake אבל בפועל real
    fn = (pred_real & (y == FAKE_LABEL))  # חזינו real אבל בפועל fake

    reward = (costs.R_correct * correct.astype(float)
              - costs.C_fp * fp.astype(float)
              - costs.C_fn * fn.astype(float)
              - costs.C_abstain * abst.astype(float))
    return float(reward.mean())

def grid_search_thresholds_per_context(
    p: np.ndarray,
    y: np.ndarray,
    contexts: List[str],
    *,
    grid_lows: Iterable[float] = np.arange(0.1, 0.6, 0.05),
    grid_highs: Iterable[float] = np.arange(0.4, 0.9, 0.05),
    min_samples: int = 100,
    costs: Costs = Costs()
) -> Tuple[Dict[str, Thresholds], Thresholds, Dict[str, dict]]:

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    contexts = np.asarray(contexts, dtype=object)

    best_global = (-1e18, Thresholds(0.3, 0.7), {"n": len(p), "coverage": None})
    for lo in grid_lows:
        for hi in grid_highs:
            if lo >= hi: 
                continue
            r = reward_vectorized(p, y, lo, hi, costs)
            if r > best_global[0]:
                abst = (p >= lo) & (p <= hi)
                coverage = float((~abst).mean())
                best_global = (r, Thresholds(lo, hi), {"n": len(p), "coverage": coverage})
    global_thresholds = best_global[1]

    thresholds_map: Dict[str, Thresholds] = {}
    stats: Dict[str, dict] = {}
    uniq = np.unique(contexts)
    for ctx in uniq:
        mask = (contexts == ctx)
        n = int(mask.sum())
        if n < min_samples:
            thresholds_map[ctx] = global_thresholds
            stats[ctx] = {"n": n, "used": "global_fallback", "reward_best": None, "coverage": None}
            continue
        p_c = p[mask]; y_c = y[mask]
        best = (-1e18, Thresholds(global_thresholds.tau_low, global_thresholds.tau_high), None)
        for lo in grid_lows:
            for hi in grid_highs:
                if lo >= hi:
                    continue
                r = reward_vectorized(p_c, y_c, lo, hi, costs)
                if r > best[0]:
                    abst = (p_c >= lo) & (p_c <= hi)
                    coverage = float((~abst).mean())
                    best = (r, Thresholds(lo, hi), coverage)
        thresholds_map[ctx] = best[1]
        stats[ctx] = {"n": n, "used": "context", "reward_best": best[0], "coverage": best[2]}
    return thresholds_map, global_thresholds, stats

def apply_thresholds(
    p: np.ndarray,
    contexts: List[str],
    thresholds_map: Dict[str, Thresholds],
    global_thresholds: Thresholds
) -> Tuple[np.ndarray, np.ndarray]:

    y_hat = np.full(len(p), -1, dtype=int)
    abst = np.zeros(len(p), dtype=bool)
    for i, prob in enumerate(p):
        th = thresholds_map.get(contexts[i], global_thresholds)
        if prob < th.tau_low:
            y_hat[i] = REAL_LABEL   # 1
        elif prob > th.tau_high:
            y_hat[i] = FAKE_LABEL   # 0
        else:
            y_hat[i] = -1
            abst[i] = True
    return y_hat, abst

def calibrate_thresholds_per_context(
    df_feat_path: str,
    vectorizer,
    model,
    *,
    costs: Costs = Costs(),
    context_keys: Iterable[str] = ("source","category","len_bin","punc_bin","clickbait"),
    grid_lows: Iterable[float] = np.arange(0.1, 0.6, 0.05),
    grid_highs: Iterable[float] = np.arange(0.4, 0.9, 0.05),
    min_samples: int = 100,
    val_fraction: float = 0.2,
    random_state: int = 42,
    save_json_to: Optional[str | Path] = None
) -> Dict:

    df = pl.read_parquet(df_feat_path) if str(df_feat_path).endswith(".parquet") else pl.read_csv(df_feat_path)
    y = df.get_column("label").cast(pl.Int64).to_numpy()

    for c in ("titleplus_text_ns", "title_ns_text", "text_ns_text"):
        if c in df.columns:
            texts = df.get_column(c).cast(pl.Utf8).to_list()
            break
    else:
        t = df.get_column("title").fill_null("").cast(pl.Utf8) if "title" in df.columns else pl.Series([""]*df.height)
        x = df.get_column("text").fill_null("").cast(pl.Utf8)  if "text"  in df.columns else pl.Series([""]*df.height)
        texts = (t + pl.lit(" [SEP] ") + x).to_list()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    idx_tr, idx_va = next(sss.split(texts, y))
    def sub(lst, idx): return [lst[i] for i in idx]

    X_va = vectorizer.transform(sub(texts, idx_va))
    y_va = y[idx_va]
    # Robust selection של שורות הולידציה ב-Polars (תואם כל גרסאות):
    df_va = (
        df.with_row_count("_rn")
        .filter(pl.col("_rn").is_in(pl.Series(idx_va, dtype=pl.UInt32)))
        .drop("_rn")
    )


    p_va = proba_fake(model, X_va)  # p(fake)

    ctx_va = build_context(df_va, keys=context_keys)

    thresholds_map, global_thresholds, stats = grid_search_thresholds_per_context(
        p=p_va, y=y_va, contexts=ctx_va,
        grid_lows=grid_lows, grid_highs=grid_highs,
        min_samples=min_samples, costs=costs
    )

    y_hat, abst = apply_thresholds(p_va, ctx_va, thresholds_map, global_thresholds)
    coverage = float((~abst).mean())
    correct = (y_hat != -1) & (y_hat == y_va)
    fp = (y_hat == FAKE_LABEL) & (y_va == REAL_LABEL) 
    fn = (y_hat == REAL_LABEL) & (y_va == FAKE_LABEL) 
    acc_on_covered = float(correct.sum() / max(1, (~abst).sum()))
    reward = reward_vectorized(p_va, y_va, global_thresholds.tau_low, global_thresholds.tau_high, costs)  # אינדיקציה גלובלית
    total_reward = 0.0
    for i in range(len(p_va)):
        th = thresholds_map.get(ctx_va[i], global_thresholds)
        total_reward += reward_vectorized(np.array([p_va[i]]), np.array([y_va[i]]), th.tau_low, th.tau_high, costs)
    total_reward /= len(p_va)

    result = {
        "thresholds_map": {k: {"tau_low": v.tau_low, "tau_high": v.tau_high} for k, v in thresholds_map.items()},
        "global_thresholds": {"tau_low": global_thresholds.tau_low, "tau_high": global_thresholds.tau_high},
        "stats_per_context": stats,
        "validation": {
            "n": int(len(y_va)),
            "coverage": coverage,
            "acc_on_covered": float(correct[correct].size / max(1, (~abst).sum())),
            "fp": int(fp.sum()),
            "fn": int(fn.sum()),
            "abstain": int(abst.sum()),
            "reward_estimate": float(total_reward),
        }
    }

    if save_json_to is not None:
        out = Path(save_json_to)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[threshold_bandit] Saved thresholds JSON → {out}")
    return result