# models_runner/ga_svm_thresholds.py
from __future__ import annotations
import json, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
import joblib
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from utils.logger import _create_log

import pygad
import gc

# ===========================
#   מיפוי מחלקות בדאטה
# ===========================
FAKE_LABEL = 0  # {'fake': 0, 'real': 1}
REAL_LABEL = 1

def set_seed(seed: int):
    np.random.seed(seed)

# ===========================
#   קונפיגים
# ===========================
@dataclass
class CostConfig:
    ok: float = 1.0
    c_fp: float = 5.0
    c_fn: float = 2.0
    c_abstain: float = 0.5

@dataclass
class GAConfig:
    # GA
    num_generations: int = 20
    sol_per_pop: int = 24
    num_parents_mating: int = 10
    mutation_percent_genes: int = 20
    crossover_type: str = "single_point"
    random_state: int = 42
    # CV
    n_splits: int = 4
    cv_mode: str = "auto"          # "auto" | "group" | "stratified"
    use_group_if_source: bool = True
    # TF-IDF feature caps (None = ללא הגבלה)
    max_features_word: Optional[int] = None
    max_features_char: Optional[int] = None

# ===========================
#   עזר: הסתברות למחלקת FAKE
# ===========================
def proba_fake(clf, X) -> np.ndarray:
    """בחר את עמודת ההסתברויות המתאימה ל-FAKE לפי clf.classes_."""
    idx = int(np.where(clf.classes_ == FAKE_LABEL)[0][0])
    return clf.predict_proba(X)[:, idx]

# ===========================
#   טעינת דאטה והכנה
# ===========================
@dataclass
class Prepared:
    X_txt: List[str]
    X_num: Optional[np.ndarray]
    num_cols: List[str]
    y: np.ndarray
    groups: Optional[List[str]]
    time_ord_idx: Optional[np.ndarray]  # אינדקסים ממוינים לפי זמן (אם יש date_published)

def _ensure_datetime(col: pl.Series) -> pl.Series:
    if col.dtype in (pl.Datetime, pl.Date):
        return col.cast(pl.Datetime)
    try:
        return col.cast(pl.Utf8).str.strptime(pl.Datetime, strict=False)
    except Exception:
        return pl.Series(values=[None]*len(col), dtype=pl.Datetime)

def prepare_data(df: pl.DataFrame) -> Prepared:
    # טקסט מועדף: titleplus_text_ns → title_ns_text → text_ns_text → title + text
    for c in ("titleplus_text_ns", "title_ns_text", "text_ns_text"):
        if c in df.columns:
            X_txt = df[c].cast(pl.Utf8).fill_null("").to_list()
            break
    else:
        t = df.get_column("title").fill_null("").cast(pl.Utf8) if "title" in df.columns else pl.Series([""]*df.height)
        x = df.get_column("text").fill_null("").cast(pl.Utf8)  if "text"  in df.columns else pl.Series([""]*df.height)
        X_txt = (t + pl.lit(" [SEP] ") + x).to_list()

    y = df.get_column("label").cast(pl.Int64).to_numpy()

    # פיצ'רים מספריים: כל Int/Float חוץ מ-label
    num_cols: List[str] = []
    for c, dt in zip(df.columns, df.dtypes):
        if c == "label":
            continue
        sdt = str(dt)
        if sdt.startswith("Int") or sdt.startswith("Float"):
            num_cols.append(c)
    X_num = df.select(num_cols).to_numpy() if num_cols else None

    # קבוצות (source) + סדר זמן
    groups = df.get_column("source").cast(pl.Utf8).fill_null("UNK").to_list() if "source" in df.columns else None

    if "date_published" in df.columns:
        dt = _ensure_datetime(df.get_column("date_published"))
        # אם יש תאריכים—סדר אינדקסים עולה בזמן, נשתמש בו אם יבקשו split בזמן
        order = np.argsort(dt.fill_null(pl.datetime(1970,1,1)).to_numpy())
        time_ord_idx = order
    else:
        time_ord_idx = None

    return Prepared(X_txt=X_txt, X_num=X_num, num_cols=num_cols, y=y, groups=groups, time_ord_idx=time_ord_idx)

# ===========================
#   מדדי מטרה (Fitness)
# ===========================
def pr_auc_reward(p_fake: np.ndarray, y: np.ndarray) -> float:
    """PR-AUC כאשר ה-Positive הוא FAKE (y==0)."""
    y_pos = (y == FAKE_LABEL).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(average_precision_score(y_pos, p_fake))

def cost_reward(
    p_fake: np.ndarray,
    y: np.ndarray,
    tau_low: float,
    tau_high: float,
    cost: CostConfig
) -> float:
    """תגמול מבוסס עלות עבור ספים נתונים; y: 0=fake, 1=real."""
    pred_fake = p_fake > tau_high
    pred_real = p_fake < tau_low
    abstain   = ~(pred_fake | pred_real)

    correct = (pred_fake & (y==FAKE_LABEL)) | (pred_real & (y==REAL_LABEL))
    fp = (pred_fake & (y==REAL_LABEL))
    fn = (pred_real & (y==FAKE_LABEL))

    r = (cost.ok * correct.astype(float)
         - cost.c_fp * fp.astype(float)
         - cost.c_fn * fn.astype(float)
         - cost.c_abstain * abstain.astype(float))
    return float(r.mean())

# ===========================
#   בניית וקטורייזרים לפי גנים
# ===========================
def build_vectorizers_from_genes(genes, ga_cfg: GAConfig):
    (word_ng_max, char_pair_idx, min_df, C_log10, use_num, tau_low, tau_high) = genes
    word_ng_max = int(round(word_ng_max))
    char_pair_idx = int(round(char_pair_idx))
    min_df = int(round(min_df))

    # ברירות מחדל בטוחות אם לא סופק בקונפיג:
    max_w = ga_cfg.max_features_word or 100_000
    max_c = ga_cfg.max_features_char or 200_000

    char_pairs = [(3,5), (4,6)]
    if char_pair_idx not in (0,1):
        return None, None
    char_ng_min, char_ng_max = char_pairs[char_pair_idx]

    v_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, word_ng_max),
        min_df=min_df,
        sublinear_tf=True,
        max_features=max_w,
        dtype=np.float32,          # ← מפחית חצי זיכרון
    )
    v_char = TfidfVectorizer(
        analyzer="char_wb",        # ← פחות פיצ'רים מ-"char"
        ngram_range=(char_ng_min, char_ng_max),
        min_df=5,                  # ← מעט מחמיר לסינון
        sublinear_tf=True,
        max_features=max_c,
        dtype=np.float32,          # ← float32
    )
    return v_word, v_char

# ===========================
#   הערכת כרומוזום יחיד (CV)
# ===========================
import gc

def evaluate_genes_cv(genes, data, ga_cfg, reward_mode, cost_cfg) -> float:
    (word_ng_max, char_pair_idx, min_df, C_log10, use_num, tau_low, tau_high) = genes
    C = 10.0 ** float(C_log10)
    use_num = bool(round(use_num))
    tau_low = float(tau_low); tau_high = float(tau_high)
    if tau_low >= tau_high:
        return -1e12

    v_word, v_char = build_vectorizers_from_genes(genes, ga_cfg)
    if v_word is None:
        return -1e12

    X_txt, X_num, y, num_cols, groups, time_ord_idx = (
        data.X_txt, data.X_num, data.y, data.num_cols, data.groups, data.time_ord_idx
    )

    # בוחר מפצל
    if ga_cfg.cv_mode == "group" or (ga_cfg.cv_mode == "auto" and ga_cfg.use_group_if_source and groups is not None):
        uniq = len(set(groups)) if groups is not None else 0
        n_splits = min(ga_cfg.n_splits, max(2, uniq)) if uniq >= 2 else 2
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X_txt, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=ga_cfg.n_splits, shuffle=True, random_state=ga_cfg.random_state)
        split_iter = splitter.split(X_txt, y)

    rewards, valid_fold = [], 0

    for tr_idx, va_idx in split_iter:
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
            continue

        Xtr_txt = [X_txt[i] for i in tr_idx]
        Xva_txt = [X_txt[i] for i in va_idx]
        ytr = y[tr_idx]; yva = y[va_idx]

        # TF-IDF (float32)
        Xtr_w = v_word.fit_transform(Xtr_txt)
        Xva_w = v_word.transform(Xva_txt)
        Xtr_c = v_char.fit_transform(Xtr_txt)
        Xva_c = v_char.transform(Xva_txt)

        Xtr = hstack([Xtr_w, Xtr_c], format="csr", dtype=np.float32)
        Xva = hstack([Xva_w, Xva_c], format="csr", dtype=np.float32)

        # שחרור זיכרון ביניים
        del Xtr_w, Xva_w, Xtr_c, Xva_c
        gc.collect()

        # פיצ'רים מספריים (אם יש), ודא float32
        if use_num and (X_num is not None) and (len(num_cols) > 0):
            # MaxAbsScaler שומר דלילות וקליל יותר בזיכרון
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler(copy=False)
            Xtr_num = scaler.fit_transform(X_num[tr_idx]).astype(np.float32)
            Xva_num = scaler.transform(X_num[va_idx]).astype(np.float32)
            Xtr = hstack([Xtr, csr_matrix(Xtr_num)], format="csr", dtype=np.float32)
            Xva = hstack([Xva, csr_matrix(Xva_num)], format="csr", dtype=np.float32)
            del Xtr_num, Xva_num
            gc.collect()

        if reward_mode == "pr_auc":
            # בלי כיול: מספיק דירוג ה-margin
            base = LinearSVC(C=C, class_weight="balanced", dual=False, random_state=ga_cfg.random_state)
            base.fit(Xtr, ytr)
            margin = base.decision_function(Xva).astype(np.float32)
            # כיוון הסימן כך ש"גדול יותר" = FAKE (label 0)
            # LinearSVC.classes_ מסודרות עולה; margin חיובי = מחלקה classes_[1]
            if base.classes_[1] == FAKE_LABEL:
                scores = margin
            else:
                scores = -margin
            r = pr_auc_reward(scores, yva)
        else:  # "cost" – צריך הסתברויות
            base = LinearSVC(C=C, class_weight="balanced", dual=False, random_state=ga_cfg.random_state)
            # כיול CV=2 כדי לחסוך זיכרון
            clf  = CalibratedClassifierCV(base, cv=2, method="sigmoid")
            clf.fit(Xtr, ytr)
            p_fake_va = proba_fake(clf, Xva)
            r = cost_reward(p_fake_va, yva, tau_low=tau_low, tau_high=tau_high, cost=cost_cfg)

        rewards.append(r)
        valid_fold += 1

        # ניקוי אגרסיבי בכל קיפול
        del Xtr, Xva
        gc.collect()

    if valid_fold == 0:
        return -1e12
    return float(np.mean(rewards))

# ===========================
#   הרצת GA
# ===========================
def run_ga_on_df(
    df: pl.DataFrame,
    ga_cfg: GAConfig = GAConfig(),
    reward_mode: str = "pr_auc",  # או "cost"
    cost_cfg: CostConfig = CostConfig()
) -> Tuple[List[float], float, Dict]:
    """
    מחזיר: best_genes, best_fitness, info
    genes = [word_ng_max, char_pair_idx, min_df, C_log10, use_num, tau_low, tau_high]
    """
    set_seed(ga_cfg.random_state)
    data = prepare_data(df)

    # תיקון/נרמול פתרון לפני הערכה (עמידות למוטציה/קרוסאובר)
    def _repair_solution(sol: List[float]) -> List[float]:
        sol = list(sol)
        # word_ng_max ∈ {1,2}
        sol[0] = float(int(round(sol[0])))
        if sol[0] < 1: sol[0] = 1.0
        if sol[0] > 2: sol[0] = 2.0
        # char_pair_idx ∈ {0,1}
        sol[1] = float(int(round(sol[1])))
        if sol[1] < 0: sol[1] = 0.0
        if sol[1] > 1: sol[1] = 1.0
        # min_df ∈ [2..20] שלם
        sol[2] = float(int(round(sol[2])))
        if sol[2] < 2:  sol[2] = 2.0
        if sol[2] > 20: sol[2] = 20.0
        # C_log10 ∈ [-3..2]
        sol[3] = float(sol[3])
        if sol[3] < -3.0: sol[3] = -3.0
        if sol[3] >  2.0: sol[3] =  2.0
        # use_num ∈ {0,1}
        sol[4] = float(int(round(sol[4])))
        if sol[4] < 0: sol[4] = 0.0
        if sol[4] > 1: sol[4] = 1.0
        # tau_low ∈ [0.2..0.5], tau_high ∈ [0.5..0.8], וגם tau_low < tau_high
        sol[5] = float(sol[5]); sol[6] = float(sol[6])
        sol[5] = max(0.2, min(0.5, sol[5]))
        sol[6] = max(0.5, min(0.8, sol[6]))
        if sol[5] >= sol[6]:
            mid = 0.5*(sol[5] + sol[6])
            sol[5] = max(0.2, min(0.49, mid - 0.05))
            sol[6] = min(0.8, max(0.51, mid + 0.05))
            if sol[5] >= sol[6]:  # אם עדיין בעייתי, תן סף דיפולטי חוקי
                sol[5], sol[6] = 0.3, 0.7
        return sol

    # PyGAD 2.20.0: fitness_func(ga_instance, solution, solution_idx)
    def fitness_func(ga_instance, solution, sol_idx):
        try:
            repaired = _repair_solution(solution)
            return evaluate_genes_cv(repaired, data, ga_cfg, reward_mode, cost_cfg)
        except Exception:
            # ענישה קשה אם משהו לא תקין
            return -1e12

    gene_space = [
        [1, 2],                       # word_ng_max
        [0, 1],                       # char_pair_idx -> (3,5) או (4,6)
        {'low': 2, 'high': 20},       # min_df
        {'low': -3.0, 'high': 2.0},   # C_log10
        [0, 1],                       # use_num_features
        {'low': 0.2, 'high': 0.5},    # tau_low
        {'low': 0.5, 'high': 0.8},    # tau_high
    ]
    assert len(gene_space) == 7, f"unexpected gene_space length: {len(gene_space)}"

    # אוכלוסייה התחלתית ידנית → עוקף warning/bug ב-init של PyGAD
    def _sample_gene(g):
        if isinstance(g, list):
            return float(np.random.choice(g))
        return float(np.random.uniform(g['low'], g['high']))

    init_pop = np.array(
        [[_sample_gene(g) for g in gene_space] for _ in range(ga_cfg.sol_per_pop)],
        dtype=float
    )

    ga = pygad.GA(
        num_generations=ga_cfg.num_generations,
        num_parents_mating=ga_cfg.num_parents_mating,
        initial_population=init_pop,          # ← במקום sol_per_pop/num_genes
        mutation_percent_genes=ga_cfg.mutation_percent_genes,
        crossover_type=ga_cfg.crossover_type,
        fitness_func=fitness_func,
        gene_space=gene_space,
        random_seed=ga_cfg.random_state,
        allow_duplicate_genes=True,           # ← מונע את באג generations_completed
        suppress_warnings=True                # ← ולמנוע warning-ים בעייתיים
    )
    ga.run()

    best_sol, best_fit, _ = ga.best_solution()
    # נעביר את האלוף דרך התיקון כדי לשמור עמידות
    best_sol = _repair_solution(best_sol)

    best = {
        "word_ng_max": int(round(best_sol[0])),
        "char_pair": (3,5) if int(round(best_sol[1])) == 0 else (4,6),
        "min_df": int(round(best_sol[2])),
        "C_log10": float(best_sol[3]),
        "use_num_features": bool(round(best_sol[4])),
        "tau_low": float(best_sol[5]),
        "tau_high": float(best_sol[6]),
    }
    info = {"best": best, "history_fitness": ga.best_solutions_fitness}
    return list(map(float, best_sol)), float(best_fit), info

# ===========================
#   אימון סופי ושמירת ארטיפקטים
# ===========================
def train_final_and_save(
    df: pl.DataFrame,
    best_sol: List[float],
    out_dir: str | Path,
    ga_cfg: GAConfig = GAConfig()
) -> Dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(df)

    (word_ng_max, char_pair_idx, min_df, C_log10, use_num, tau_low, tau_high) = best_sol
    C = 10.0 ** float(C_log10)
    use_num = bool(round(use_num))
    tau_low = float(tau_low); tau_high = float(tau_high)

    v_word, v_char = build_vectorizers_from_genes(best_sol, ga_cfg)
    Xw = v_word.fit_transform(data.X_txt)
    Xc = v_char.fit_transform(data.X_txt)
    X  = hstack([Xw, Xc], format="csr")

    scaler = None
    if use_num and (data.X_num is not None) and (len(data.num_cols) > 0):
        scaler = StandardScaler(with_mean=False)
        Xnum = scaler.fit_transform(data.X_num)
        X = hstack([X, csr_matrix(Xnum)], format="csr")

    base = LinearSVC(C=C, class_weight="balanced", dual=False, random_state=ga_cfg.random_state)
    clf  = CalibratedClassifierCV(base, cv=5)
    clf.fit(X, data.y)

    # שמירה
    joblib.dump(v_word, out_dir / "tfidf_word.pkl")
    joblib.dump(v_char, out_dir / "tfidf_char.pkl")
    if scaler is not None:
        joblib.dump({"scaler": scaler, "num_cols": data.num_cols}, out_dir / "num_scaler.pkl")
    joblib.dump(clf, out_dir / "svm_calibrated.pkl")
    with open(out_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"tau_low": tau_low, "tau_high": tau_high}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_solution": {
                "word_ng_max": int(round(word_ng_max)),
                "char_pair": (3,5) if int(round(char_pair_idx)) == 0 else (4,6),
                "min_df": int(round(min_df)),
                "C_log10": float(C_log10),
                "use_num_features": use_num,
                "tau_low": tau_low,
                "tau_high": tau_high,
            },
            "num_feature_cols": data.num_cols,
            "label_mapping": {"fake": 0, "real": 1}
        }, f, ensure_ascii=False, indent=2)
    return {"out_dir": str(out_dir)}

# ===========================
#   חיזוי על כתבה חדשה
# ===========================
def predict_article(
    title: str,
    text: str,
    engineered_numeric: Optional[Dict[str, float]],
    artifacts_dir: str | Path,
    tau_low: Optional[float] = None,
    tau_high: Optional[float] = None
) -> Dict:
    artifacts_dir = Path(artifacts_dir)
    v_word = joblib.load(artifacts_dir / "tfidf_word.pkl")
    v_char = joblib.load(artifacts_dir / "tfidf_char.pkl")
    clf    = joblib.load(artifacts_dir / "svm_calibrated.pkl")

    if tau_low is None or tau_high is None:
        with open(artifacts_dir / "thresholds.json","r",encoding="utf-8") as f:
            th = json.load(f)
        tau_low, tau_high = float(th["tau_low"]), float(th["tau_high"])

    raw = (title or "") + " [SEP] " + (text or "")
    Xw = v_word.transform([raw])
    Xc = v_char.transform([raw])
    X  = hstack([Xw, Xc], format="csr")

    # פיצ'רים מספריים אם נשמרו
    try:
        d = joblib.load(artifacts_dir / "num_scaler.pkl")
        scaler = d["scaler"]; num_cols = d["num_cols"]
        row = np.array([[ (engineered_numeric or {}).get(c, 0.0) for c in num_cols ]])
        Xnum = scaler.transform(row)
        X = hstack([X, csr_matrix(Xnum)], format="csr")
    except FileNotFoundError:
        pass

    p_fake = float(proba_fake(clf, X)[0])
    if p_fake < tau_low:
        label = "REAL"   # 1
    elif p_fake > tau_high:
        label = "FAKE"   # 0
    else:
        label = "ABSTAIN"
    return {"prob_fake": p_fake, "label": label}