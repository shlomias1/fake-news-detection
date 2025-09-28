# src/service/predict_explain_cli.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import joblib

from service.predictor import FakeNewsPredictor, FAKE_LABEL, REAL_LABEL
from sklearn.pipeline import make_pipeline

# --- LIME ---
try:
    from lime.lime_text import LimeTextExplainer
    _HAS_LIME = True
except Exception as _e:
    _HAS_LIME = False
    _LIME_ERR = str(_e)

# --- SHAP ---
try:
    import shap
    _HAS_SHAP = True
except Exception as _e:
    _HAS_SHAP = False
    _SHAP_ERR = str(_e)

def _build_lime_explainer(clf):
    # class_names בסדר תואם ל- clf.classes_
    classes = list(clf.classes_)
    class_names = [f"FAKE({FAKE_LABEL})" if c == FAKE_LABEL else f"REAL({REAL_LABEL})" for c in classes]
    explainer = LimeTextExplainer(class_names=class_names, split_expression=r"\W+")
    return explainer

def _explain_lime(explainer, pipe, raw_text, num_features=10, html_path: Path | None = None):
    exp = explainer.explain_instance(
        text_instance=raw_text,
        classifier_fn=lambda xs: pipe.predict_proba(xs),
        num_features=num_features
    )
    out = {
        "top_features": [(tok, float(w)) for tok, w in exp.as_list()],
    }
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(exp.as_html(), encoding="utf-8")
        out["html_saved_to"] = str(html_path)
    return out

def _explain_shap(vect, clf, raw_text, background_texts=None, topk=20, target_label=FAKE_LABEL):
    if background_texts is None:
        # עדיף לספק דוגמאות אמיתיות ע"י --background_txt / --background_jsonl
        background_texts = ["דוגמה קצרה", "בדיקה נוספת"]
    X_bg = vect.transform(background_texts)

    explainer = shap.LinearExplainer(clf, X_bg, feature_perturbation="interventional")
    X = vect.transform([raw_text])
    sv = explainer.shap_values(X)   # ייתכן list או ndarray, תלוי בגרסת SHAP
    feats = vect.get_feature_names_out()

    if isinstance(sv, list):
        # Multiclass: נבחר את האינדקס של FAKE_LABEL
        idx = list(clf.classes_).index(target_label)
        vals = sv[idx]
    else:
        vals = sv
    vals = vals.toarray().ravel() if hasattr(vals, "toarray") else np.ravel(vals)

    top_idx = np.argsort(np.abs(vals))[::-1][:topk]
    return [(feats[i], float(vals[i])) for i in top_idx]

def _load_background_from_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(l).get("text","") for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # טקסט רגיל: כל שורה דוגמה
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    ap = argparse.ArgumentParser("predict_explain_cli")
    ap.add_argument("--model_dir", required=True, help="Artifacts dir with tfidf.pkl, sgd_logloss.pkl")
    ap.add_argument("--thresholds_json", required=True, help="thresholds_per_context.json from threshold_bandit")
    ap.add_argument("--title", default="")
    ap.add_argument("--text", default="", help="אם יש רק טקסט – שים כאן. אם יש גם כותרת, השתמש בשניהם.")
    ap.add_argument("--source", default="UNK")
    ap.add_argument("--category", default="UNK")
    ap.add_argument("--date", default="")
    ap.add_argument("--lime_num_features", type=int, default=10)
    ap.add_argument("--lime_html", help="נתיב לשמירת דו\"ח LIME HTML (אופציונלי)")
    ap.add_argument("--background_file", help="קובץ רקע ל-SHAP: txt (שורה=דוגמה) או JSONL עם שדה text")
    ap.add_argument("--shap_topk", type=int, default=20)
    ap.add_argument("--out", help="נתיב פלט JSON")
    args = ap.parse_args()

    pred = FakeNewsPredictor(args.model_dir, args.thresholds_json)

    # raw text כמו באימון: title + [SEP] + text
    raw_text = (args.title or "") + (" [SEP] " if args.title else "") + (args.text or "")

    # 1) סיווג עם ספי קונטקסט
    res = pred.predict_one(
        title=args.title, text=args.text,
        meta={"source": args.source, "category": args.category, "date_published": args.date}
    )

    # 2) LIME
    lime_out = {"available": False}
    if _HAS_LIME:
        pipe = make_pipeline(pred.vect, pred.clf)
        lime_explainer = _build_lime_explainer(pred.clf)
        html_path = Path(args.lime_html) if args.lime_html else None
        lime_out = _explain_lime(lime_explainer, pipe, raw_text, num_features=args.lime_num_features, html_path=html_path)
        lime_out["available"] = True
    else:
        lime_out = {"available": False, "error": _LIME_ERR}

    # 3) SHAP
    shap_out = {"available": False}
    if _HAS_SHAP:
        bg = None
        if args.background_file:
            bg = _load_background_from_file(Path(args.background_file))
        try:
            shap_pairs = _explain_shap(pred.vect, pred.clf, raw_text, background_texts=bg, topk=args.shap_topk)
            shap_out = {"available": True, "top_features": shap_pairs}
        except Exception as e:
            shap_out = {"available": False, "error": f"{type(e).__name__}: {e}"}
    else:
        shap_out = {"available": False, "error": _SHAP_ERR}

    out = {
        "prediction": res,               # {"prob_fake":..., "label":..., "context":..., "tau_used":...}
        "lime": lime_out,                # {"available": True, "top_features":[(token,weight),...], "html_saved_to": ...}
        "shap": shap_out                 # {"available": True, "top_features":[(feature,contribution),...]}
    }

    js = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
    else:
        print(js)

if __name__ == "__main__":
    main()
