# app.py — FastAPI + UI + LIME/SHAP chooser
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import joblib
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import shap
import numpy as np

from src.utils.text_utils import detect_lang, translate_text
import json
from datetime import datetime

# --------------------
# Config
# --------------------
MODEL_DIR = Path(os.getenv("FND_MODEL_DIR", r"C:/Users/HadassaAssayag/OneDrive - Harlan Holding/Desktop/fake_news_detection/artifacts_simple"))

VECT_PATH = MODEL_DIR / "tfidf.pkl"
CLF_PATH  = MODEL_DIR / "sgd_logloss.pkl"

AUTO_TRANSLATE_DEFAULT = os.getenv("AUTO_TRANSLATE", "false").strip().lower() == "true"
TARGET_LANG_DEFAULT    = os.getenv("TARGET_LANG", "en").strip()

def normalize_lang(lang: Optional[str]) -> str:
    if not lang:
        return "auto"
    lang = lang.lower()
    if lang in ("he", "he-il", "heb", "hebrew"): return "iw"
    if lang in ("zh", "zh-cn"): return "zh-CN"
    if lang in ("zh-tw", "zh-hant"): return "zh-TW"
    return lang

PREDICTIONS_FILE = Path("predictions/preds.jsonl")
PREDICTIONS_FILE.parent.mkdir(exist_ok=True) 

def log_prediction(input_text: str, result: dict):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": input_text,
        "result": result
    }
    with PREDICTIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# --------------------
# Load model
# --------------------
vect = joblib.load(VECT_PATH)
clf  = joblib.load(CLF_PATH)
pipe = make_pipeline(vect, clf)
FAKE_LABEL = 0  # {'fake':0,'real':1}

# --------------------
# LIME explainer
# --------------------
explainer_lime = LimeTextExplainer(
    class_names=["FAKE(0)", "REAL(1)"],
    split_expression=r"\W+",
)

# --------------------
# SHAP explainer
# --------------------
# We use KernelExplainer (general-purpose)
background = pipe[:-1].transform([""]).toarray()

shap_explainer = shap.KernelExplainer(
    lambda X: pipe[-1].predict_proba(X)[:, 0],
    background
)
# --------------------
# Translation helper
# --------------------
def maybe_translate(text: str, force_translate: bool | None = None, target_lang: str | None = None) -> Dict[str, str]:
    info = {
        "text_in": text, "text_used": text,
        "src_lang": None, "tgt_lang": None,
        "translated": False,
    }

    do_translate = AUTO_TRANSLATE_DEFAULT if force_translate is None else bool(force_translate)
    if not do_translate:
        return info

    try:
        src = detect_lang(text) or "auto"
        src_norm = normalize_lang(src)
        tgt_norm = normalize_lang(target_lang or TARGET_LANG_DEFAULT)
        info["src_lang"] = src_norm
        info["tgt_lang"] = tgt_norm

        if src_norm == tgt_norm:
            return info
        if src_norm == "he":
            src_norm = "iw"
            info["src_lang"] = "iw"

        out = translate_text(text, src_norm, tgt_norm)
        if out and isinstance(out, str) and out.strip():
            info["text_used"] = out
            info["translated"] = True
        return info
    except Exception:
        return info

# --------------------
# Predict with LIME
# --------------------
def predict_with_lime(text: str, top_k: int = 15) -> Dict:
    proba = pipe.predict_proba([text])[0]
    classes = list(clf.classes_)
    fake_idx = classes.index(FAKE_LABEL)
    p_fake = float(proba[fake_idx])
    label  = "FAKE" if p_fake >= 0.5 else "REAL"

    exp = explainer_lime.explain_instance(
        text_instance=text,
        classifier_fn=lambda xs: pipe.predict_proba(xs),
        num_features=top_k,
    )
    weights = []
    for tok, w in exp.as_list():
        weights.append({
            "token": tok,
            "weight": float(w),
            "towards": "FAKE" if w > 0 else "REAL",
        })

    return {"label": label, "prob_fake": p_fake, "top_tokens": weights}

# --------------------
# Predict with SHAP
# --------------------
def predict_with_shap(text: str, top_k: int = 15) -> Dict:
    proba = pipe.predict_proba([text])[0]
    classes = list(clf.classes_)
    fake_idx = classes.index(FAKE_LABEL)
    p_fake = float(proba[fake_idx])
    label  = "FAKE" if p_fake >= 0.5 else "REAL"

    X_text = pipe[:-1].transform([text])
    shap_values = shap_explainer.shap_values(X_text)[0]
    tokens = text.split()

    weights = []
    for tok, val in zip(tokens, shap_values):
        weights.append({
            "token": tok,
            "weight": float(val),
            "towards": "FAKE" if val > 0 else "REAL",
        })

    weights = sorted(weights, key=lambda x: abs(x["weight"]), reverse=True)[:top_k]

    return {"label": label, "prob_fake": p_fake, "top_tokens": weights}

# --------------------
# FastAPI
# --------------------
app = FastAPI(title="Fake News UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------------------
# API
# --------------------
@app.post("/api/predict")
async def api_predict(payload: Dict):
    text = (payload or {}).get("text", "").strip()
    if not text:
        return JSONResponse({"error": "missing 'text'"}, status_code=400)

    force_translate = (payload or {}).get("translate", None)
    target_lang     = (payload or {}).get("target_lang", None)
    tinfo = maybe_translate(text, force_translate=force_translate, target_lang=target_lang)

    # choose explainer
    expl = (payload or {}).get("explainer", "lime").lower()

    if expl == "shap":
        res_pred = predict_with_shap(tinfo["text_used"], top_k=int((payload or {}).get("top_k", 15)))
    else:
        res_pred = predict_with_lime(tinfo["text_used"], top_k=int((payload or {}).get("top_k", 15)))

    res = {
        **res_pred,
        "explainer": expl,
        "translation": {
            "enabled": True,
            "did_translate": tinfo["translated"],
            "src_lang": tinfo["src_lang"],
            "tgt_lang": tinfo["tgt_lang"],
        }
    }
    log_prediction(input_text=text, result=res)
    return JSONResponse(res)

# --------------------
# UI
# --------------------
@app.get("/", response_class=HTMLResponse)
async def ui(_: Request):
    checked   = "checked" if AUTO_TRANSLATE_DEFAULT else ""
    sel_en    = "selected" if TARGET_LANG_DEFAULT == "en" else ""
    sel_iw    = "selected" if TARGET_LANG_DEFAULT == "iw" else ""
    sel_ar    = "selected" if TARGET_LANG_DEFAULT == "ar" else ""
    sel_ru    = "selected" if TARGET_LANG_DEFAULT == "ru" else ""

    html = """
<!doctype html>
<html lang='he' dir='rtl'>
<head>
  <meta charset='utf-8'> 
  <meta name='viewport' content='width=device-width,initial-scale=1'> 
  <title>Fake News Detector</title>
  <script src='https://cdn.tailwindcss.com'></script>
  <style>
    .token-bad { background: rgba(239,68,68,0.15); box-shadow: inset 0 -2px 0 rgba(239,68,68,0.7); }
    .token-good { background: rgba(16,185,129,0.12); box-shadow: inset 0 -2px 0 rgba(16,185,129,0.7); }
    .fade-in { animation: fade .25s ease-in; }
    @keyframes fade { from { opacity:.2 } to { opacity:1 } }
    mark { padding: 0 .15rem; border-radius: .25rem; }
  </style>
</head>
<body class='bg-slate-50 min-h-screen text-slate-800'>
  <div class='max-w-5xl mx-auto p-6'>
    <header class='mb-6'>
      <h1 class='text-3xl font-bold'>בודק פייק ניוז</h1>
      <p class='text-slate-500'>בחר/י הסבר LIME או SHAP, הדבק/י טקסט, ולחצי בדיקה.</p>
    </header>

    <div class='grid gap-4'>
      <textarea id='txt' rows='10' class='w-full p-4 rounded-2xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white' placeholder='הדבק/י כאן את הכתבה...'></textarea>

      <div class='flex items-center flex-wrap gap-4'>
        <button id='btn' class='px-5 py-2.5 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 shadow'>בדיקה</button>

        <label class='inline-flex items-center gap-2 text-sm'>
          <input id='translate' type='checkbox' class='w-4 h-4' __CHECKED__>
          תרגם לפני בדיקה
        </label>

        <label class='inline-flex items-center gap-2 text-sm'>
          שפת יעד:
          <select id='tgt' class='border rounded px-2 py-1 text-sm'>
            <option value='en' __SEL_EN__>en</option>
            <option value='iw' __SEL_IW__>iw</option>
            <option value='ar' __SEL_AR__>ar</option>
            <option value='ru' __SEL_RU__>ru</option>
          </select>
        </label>

        <label class='inline-flex items-center gap-2 text-sm'>
          אלגוריתם הסבר:
          <select id='expl' class='border rounded px-2 py-1 text-sm'>
            <option value='lime'>LIME</option>
            <option value='shap'>SHAP</option>
          </select>
        </label>

        <span id='status' class='text-sm text-slate-500'></span>
      </div>

      <section id='result' class='hidden fade-in'>
        <div class='flex items-center gap-3'>
          <div id='badge' class='px-3 py-1 rounded-full text-white text-sm'></div>
          <div class='w-full bg-slate-200 rounded-full h-2'>
            <div id='bar' class='h-2 rounded-full' style='width:0%'></div>
          </div>
          <span id='prob' class='text-sm font-mono'>0.00</span>
        </div>

        <div class='mt-2 text-xs text-slate-500' id='trinfo'></div>

        <div class='mt-4 bg-white rounded-2xl border border-slate-200 p-4'>
          <h3 class='font-semibold mb-2'>טקסט עם הדגשות</h3>
          <div id='highlight' class='leading-8 whitespace-pre-wrap break-words'></div>
        </div>

        <div class='mt-4 bg-white rounded-2xl border border-slate-200 p-4'>
          <h3 class='font-semibold mb-2'>מאפיינים תורמים</h3>
          <ul id='toplist' class='grid grid-cols-1 md:grid-cols-2 gap-2'></ul>
        </div>
      </section>
    </div>
  </div>

<script>
const btn    = document.getElementById('btn');
const txt    = document.getElementById('txt');
const cbTr   = document.getElementById('translate');
const selTgt = document.getElementById('tgt');
const selExpl= document.getElementById('expl');
const status = document.getElementById('status');
const result = document.getElementById('result');
const badge  = document.getElementById('badge');
const bar    = document.getElementById('bar');
const prob   = document.getElementById('prob');
const hl     = document.getElementById('highlight');
const list   = document.getElementById('toplist');
const trinfo = document.getElementById('trinfo');

btn.addEventListener('click', async () => {
  const text = txt.value.trim();
  if (!text) { status.textContent = 'נא להזין טקסט'; return; }
  status.textContent = 'בודק...';
  result.classList.add('hidden');

  try {
    const r = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        top_k: 20,
        translate: cbTr.checked,
        target_lang: selTgt.value,
        explainer: selExpl.value
      })
    });

    const data = await r.json();
    if (data.error) throw new Error(data.error);

    const isFake = data.label === 'FAKE';
    badge.textContent = isFake ? 'FAKE' : 'REAL';
    badge.className = 'px-3 py-1 rounded-full text-white text-sm ' + (isFake ? 'bg-red-600' : 'bg-emerald-600');

    const pct = Math.round(100 * data.prob_fake);
    bar.style.width = pct + '%';
    bar.className = 'h-2 rounded-full ' + (isFake ? 'bg-red-500' : 'bg-emerald-500');
    prob.textContent = data.prob_fake.toFixed(3);

    const tr = data.translation || {};
    trinfo.textContent = tr.did_translate ? ('בוצע תרגום: ' + (tr.src_lang || '') + ' → ' + (tr.tgt_lang || '')) : '';

    // Map tokens by max weight
    const tokMap = new Map();
    (data.top_tokens || []).forEach(t => {
      const key = (t.token || '').toLowerCase();
      if (!key) return;
      const prev = tokMap.get(key);
      if (!prev || Math.abs(t.weight) > Math.abs(prev.weight)) tokMap.set(key, t);
    });

    // Highlight text
    const parts = text.split(/(\p{L}+|\d+)/u);
    const frag  = document.createDocumentFragment();
    parts.forEach(p => {
      if (!p) return;
      const key = p.toLowerCase();
      const info = tokMap.get(key);
      if (info && ((info.towards === 'FAKE' && isFake) || (info.towards === 'REAL' && !isFake))) {
        const span = document.createElement('mark');
        span.className = info.towards === 'FAKE' ? 'token-bad' : 'token-good';
        span.title = info.towards + ': ' + Number(info.weight).toFixed(3);
        span.textContent = p;
        frag.appendChild(span);
      } else {
        frag.appendChild(document.createTextNode(p));
      }
    });
    hl.innerHTML = '';
    hl.appendChild(frag);

    list.innerHTML = '';
    (data.top_tokens || []).slice(0, 20).forEach(it => {
      const li = document.createElement('li');
      li.className = 'flex items-center justify-between px-3 py-2 rounded-xl border';
      li.innerHTML =
        '<span class="font-mono">' + it.token + '</span>' +
        '<span class="text-xs ' + (it.towards === 'FAKE' ? 'text-red-600' : 'text-emerald-600') + '">' + it.towards + '</span>' +
        '<span class="text-xs text-slate-500">' + Number(it.weight).toFixed(3) + '</span>';
      list.appendChild(li);
    });

    result.classList.remove('hidden');
    status.textContent = '';
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  }
});
</script>
</body>
</html>
    """

    html = (html
            .replace("__CHECKED__", checked)
            .replace("__SEL_EN__", sel_en)
            .replace("__SEL_IW__", sel_iw)
            .replace("__SEL_AR__", sel_ar)
            .replace("__SEL_RU__", sel_ru))

    return HTMLResponse(html)

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', '8000'))
    uvicorn.run(app, host='0.0.0.0', port=port)
