# /home/shlomias/fake_news_detection/src/main.py
import os, json
from pathlib import Path
from utils.text_utils import detect_lang, translate_text
from utils.logger import _create_log
from models_runner.main_model import (
    predict_with_explanation,
    ExternalKnowledgeCfg,
)

def _log(msg: str):
    try:
        print(msg, flush=True)
        _create_log(msg)
    except Exception:
        print(msg, flush=True)

def pipeline():

    OUT = "/home/shlomias/fake_news_detection/artifacts_simple"
    MODEL_DIR = OUT

    use_external = bool(os.getenv("OPENAI_API_KEY", "").strip())
    ext_cfg = ExternalKnowledgeCfg(enabled=use_external)

    text = (
        "היום יום ראשון, אני בטוח בכך!!!”"
    )
    src_lang = detect_lang(text)
    if src_lang == "he":
        src_lang = "iw"
    target_lang = "en"
    demo_text = translate_text(text, src_lang, target_lang)

    if use_external:
        _log("[MAIN] External knowledge: ENABLED (OPENAI_API_KEY found)")
    else:
        _log("[MAIN] External knowledge: DISABLED (no OPENAI_API_KEY)")

    _log("[MAIN] running demo prediction with LIME...")

    res_pred = predict_with_explanation(
        text=demo_text,
        model_dir=MODEL_DIR,
        explain="lime", 
        top_k=10,
        ext_cfg=ext_cfg,
    )

    pretty = json.dumps(res_pred, ensure_ascii=False, indent=2)
    _log("[MAIN] Demo prediction + explanation:")
    print(pretty)

    pred_out = Path(OUT) / "demo_prediction.json"
    pred_out.write_text(pretty, encoding="utf-8")
    _log(f"[MAIN] Demo prediction saved → {pred_out}")

if __name__ == "__main__":
    pipeline()
