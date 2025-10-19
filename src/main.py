# /home/shlomias/fake_news_detection/src/main.py
import os, json
from pathlib import Path

from utils.logger import _create_log
from models_runner.main_model import (
    train_simple_aug_text,
    TrainCfg,
    ExternalKnowledgeCfg,
    predict_with_explanation,
)

def _log(msg: str):
    try:
        print(msg, flush=True)
        _create_log(msg)
    except Exception:
        print(msg, flush=True)

def pipeline():
    DF   = "/home/shlomias/fake_news_detection/data/df_feat.csv"  
    OUT  = "/home/shlomias/fake_news_detection/artifacts_simple"
    Path(OUT).mkdir(parents=True, exist_ok=True)

    use_external = bool(os.getenv("OPENAI_API_KEY", "").strip())
    ext_cfg = ExternalKnowledgeCfg(enabled=use_external)

    _log("[MAIN] starting training...")
    train_cfg = TrainCfg(max_features=200_000, aug_frac=0.7, seed=42, test_fraction=0.20)

    res_train = train_simple_aug_text(
        df_path=DF,
        out_dir=OUT,
        ext_cfg=ext_cfg,
        cfg=train_cfg,
    )
    _log(f"[MAIN] Saved artifacts to: {res_train['out_dir']}")
    _log(f"[MAIN] Metrics: {json.dumps(res_train['metrics'], ensure_ascii=False)}")

    MODEL_DIR = OUT
    demo_text = "כאן הטקסט של הכתבה... הרבה סימני קריאה!!!"
    _log("[MAIN] running demo prediction with LIME...")

    res_pred = predict_with_explanation(
        text=demo_text,
        model_dir=MODEL_DIR,
        explain="lime",          # "lime" | "shap" | "none"
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