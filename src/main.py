import preproccesing
import polars as pl
import data_io
import os
import utils.analyze as analyze
import utils.text_utils as text_utils
import utils.scrapper as scrapper
from utils.category_predictor_utils import auto_fill_missing_categories
from proccesing import build_text_features, add_without_stopwords, feat_pipeline
from pathlib import Path
import joblib, json, pandas as pd
from experiments.noise_robustness import run_noise_robustness
import pandas as pd
from models_runner.eval_simple import eval_plain, eval_with_thresholds

import numpy as np
from utils.logger import _create_log
import importlib, torch
from utils.robust_metrics import area_under_degradation

def pipeline():
    BASE = Path("/home/shlomias/fake_news_detection")
    DF   = BASE / "data/df_feat.csv"
    ART  = BASE / "artifacts_aug/run_20250819-125108"
    THJ  = BASE / "artifacts_thresholds/thresholds_per_context.json"
    plain = eval_plain(df_path=str(DF), model_dir=str(ART), val_frac=0.2, seed=42)
    abst  = eval_with_thresholds(df_path=str(DF), model_dir=str(ART),
                                thresholds_json=str(THJ),
                                context_keys=("source","category","len_bin","punc_bin","clickbait"),
                                val_frac=0.2, seed=42)
    print("Plain:", plain)
    print("With abstention:", abst)

if __name__ == "__main__":
    pipeline()
