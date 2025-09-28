# utils/robust_metrics.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict

def area_under_degradation(
    rows_or_df: Union[str, Path, pd.DataFrame, List[Dict]],
    *,
    metric_col: str = "pr_auc",    
    name_col: str = "noise",      
    alpha_col: str = "intensity",  
    baseline_key: str = "clean"    
) -> Dict[str, float]:

    if isinstance(rows_or_df, (str, Path)):
        df = pd.read_csv(rows_or_df)
    elif isinstance(rows_or_df, list):
        df = pd.DataFrame(rows_or_df)
    else:
        df = rows_or_df.copy()

    missing = {metric_col, name_col, alpha_col} - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in data: {missing}")

    base = float(df.loc[df[name_col] == baseline_key, metric_col].iloc[0])

    out = {}
    for nm, g in df[df[name_col] != baseline_key].groupby(name_col):
        g = g.sort_values(alpha_col)
        drop = base - g[metric_col].values
        aud = float(np.trapz(drop, g[alpha_col].values)) 
        out[nm] = aud
    return out
