# explain_shap:py

import polars as pl
import random
random.seed(42)

DF = "/home/shlomias/fake_news_detection/data/df_feat.csv"  # או parquet
import shap, numpy as np, joblib
from sklearn.feature_extraction.text import TfidfVectorizer

ART = "/home/shlomias/fake_news_detection/artifacts_aug/run_20250819-125108"
vect = joblib.load(f"{ART}/tfidf.pkl")
clf  = joblib.load(f"{ART}/sgd_logloss.pkl")

# Choose a small background for stability (several training examples)
df = pl.read_parquet(DF) if DF.endswith(".parquet") else pl.read_csv(DF)

text_col = "text_ns_text" if "text_ns_text" in df.columns else "text"
all_texts = [t or "" for t in df.get_column(text_col).cast(pl.Utf8).fill_null("").to_list()]

k = min(100, len(all_texts))
background_texts = random.sample(all_texts, k=k)

X_bg = vect.transform(background_texts)

explainer = shap.LinearExplainer(clf, X_bg, feature_perturbation="interventional")

def shap_explain(text):
    X = vect.transform([text])
    sv = explainer.shap_values(X)  
    # sv is a matrix of size [1, n_features]; feature→word mapping requires get_feature_names_out()
    feats = vect.get_feature_names_out()
    contrib = sorted(zip(feats, sv[1].toarray()[0]), key=lambda z: abs(z[1]), reverse=True)[:20]
    return contrib
