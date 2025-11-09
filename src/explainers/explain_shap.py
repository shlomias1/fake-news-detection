# explain_shap:py

import shap, numpy as np, joblib
from sklearn.feature_extraction.text import TfidfVectorizer

ART = "/home/shlomias/fake_news_detection/artifacts_aug/run_20250819-125108"
vect = joblib.load(f"{ART}/tfidf.pkl")
clf  = joblib.load(f"{ART}/sgd_logloss.pkl")

# בחר background קטן ליציבות (מספר דוגמאות אימון)
background_texts = ["דוגמה קצרה", "בדיקה נוספת"]  # החלף לתתי-מדגם שלך
X_bg = vect.transform(background_texts)

explainer = shap.LinearExplainer(clf, X_bg, feature_perturbation="interventional")

def shap_explain(text):
    X = vect.transform([text])
    sv = explainer.shap_values(X)  # החזרות לכל מחלקה
    # sv הוא מטריצה בגודל [1, n_features]; מיפוי פיצ'ר→מילה דורש get_feature_names_out()
    feats = vect.get_feature_names_out()
    contrib = sorted(zip(feats, sv[1].toarray()[0]), key=lambda z: abs(z[1]), reverse=True)[:20]
    return contrib
