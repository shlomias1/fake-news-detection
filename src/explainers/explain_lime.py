# explain_lime:py
import joblib
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load artifacts that already exist
ART = "/home/shlomias/fake_news_detection/artifacts_aug/run_20250819-125108"
vect = joblib.load(f"{ART}/tfidf.pkl")
clf  = joblib.load(f"{ART}/sgd_logloss.pkl")

# Build a predict_proba pipeline on a single string
pipe = make_pipeline(vect, clf)
class_names = ["FAKE(0)","REAL(1)"]  # {'fake':0,'real':1}
explainer = LimeTextExplainer(class_names=class_names, split_expression=r"\W+")

def explain_text(text, num_features=10):
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda xs: pipe.predict_proba(xs),
        num_features=num_features
    )
    # exp.as_list() → [(token, weight), ...]
    return exp
