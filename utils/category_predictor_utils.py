import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import polars as pl
import os

class CategoryPredictor:
    def __init__(self, df, target_col, text_cols, cat_cols, model_output_prefix="model"):
        self.df = df.copy()
        self.df_original = df.copy()
        self.target_col = target_col
        self.text_cols = text_cols
        self.cat_cols = cat_cols
        self.model_output_prefix = model_output_prefix
        self.rf_model = None
        self.xgb_model = None

    def prepare_data(self):
        self.df = self.df[self.df[self.target_col].str.lower() != "unknown"].copy()
        self.df[self.target_col] = self.df[self.target_col].str.lower()
        self.X = self.df[self.text_cols + self.cat_cols]
        self.y = self.df[self.target_col]
        text_transformers = [
            (f"text_{col}", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                ("tfidf", TfidfVectorizer(max_features=300))
            ]), col)
            for col in self.text_cols
        ]
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers = text_transformers + [("cat", cat_transformer, self.cat_cols)]
        self.preprocessor = ColumnTransformer(transformers)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, stratify=self.y, test_size=0.2, random_state=42
        )

    def train_models(self):
        self.rf_model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.rf_model.fit(self.X_train, self.y_train)

        self.xgb_model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"))
        ])
        self.xgb_model.fit(self.X_train, self.y_train)

    def evaluate(self):
        print(f"\nEvaluation for target: {self.target_col}\n")
        rf_preds = self.rf_model.predict(self.X_test)
        xgb_preds = self.xgb_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_preds)
        xgb_accuracy = accuracy_score(self.y_test, xgb_preds)
        rf_f1 = f1_score(self.y_test, rf_preds, average='weighted')
        xgb_f1 = f1_score(self.y_test, xgb_preds, average='weighted')
        print("✅ Random Forest:")
        print(classification_report(self.y_test, rf_preds))
        print("✅ XGBoost:")
        print(classification_report(self.y_test, xgb_preds))
        print(f"📊 Accuracy - RF: {rf_accuracy:.3f}, XGB: {xgb_accuracy:.3f}")
        print(f"📊 F1 Score - RF: {rf_f1:.3f}, XGB: {xgb_f1:.3f}")
        if xgb_f1 >= rf_f1:
            print("✅ Using XGBoost (better F1)")
            return "xgb"
        else:
            print("✅ Using RandomForest (better F1)")
            return "rf"

    def predict_unknown(self, model_choice="xgb"):
        unknowns = self.df_original[self.df_original[self.target_col].str.lower() == "unknown"].copy()
        if unknowns.empty:
            print("No unknown values to predict.")
            return pd.DataFrame()
        unknown_X = unknowns[self.text_cols + self.cat_cols]
        if model_choice == "xgb":
            predictions = self.xgb_model.predict(unknown_X)
        elif model_choice == "rf":
            predictions = self.rf_model.predict(unknown_X)
        else:
            raise ValueError(f"Unknown model_choice '{model_choice}'. Use 'xgb' or 'rf'.")
        unknowns[f"{self.target_col}_{model_choice}_pred"] = predictions
        return unknowns

    def save_models(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.rf_model, f"{self.model_output_prefix}_{self.target_col}_rf.pkl")
        joblib.dump(self.xgb_model, f"{self.model_output_prefix}_{self.target_col}_xgb.pkl")
        print("Models saved.")

def inject_predictions_to_polars(df_polars, predicted_df, target_col, pred_col_name):
    df_pd = df_polars.to_pandas()
    mask_unknown = df_pd[target_col].str.lower() == "unknown"
    df_pd.loc[mask_unknown, target_col] = predicted_df[pred_col_name].values
    return pl.DataFrame(df_pd)

def runner(df_polars, target_col, cat_cols):
    df_pd = df_polars.to_pandas()
    text_cols = ["title", "text"]
    predictor = CategoryPredictor(df_pd, target_col=target_col, text_cols=text_cols, cat_cols=cat_cols, model_output_prefix=f"{target_col}_predictor_model")
    predictor.prepare_data()
    predictor.train_models()
    best_model = predictor.evaluate()
    predictor.save_models()
    predicted_unknowns = predictor.predict_unknown(model_choice=best_model)
    updated_df_polars = inject_predictions_to_polars(
        df_polars=df_polars,
        predicted_df=predicted_unknowns,
        target_col=target_col,
        pred_col_name=f"{target_col}_{best_model}_pred"
    )
    return updated_df_polars

def auto_fill_missing_categories(df_polars):
    cat_cols = ["source", "category", "author", "platform", "has_image", "rating", "label"]
    updated_df_polars = runner(df_polars, "type")
    cat_cols = ["source", "type", "author", "platform", "has_image", "rating", "label"]
    updated_df_polars = runner(updated_df_polars, "category")
    updated_df_polars.write_csv("data/combined_dataset.csv")
    return updated_df_polars