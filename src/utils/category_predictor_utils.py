import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import polars as pl
import os
from utils.logger import _create_log
from sklearn.preprocessing import LabelEncoder

class CategoryPredictor:
    def __init__(self, df, target_col, text_cols, cat_cols, model_output_prefix="model"):
        self.df = df.copy()
        self.df_original = df.copy()
        self.target_col = target_col
        self.text_cols = text_cols
        self.cat_cols = cat_cols
        self.model_output_prefix = model_output_prefix
        self.xgb_model = None

    def prepare_data(self):
        print(f"[PREPARE] Cleaning dataset for target: {self.target_col}")
        _create_log(f"[PREPARE] Cleaning dataset for target: {self.target_col}","info")
        self.df = self.df[self.df[self.target_col].str.lower() != "unknown"].copy()
        self.df[self.target_col] = self.df[self.target_col].str.lower()
        self.X = self.df[self.text_cols + self.cat_cols]
        self.label_encoder = LabelEncoder()
        self.df[self.target_col] = self.label_encoder.fit_transform(self.df[self.target_col])
        self.y = self.df[self.target_col]
        text_transformers = [
            (f"text_{col}", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                ("flatten", FunctionTransformer(ravel_1d, validate=False)), 
                ("tfidf", TfidfVectorizer(max_features=300))
            ]), [col])
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
        print(f"[PREPARE] Split into train ({len(self.X_train)}) and test ({len(self.X_test)})")
        _create_log(f"[PREPARE] Split into train ({len(self.X_train)}) and test ({len(self.X_test)})","info")
        
    def train_models(self):
        print("[TRAIN] Training XGBoost...")
        _create_log("[TRAIN] Training XGBoost...","info")
        self.xgb_model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", XGBClassifier(eval_metric="mlogloss"))
        ])
        self.xgb_model.fit(self.X_train, self.y_train)
        print("[TRAIN] XGBoost training completed.")
        _create_log("[TRAIN] XGBoost training completed.","info")

    def evaluate(self):
        print(f"\n[EVALUATE] Evaluation for target: {self.target_col}\n")
        _create_log(f"[EVALUATE] Evaluation for target: {self.target_col}", "info")
        xgb_preds = self.xgb_model.predict(self.X_test)

        if hasattr(self, "label_encoder"):
            y_test_labels = self.label_encoder.inverse_transform(self.y_test)
            preds_labels = self.label_encoder.inverse_transform(xgb_preds)
        else:
            y_test_labels = self.y_test
            preds_labels = xgb_preds

        xgb_accuracy = accuracy_score(y_test_labels, preds_labels)
        xgb_f1 = f1_score(y_test_labels, preds_labels, average='weighted')

        print("[EVALUATE] XGBoost Classification Report:")
        _create_log("[EVALUATE] XGBoost Classification Report:", "info")
        report = classification_report(y_test_labels, preds_labels)
        print(report)
        _create_log(report, "info")

        print(f"[EVALUATE] Accuracy - XGB: {xgb_accuracy:.3f}")
        _create_log(f"[EVALUATE] Accuracy - XGB: {xgb_accuracy:.3f}", "info")
        print(f"[EVALUATE] F1 Score - XGB: {xgb_f1:.3f}")
        _create_log(f"[EVALUATE] F1 Score - XGB: {xgb_f1:.3f}", "info")
        return "xgb"

    def predict_unknown(self):
        print(f"[PREDICT] Predicting unknown values for '{self.target_col}' using XGBoost...")
        _create_log(f"[PREDICT] Predicting unknown values for '{self.target_col}' using XGBoost...", "info")

        unknowns = self.df_original[self.df_original[self.target_col].str.lower() == "unknown"].copy()
        if unknowns.empty:
            print("No unknown values to predict.")
            _create_log("No unknown values to predict.", "Warning")
            return pd.DataFrame()

        unknown_X = unknowns[self.text_cols + self.cat_cols]
        predictions = self.xgb_model.predict(unknown_X)
        if hasattr(self, "label_encoder"):
            decoded_preds = self.label_encoder.inverse_transform(predictions)
        else:
            decoded_preds = predictions

        unknowns[f"{self.target_col}_xgb_pred"] = decoded_preds
        print(f"[PREDICT] {len(decoded_preds)} predictions completed.")
        _create_log(f"[PREDICT] {len(decoded_preds)} predictions completed.", "info")
        return unknowns

    def save_models(self):
        print(f"[SAVE] Saving models for target: {self.target_col}")
        _create_log(f"[SAVE] Saving models for target: {self.target_col}", "info")
        os.makedirs("fake_news_detection/models", exist_ok=True)
        xgb_path = os.path.join("fake_news_detection/models", f"{self.model_output_prefix}_{self.target_col}_xgb.pkl")
        joblib.dump(self.xgb_model, xgb_path)
        if hasattr(self, "label_encoder"):
            le_path = os.path.join("fake_news_detection/models", f"{self.model_output_prefix}_{self.target_col}_label_encoder.pkl")
            joblib.dump(self.label_encoder, le_path)
        print(f"[SAVE] Model saved to '{xgb_path}'")
        _create_log(f"[SAVE] Model saved to '{xgb_path}'", "info")


def ravel_1d(x):
    return np.ravel(x)

def inject_predictions_to_polars(df_polars, predicted_df, target_col, pred_col_name):
    print(f"[INJECT] Injecting predictions into Polars DataFrame for '{target_col}'")
    _create_log(f"[INJECT] Injecting predictions into Polars DataFrame for '{target_col}'","info")
    df_pd = df_polars.to_pandas()
    mask_unknown = df_pd[target_col].str.lower() == "unknown"
    df_pd.loc[mask_unknown, target_col] = predicted_df[pred_col_name].values
    print(f"[INJECT] Injected {mask_unknown.sum()} predictions.")
    _create_log(f"[INJECT] Injected {mask_unknown.sum()} predictions.","info")
    return pl.DataFrame(df_pd)

def runner(df_polars, target_col, cat_cols):
    print(f"[RUNNER] Starting prediction for '{target_col}'")
    _create_log(f"[RUNNER] Starting prediction for '{target_col}'","info")
    df_pd = df_polars.to_pandas()
    text_cols = ["title", "text"]
    predictor = CategoryPredictor(df_pd, target_col=target_col, text_cols=text_cols, cat_cols=cat_cols, model_output_prefix=f"{target_col}_predictor_model")
    predictor.prepare_data()
    predictor.train_models()
    best_model = predictor.evaluate()
    predictor.save_models()
    predicted_unknowns = predictor.predict_unknown()
    updated_df_polars = inject_predictions_to_polars(
        df_polars=df_polars,
        predicted_df=predicted_unknowns,
        target_col=target_col,
        pred_col_name=f"{target_col}_{best_model}_pred"
    )
    print(f"[RUNNER] Finished prediction for '{target_col}'")
    _create_log(f"[RUNNER] Finished prediction for '{target_col}'","info")
    return updated_df_polars

def auto_fill_missing_categories(df_polars):
    print("[AUTO] Starting autofill of missing categories.")
    _create_log("[AUTO] Starting autofill of missing categories.","info")
    #cat_cols = ["source", "category", "author", "rating", "label"]
    #updated_df_polars = runner(df_polars, "type",cat_cols)
    #updated_df_polars.write_csv("fake_news_detection/data/combined_dataset_v1.csv")
    cat_cols = ["source", "type", "rating"]
    #updated_df_polars = runner(updated_df_polars, "category",cat_cols)
    updated_df_polars = runner(df_polars, "category",cat_cols)
    updated_df_polars.write_csv("fake_news_detection/data/combined_dataset_v2.csv")
    print("[AUTO] All categories filled and dataset saved.")
    _create_log("[AUTO] All categories filled and dataset saved.","info")
    return updated_df_polars