"""
Prediction service: load model, preprocess input, predict, and compute SHAP.
"""

import numpy as np
import pandas as pd
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    ENCODER_MAPPINGS_PATH,
    EVALUATION_METRICS_PATH,
    DATA_PATH,
    DATA_PATH_ALT,
    FEATURE_COLUMNS,
    RISK_CATEGORIES,
    ID_COL,
)
from ml.preprocess import load_preprocessor, transform, prepare_raw_df, get_id_column
from ml.explain import (
    compute_shap_values,
    shap_to_feature_impact,
    get_counterfactual_suggestion,
    get_display_name,
)


def _risk_category(prob):
    for name, (lo, hi) in RISK_CATEGORIES.items():
        if lo <= prob < hi:
            return name
    return "High"


def _risk_score(prob):
    """0-100 risk score (higher = more risk)."""
    return round(float(prob) * 100)


def _get_positive_class_probability(model, X):
    """
    Get probability of the positive class (index 1) from model.predict_proba(X).
    Model-agnostic: works for binary classifiers with output shape (1, 2) or (n, 2).
    Handles binary and multiclass probability arrays safely.
    """
    # Compatibility: older pickled LogisticRegression may lack multi_class (newer sklearn).
    if not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    # Flattened single-sample output, e.g. shape (2,) for binary.
    if proba.ndim == 1:
        return float(proba[1]) if len(proba) > 1 else float(proba[0])
    # (n_samples, n_classes): use first row, positive class index 1 for binary.
    row = proba[0]
    if len(row) >= 2:
        return float(row[1])
    return float(row[0])


def load_model_and_artifacts():
    """Load trained model, preprocessor, and encoder mappings."""
    model = joblib.load(MODEL_PATH)
    preprocessor, encoder_mappings = load_preprocessor()
    return model, preprocessor, encoder_mappings


def load_evaluation_metrics():
    """
    Load offline evaluation metrics computed on the test dataset.
    Returns dict with accuracy, precision, recall, f1_score, roc_auc,
    confusion_matrix, roc_curve (fpr, tpr, thresholds), or None if not found.
    """
    if not os.path.isfile(EVALUATION_METRICS_PATH):
        return None
    return joblib.load(EVALUATION_METRICS_PATH)


def _resolve_data_path():
    """Resolve path to inference data (unlabeled; no TARGET required)."""
    for path in [DATA_PATH, DATA_PATH_ALT]:
        if path and os.path.isfile(path):
            return path
    return None


def get_background_data(n_samples=100):
    """Load a sample from dataset for SHAP background. No TARGET required."""
    path = _resolve_data_path()
    if path is None:
        return None
    df = pd.read_csv(path, nrows=5000)
    df = prepare_raw_df(df)
    df = df[FEATURE_COLUMNS].dropna(how="all")
    if len(df) == 0:
        return None
    return df.sample(n=min(n_samples, len(df)), random_state=42)


def predict_single(
    X_raw_df,
    model=None,
    preprocessor=None,
    encoder_mappings=None,
    compute_shap=True,
    background_df=None,
):
    """
    Run prediction and optional SHAP for a single row (DataFrame with FEATURE_COLUMNS).
    Returns dict: probability, risk_score, risk_category, shap_impact, base_value, counterfactual_text.
    """
    if model is None or preprocessor is None or encoder_mappings is None:
        model, preprocessor, encoder_mappings = load_model_and_artifacts()
    
    for col in FEATURE_COLUMNS:
        if col not in X_raw_df.columns:
            X_raw_df[col] = np.nan
    X = transform(X_raw_df[FEATURE_COLUMNS], preprocessor, encoder_mappings)
    if isinstance(X, pd.DataFrame):
        X = X.values
    prob = _get_positive_class_probability(model, X)
    risk_score = _risk_score(prob)
    risk_category = _risk_category(prob)
    
    result = {
        "probability": prob,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "shap_impact": [],
        "base_value": None,
        "counterfactual_text": None,
    }
    
    if compute_shap:
        if background_df is None:
            background_df = get_background_data()
        if background_df is not None and len(background_df) >= 5:
            X_bg = transform(background_df, preprocessor, encoder_mappings)
            if isinstance(X_bg, pd.DataFrame):
                X_bg = X_bg.values
            base_value, shap_vals, _ = compute_shap_values(
                model, X_bg, X, feature_names=FEATURE_COLUMNS
            )
            if shap_vals is not None:
                shap_one = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
                result["base_value"] = base_value
                result["shap_impact"] = shap_to_feature_impact(
                    shap_one, FEATURE_COLUMNS, base_value, prob
                )
                result["counterfactual_text"] = get_counterfactual_suggestion(
                    result["shap_impact"], prob, target_lower=True
                )
    
    return result


def get_customer_by_id(customer_id):
    """Fetch one row by Customer_ID from dataset; return DataFrame with FEATURE_COLUMNS or None."""
    path = _resolve_data_path()
    if path is None:
        return None
    df = pd.read_csv(path)
    id_col = get_id_column(df)
    row = df[df[id_col].astype(str) == str(customer_id)]
    if row.empty:
        return None
    row = prepare_raw_df(row)
    return row
