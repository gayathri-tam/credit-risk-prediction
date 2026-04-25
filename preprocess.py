"""
Preprocessing pipeline for Home Credit dataset.
Handles encoding, scaling, and imputation aligned with application_train.csv.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    ID_COL,
    PREPROCESSOR_PATH,
    ENCODER_MAPPINGS_PATH,
    MODEL_DIR,
)

# Home Credit uses SK_ID_CURR; we standardize to Customer_ID
ID_COL_ALIASES = ["Customer_ID", "SK_ID_CURR"]


def _normalize_id_column(df):
    """Ensure ID column exists; rename SK_ID_CURR to Customer_ID if needed."""
    out = df.copy()
    for alias in ID_COL_ALIASES:
        if alias in out.columns and ID_COL not in out.columns:
            out = out.rename(columns={alias: ID_COL})
            break
    if ID_COL not in out.columns:
        out[ID_COL] = np.arange(len(out))
    return out


def get_id_column(df):
    """Return the ID column name present in df (Customer_ID or SK_ID_CURR)."""
    for alias in ID_COL_ALIASES:
        if alias in df.columns:
            return alias
    return ID_COL


def _days_to_years(days_series):
    """Convert negative days (from application date) to positive years."""
    return (-days_series / 365.25).clip(lower=0)


def prepare_raw_df(df, id_col=None):
    """Select and derive features from raw dataframe. Works with or without TARGET."""
    if id_col is None:
        id_col = ID_COL
    out = df.copy()
    out = _normalize_id_column(out)
    # Ensure we have required columns; create missing with NaN
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[[id_col] + FEATURE_COLUMNS].copy()
    # Clip extreme employment days (e.g. 365243 = pensioner)
    if "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)
    return out


def fit_preprocessor(df):
    """
    Fit imputers, encoders, and scaler on training data.
    Returns fitted preprocessor dict and encoder mappings.
    """
    df = prepare_raw_df(df)
    X = df[FEATURE_COLUMNS].copy()
    encoder_mappings = {}
    
    # Encode categoricals: map each category to integer
    for col in CATEGORICAL_COLUMNS:
        if col not in X.columns:
            continue
        uniques = X[col].dropna().astype(str).unique().tolist()
        encoder_mappings[col] = {v: i for i, v in enumerate(sorted(uniques))}
        X[col] = X[col].astype(str).map(encoder_mappings[col]).replace(np.nan, -1)
    
    # Numeric columns
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    imputer = SimpleImputer(strategy="median")
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    preprocessor = {
        "imputer": imputer,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "categorical_cols": [c for c in CATEGORICAL_COLUMNS if c in FEATURE_COLUMNS],
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(encoder_mappings, ENCODER_MAPPINGS_PATH)
    return preprocessor, encoder_mappings


def _ensure_imputer_compat(imputer):
    """
    Ensure SimpleImputer has _fill_dtype for sklearn version compatibility.
    Newer sklearn adds _fill_dtype in fit(); older or cross-version pickles may lack it.
    """
    if not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "statistics_"):
        try:
            imputer._fill_dtype = np.asarray(imputer.statistics_).dtype
        except Exception:
            imputer._fill_dtype = np.float64


def transform(X_raw, preprocessor, encoder_mappings):
    """
    Transform raw feature dataframe using fitted preprocessor.
    X_raw: DataFrame with FEATURE_COLUMNS (or subset); can have NaNs.
    """
    X = X_raw[FEATURE_COLUMNS].copy()
    numeric_cols = preprocessor["numeric_cols"]
    cat_cols = preprocessor["categorical_cols"]
    imputer = preprocessor["imputer"]
    _ensure_imputer_compat(imputer)
    
    for col in cat_cols:
        if col not in X.columns:
            X[col] = -1
        else:
            mapping = encoder_mappings.get(col, {})
            X[col] = X[col].astype(str).map(lambda x: mapping.get(x, -1)).fillna(-1).astype(int)
    
    X[numeric_cols] = imputer.transform(X[numeric_cols])
    X[numeric_cols] = preprocessor["scaler"].transform(X[numeric_cols])
    return X


def load_preprocessor():
    """Load saved preprocessor and encoder mappings."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    encoder_mappings = joblib.load(ENCODER_MAPPINGS_PATH)
    _ensure_imputer_compat(preprocessor["imputer"])
    return preprocessor, encoder_mappings
