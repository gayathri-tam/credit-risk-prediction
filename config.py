"""
Configuration and feature mapping for the Loan Risk Prediction System.
Maps application_train.csv-aligned columns to business-friendly names.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inference: unlabeled data for customer lookup & SHAP background (no TARGET required)
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
# Fallback: Home Credit often uses application.csv
DATA_PATH_ALT = os.path.join(BASE_DIR, "application.csv")

# Training: labeled data with TARGET column (e.g. application_train.csv from Home Credit)
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "application_train.csv")
# Fallback to DATA_PATH if application_train.csv does not exist
TRAINING_DATA_PATH_ALT = os.path.join(BASE_DIR, "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
ENCODER_MAPPINGS_PATH = os.path.join(MODEL_DIR, "encoder_mappings.joblib")
EVALUATION_METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_metrics.joblib")
EVALUATION_METRICS_JSON = os.path.join(MODEL_DIR, "evaluation_metrics.json")

# Target
TARGET_COL = "TARGET"
ID_COL = "Customer_ID"

# Features used by the model (subset aligned with user inputs + dataset)
FEATURE_COLUMNS = [
    "CODE_GENDER",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "REGION_RATING_CLIENT",
    "DAYS_REGISTRATION",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "DAYS_ID_PUBLISH",
]

# Business-friendly names for SHAP and UI
FEATURE_DISPLAY_NAMES = {
    "CODE_GENDER": "Gender",
    "CNT_CHILDREN": "Number of Children",
    "AMT_INCOME_TOTAL": "Annual Income",
    "AMT_CREDIT": "Loan Amount",
    "NAME_INCOME_TYPE": "Income Type",
    "NAME_EDUCATION_TYPE": "Education Level",
    "NAME_FAMILY_STATUS": "Marital Status",
    "NAME_HOUSING_TYPE": "Housing Type",
    "DAYS_BIRTH": "Age (years)",
    "DAYS_EMPLOYED": "Employment Length (years)",
    "OCCUPATION_TYPE": "Occupation Type",
    "ORGANIZATION_TYPE": "Employment Type",
    "REGION_RATING_CLIENT": "Region Rating",
    "DAYS_REGISTRATION": "Years at Current Residence",
    "EXT_SOURCE_1": "Credit Score (Source 1)",
    "EXT_SOURCE_2": "Credit Score (Source 2)",
    "EXT_SOURCE_3": "Credit Score (Source 3)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Existing Credit Enquiries (Year)",
    "DAYS_ID_PUBLISH": "Credit History Length (days)",
}

# Categorical columns (for encoding)
CATEGORICAL_COLUMNS = [
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

# Risk categories
RISK_CATEGORIES = {
    "Low": (0.0, 0.33),
    "Medium": (0.33, 0.66),
    "High": (0.66, 1.01),
}

# Region rating display (proxy for Urban / Semi-urban / Rural)
REGION_RATING_LABELS = {1: "Urban", 2: "Semi-urban", 3: "Rural"}

# Admin password for View Records (entered via web UI, not terminal). Demo only; use env var in production.
ADMIN_PASSWORD = "RapunZel2567"
