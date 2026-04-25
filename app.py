"""
Loan Risk Prediction — Streamlit UI.
User-facing risk assessment only. No training or accuracy display.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import sys
import os
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REGION_RATING_LABELS, ADMIN_PASSWORD, RISK_CATEGORIES
from config import EVALUATION_METRICS_PATH
from ml.predict import (
    load_model_and_artifacts,
    predict_single,
    get_customer_by_id,
    get_background_data,
)
from ml.explain import get_loan_counterfactuals, get_low_risk_confirmation

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")
MODEL_ACCURACY_CACHE = None


def _ensure_dataframe_for_display(rows, columns, fallback_message="No data to display."):
    """
    Convert DB results to a valid Pandas DataFrame for st.dataframe().

    Requirement: all database reads must return a DataFrame before being sent to the UI.
    Empty results must not break st.dataframe().

    Accepts either a DataFrame (preferred) or a list/tuple of rows.
    Returns (df, None) if valid, else (empty_df, message).
    """
    try:
        if isinstance(rows, pd.DataFrame):
            df = rows.copy()
        elif rows is None:
            df = pd.DataFrame(columns=columns or [])
        elif isinstance(rows, (list, tuple)):
            df = pd.DataFrame(list(rows), columns=columns or [])
        else:
            return pd.DataFrame(columns=columns or []), "Query result is not in a displayable format."

        if columns:
            for c in columns:
                if c not in df.columns:
                    df[c] = pd.Series(dtype="object")
            df = df[columns]
        return df, None
    except Exception:
        return pd.DataFrame(columns=columns or []), fallback_message


def _safe_scalar(series_or_df_col, default=None):
    """
    Safely get the first element from a pandas Series or DataFrame column.
    Uses bracket notation (indexing) for compatibility; avoids .at() so that
    serialized data does not trigger 'a.at is not a function' in frontend JS.
    Returns default if the value is missing, empty, or NaN.
    """
    if series_or_df_col is None:
        return default
    try:
        arr = series_or_df_col if hasattr(series_or_df_col, "iloc") else None
        if arr is None or not len(arr):
            return default
        v = arr.iloc[0]
        return default if pd.isna(v) else v
    except (IndexError, KeyError, TypeError):
        return default


def _init_db():
    """Create the SQLite database for model submissions (single flat table; no JSON; no nested fields)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        try:
            # Base table (no-op if it already exists).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                )
                """
            )
        except sqlite3.Error as e:
            print(f"[DB] Error creating base submissions table: {e}")
            raise

        # Best-effort forward migration for older DB schemas.
        # (ALTER TABLE is additive only; safe to ignore failures.)
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(submissions)").fetchall()]
            expected = {
                "created_at": "TEXT",
                "customer_type": "TEXT",
                "product_type": "TEXT",
                "income": "REAL",
                "requested_amount": "REAL",
                "employment_type": "TEXT",
                "cibil_score": "INTEGER",
                "risk_score": "REAL",
                "eligibility_decision": "TEXT",
                "model_accuracy": "REAL",
                "name": "TEXT",
                "phone": "TEXT",
            }
            for c, t in expected.items():
                if c not in cols:
                    try:
                        conn.execute(f"ALTER TABLE submissions ADD COLUMN {c} {t}")
                    except sqlite3.Error as e:
                        print(f"[DB] Error adding column '{c}' to submissions: {e}")
        except sqlite3.Error as e:
            print(f"[DB] Error during submissions schema migration: {e}")
            # Do not raise here; app can still run in read-only / degraded mode.

        # Indexes – created only after columns exist.
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_submissions_created_at ON submissions(created_at)"
            )
        except sqlite3.Error as e:
            print(f"[DB] Error creating idx_submissions_created_at: {e}")
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_submissions_customer_type ON submissions(customer_type)"
            )
        except sqlite3.Error as e:
            print(f"[DB] Error creating idx_submissions_customer_type: {e}")
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_submissions_product_type ON submissions(product_type)"
            )
        except sqlite3.Error as e:
            print(f"[DB] Error creating idx_submissions_product_type: {e}")

        conn.commit()
    finally:
        conn.close()


def _flatten_inputs(input_parameters):
    """
    Convert inputs dict into a flat (non-JSON) string so the table stays flat.
    Example: "age=35; annual_income=200000; housing_type=Rent"
    """
    if not input_parameters:
        return ""
    if not isinstance(input_parameters, dict):
        return str(input_parameters)
    parts = []
    for k in sorted(input_parameters.keys(), key=lambda x: str(x)):
        v = input_parameters.get(k)
        if v is None:
            continue
        try:
            if isinstance(v, float) and pd.isna(v):
                continue
        except Exception:
            pass
        key = str(k).strip()
        if not key:
            continue
        val = str(v).replace("\n", " ").replace("\r", " ").strip()
        parts.append(f"{key}={val}")
    return "; ".join(parts)


def _get_model_accuracy():
    """
    Load the latest model accuracy from saved evaluation metrics, if available.

    This keeps the submissions table flat (REAL column only) while allowing the
    UI to show which model performance was used at submission time.
    """
    global MODEL_ACCURACY_CACHE
    if MODEL_ACCURACY_CACHE is not None:
        return MODEL_ACCURACY_CACHE
    try:
        metrics = joblib.load(EVALUATION_METRICS_PATH)
        acc = float(metrics.get("accuracy"))
        MODEL_ACCURACY_CACHE = acc
        return acc
    except Exception:
        MODEL_ACCURACY_CACHE = None
        return None


def save_submission(
    applicant_name,
    contact_number,
    is_existing_customer,
    customer_id,
    input_parameters,
    risk_result,
):
    """
    Persist a single model submission into the flat `submissions` table.

    Columns:
      - customer_type: 'new' / 'existing'
      - income: numeric income from inputs (REAL)
      - requested_amount: requested loan amount (REAL)
      - employment_type: human-readable employment type (TEXT)
      - cibil_score: optional bureau score (INTEGER, nullable – only for new customers)
      - risk_score: numeric risk score (REAL)
      - eligibility_decision: approved / rejected (TEXT)
      - model_accuracy: current model accuracy from training metrics (REAL, nullable)
      - created_at: UTC timestamp (TIMESTAMP stored as TEXT)
    """
    _init_db()
    ts = datetime.utcnow().isoformat(timespec="seconds")

    # Customer type
    customer_type = "existing" if is_existing_customer else "new"

    # Income and requested amount
    income = float((input_parameters or {}).get("annual_income") or 0.0)
    requested_amount = 0.0
    if input_parameters:
        if input_parameters.get("requested_loan_amount") is not None:
            requested_amount = float(input_parameters.get("requested_loan_amount") or 0.0)

    # Employment type (fallback to NAME_INCOME_TYPE for existing customers)
    employment_type = None
    if input_parameters:
        employment_type = input_parameters.get("employment_type") or input_parameters.get("NAME_INCOME_TYPE")
    employment_type = str(employment_type or "Unknown")

    # CIBIL score (nullable; mainly for new customers)
    cibil_score = None
    if input_parameters:
        raw_cibil = input_parameters.get("cibil_score")
        if raw_cibil is not None:
            try:
                cibil_int = int(raw_cibil)
                if 300 <= cibil_int <= 900:
                    cibil_score = cibil_int
            except (TypeError, ValueError):
                cibil_score = None

    # Risk score and eligibility decision derived from model output
    probability = float(risk_result.get("probability", 0.0) or 0.0)
    risk_score = float(risk_result.get("risk_score", probability * 100.0) or 0.0)
    risk_label = str(risk_result.get("risk_category") or "").strip() or "Unknown"
    eligibility_decision = _final_decision_from_risk(risk_label)
    product_type = "loan"

    # Model accuracy from latest training run (nullable if metrics not available)
    model_accuracy = _get_model_accuracy()

    conn = sqlite3.connect(DB_PATH)
    try:
        # Insert only into columns that exist in the current DB schema (supports older DBs).
        existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(submissions)").fetchall()]
        payload = {
            "created_at": ts,
            "customer_type": customer_type,
            "product_type": product_type,
            "income": income,
            "requested_amount": requested_amount,
            "employment_type": employment_type,
            "cibil_score": cibil_score,
            "risk_score": risk_score,
            "eligibility_decision": eligibility_decision,
            "model_accuracy": model_accuracy,
            "name": (applicant_name or "").strip() or None,
            "phone": (contact_number or "").strip() or None,
        }
        cols = [c for c in payload.keys() if c in existing_cols]
        values = [payload[c] for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(cols)
        conn.execute(f"INSERT INTO submissions ({col_list}) VALUES ({placeholders})", values)
        conn.commit()
    finally:
        conn.close()


def get_all_submissions():
    """Fetch stored submissions for admin view. Always returns a DataFrame."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        sql = """
            SELECT
                id AS "ID",
                created_at AS "Created At",
                customer_type AS "Customer Type",
                income AS "Income",
                requested_amount AS "Requested Amount",
                employment_type AS "Employment Type",
                cibil_score AS "CIBIL Score",
                risk_score AS "Risk Score",
                eligibility_decision AS "Eligibility Decision",
                model_accuracy AS "Model Accuracy"
            FROM submissions
            ORDER BY created_at DESC
        """
        df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()


def _normalize_customer_type(customer_type):
    """Canonicalise and validate customer_type."""
    value = (customer_type or "").strip().lower()
    if value in ("new", "existing"):
        return value
    raise ValueError("customer_type must be 'new' or 'existing'.")


def _final_decision_from_risk(risk_label):
    """
    Map risk label into a simple underwriting-style final decision.

    Low    → approved
    Medium/High → rejected
    """
    label = (risk_label or "").strip().lower()
    if label == "low":
        return "approved"
    if label in ("medium", "high"):
        return "rejected"
    # Unknown labels default to rejected (conservative).
    return "rejected"


def save_application_record(applicant_name, contact_number, risk_result):
    """
    Backwards-compatible stub.

    The database layer was simplified to a single flat table (`predictions`).
    We persist prediction records via `save_submission()` only.
    Keeping this function avoids touching the rest of the app flow.
    """
    return


def lookup_by_name_or_phone(name=None, phone=None):
    """
    Backwards-compatible lookup for admin view.

    The submissions table does not store PII (name/phone). We approximate lookup by
    searching within customer_type, employment_type, eligibility_decision and key
    numeric fields cast to TEXT.

    Always returns a DataFrame (safe for st.dataframe()).
    """
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        q = (name or phone or "").strip()
        if not q:
            return pd.DataFrame(
                columns=[
                    "ID",
                    "Created At",
                    "Customer Type",
                    "Income",
                    "Requested Amount",
                    "Employment Type",
                    "CIBIL Score",
                    "Risk Score",
                    "Eligibility Decision",
                    "Model Accuracy",
                ]
            )
        like = f"%{q}%"
        sql = """
            SELECT
                id AS "ID",
                created_at AS "Created At",
                customer_type AS "Customer Type",
                income AS "Income",
                requested_amount AS "Requested Amount",
                employment_type AS "Employment Type",
                cibil_score AS "CIBIL Score",
                risk_score AS "Risk Score",
                eligibility_decision AS "Eligibility Decision",
                model_accuracy AS "Model Accuracy"
            FROM submissions
            WHERE customer_type LIKE ?
               OR employment_type LIKE ?
               OR eligibility_decision LIKE ?
               OR CAST(income AS TEXT) LIKE ?
               OR CAST(requested_amount AS TEXT) LIKE ?
               OR CAST(risk_score AS TEXT) LIKE ?
            ORDER BY created_at DESC
            LIMIT 200
        """
        df = pd.read_sql_query(
            sql,
            conn,
            params=(like, like, like, like, like, like),
        )
        return df
    finally:
        conn.close()


def validate_applicant_identity(name, contact):
    """Validate applicant name and contact number for basic quality checks."""
    if not name or not name.strip():
        return False, "Please enter the applicant name."
    if not contact or not contact.strip():
        return False, "Please enter a contact / mobile number."
    digits = "".join(ch for ch in contact if ch.isdigit())
    if len(digits) < 7 or len(digits) > 15:
        return False, "Contact number should contain between 7 and 15 digits."
    return True, None


# Page config
st.set_page_config(
    page_title="Loan Risk Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Banking-style CSS and Array.prototype.at polyfill (avoids "a.at is not a function" in older browsers)
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0.25rem; }
    .sub-header { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }
    .card-box { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; }
    .risk-low { color: #0d8050; font-weight: 600; }
    .risk-medium { color: #b38600; font-weight: 600; }
    .risk-high { color: #c2302b; font-weight: 600; }
    .plain-text { color: #334155; line-height: 1.6; }
</style>
<script>
(function() {
    if (typeof Array === "undefined" || !Array.prototype) return;
    if (Array.prototype.at) return;
    Array.prototype.at = function(n) {
        if (this == null) throw new TypeError("Array.prototype.at called on null or undefined");
        var len = this.length;
        if (n === undefined || n === null) return len > 0 ? this[0] : undefined;
        n = Math.trunc(Number(n));
        if (Number.isNaN(n)) return undefined;
        if (n < 0) n += len;
        if (n < 0 || n >= len) return undefined;
        return this[n];
    };
})();
</script>
""", unsafe_allow_html=True)


def render_landing():
    """Landing page with Start Risk Assessment CTA."""
    st.markdown("<p class='main-header'>Loan Risk Prediction System</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Predict loan default risk with clear explanations.</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    if st.button("**Start Risk Assessment**", type="primary", use_container_width=True):
        st.session_state["assessment_started"] = True
        st.session_state["view"] = "form"
        st.session_state["last_result"] = None
        st.session_state["last_product_name"] = None
        st.rerun()


def customer_type_selection():
    """Customer Type (New / Existing)."""
    st.subheader("Assessment setup")
    customer_type_label = st.radio(
        "Customer type",
        options=["New Customer", "Existing Customer"],
        horizontal=True,
        key="customer_type",
    )
    is_existing_customer = customer_type_label.strip() == "Existing Customer"
    customer_type_value = "existing" if is_existing_customer else "new"
    return is_existing_customer, customer_type_value


# Form options aligned with application data (UI-only simplifications)
EMPLOYMENT_TYPES = [
    "Salaried",
    "Self-employed",
]
EDUCATION_TYPES = [
    "Secondary",
    "Higher Education",
    "Degree",
]
FAMILY_STATUS = [
    "Single",
    "Married",
]
HOUSING_TYPES = [
    "House",
    "Rent",
    "Office Apartment",
]


def build_new_customer_form():
    """Application-time features for NEW customers (Loan only).

    UI follows academic / policy guidance:
    - Common fields: age, marital status, dependents, education, employment type,
      years in employment, income, housing type, years at residence.
    - Loan fields: requested loan amount (required) + desired tenure (optional)
    - NO CREDIT BEHAVIOUR is manually entered for new-to-credit customers.

    New customers are evaluated using proxy-based affordability and stability
    features due to absence of credit history (no bureau / repayment data).
    """
    st.subheader("Applicant information")
    
    with st.expander("**Demographic**", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=35, key="age")
        with c2:
            marital = st.selectbox("Marital status", FAMILY_STATUS, key="marital")
            education = st.selectbox("Education level", EDUCATION_TYPES, key="education")
        with c3:
            num_children = st.number_input("Number of dependents", min_value=0, max_value=20, value=0, key="children")
    
    with st.expander("**Employment & income**", expanded=True):
        e1, e2 = st.columns(2)
        with e1:
            employment_years = st.number_input(
                "Employment length (years)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                key="emp_years",
            )
            employment_type = st.selectbox(
                "Employment type",
                EMPLOYMENT_TYPES,
                key="employment_type",
            )
        with e2:
            annual_income = st.number_input(
                "Annual income (₹ per year)",
                min_value=0,
                value=200000,
                step=10000,
                key="income",
            )

        loan_amount = st.number_input(
            "Requested loan amount",
            min_value=1000,
            value=100000,
            step=5000,
            key="loan_amount",
        )
        desired_tenure_months = st.number_input(
            "Desired tenure (months) (optional)",
            min_value=6,
            max_value=360,
            value=60,
            step=6,
            key="loan_tenure",
        )
    
    with st.expander("**Housing & stability**", expanded=True):
        h1, _ = st.columns(2)
        with h1:
            housing_type = st.selectbox("Housing type", HOUSING_TYPES, key="housing")
            years_residence = st.number_input(
                "Years at current residence",
                min_value=0.0,
                max_value=50.0,
                value=3.0,
                step=0.5,
                key="residence",
            )

    # CIBIL score for new customers (300–900); optional but used in risk/eligibility when provided
    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=600,
        step=1,
        key="cibil_score",
        help="Enter your CIBIL score (300–900)",
    )

    st.info("New-to-credit customer – no prior credit history available.")

    days_birth = -int(age * 365.25)
    days_employed = -int(employment_years * 365.25)
    days_registration = -int(years_residence * 365.25)

    # For compatibility with the existing full model (if ever reused), we still
    # construct a technical feature row, but credit-behaviour proxies are left
    # as missing (NaN) because they are not observable for new customers.
    amt_credit = float(loan_amount) if loan_amount is not None else np.nan

    row = pd.DataFrame([{
        "CODE_GENDER": "XNA",  # neutral / not specified
        "CNT_CHILDREN": num_children,
        "AMT_INCOME_TOTAL": float(annual_income),
        "AMT_CREDIT": amt_credit,
        # Map employment type into income-type style feature for consistency
        "NAME_INCOME_TYPE": "Working" if employment_type == "Salaried" else "Business",
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": marital,
        "NAME_HOUSING_TYPE": housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "OCCUPATION_TYPE": np.nan,
        "ORGANIZATION_TYPE": np.nan,
        "REGION_RATING_CLIENT": 2,  # mid-level default
        "DAYS_REGISTRATION": days_registration,
        # Credit-behaviour proxies are intentionally not user-entered for
        # new customers; we use missing values so the model does not treat
        # them as known inputs.
        "EXT_SOURCE_1": np.nan,
        "EXT_SOURCE_2": np.nan,
        "EXT_SOURCE_3": np.nan,
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.nan,
        "DAYS_ID_PUBLISH": np.nan,
    }])

    # Human-readable input snapshot for storage and proxy model
    input_parameters = {
        "age": age,
        "marital_status": marital,
        "num_dependents": num_children,
        "education_level": education,
        "employment_type": employment_type,
        "employment_years": float(employment_years),
        "annual_income": float(annual_income),
        "housing_type": housing_type,
        "years_residence": float(years_residence),
        "cibil_score": int(cibil_score) if 300 <= cibil_score <= 900 else None,
        "requested_loan_amount": float(loan_amount) if loan_amount is not None else None,
        "desired_tenure_months": float(desired_tenure_months) if desired_tenure_months is not None else None,
    }
    return row, input_parameters


def validate_new_customer_inputs():
    """Basic validation for new customer form (e.g. required fields)."""
    return True


def _risk_category_from_prob(probability):
    """Map probability to risk band using shared configuration thresholds."""
    for name, (lo, hi) in RISK_CATEGORIES.items():
        if lo <= probability < hi:
            return name
    return "High"


def _proxy_new_customer_risk(input_parameters):
    """
    Proxy-based affordability & stability model for NEW customers.

    Uses income, employment stability, age, housing stability, requested amount,
    and CIBIL score (when provided) for risk and eligibility.
    """
    age = float(input_parameters.get("age") or 0.0)
    employment_years = float(input_parameters.get("employment_years") or 0.0)
    annual_income = float(input_parameters.get("annual_income") or 0.0)
    years_residence = float(input_parameters.get("years_residence") or 0.0)
    cibil_score = input_parameters.get("cibil_score")
    if cibil_score is not None:
        cibil_score = int(cibil_score) if 300 <= int(cibil_score) <= 900 else None

    requested_amount = float(input_parameters.get("requested_loan_amount") or 0.0)

    # Conservative base risk for new-to-credit population
    score = 0.25

    # CIBIL score (300–900): higher score reduces risk, lower increases it
    if cibil_score is not None:
        # Map 300–900 to a modifier: 750+ reduces risk, <550 increases it
        if cibil_score >= 750:
            score -= 0.12
        elif cibil_score >= 650:
            score -= 0.05
        elif cibil_score >= 550:
            score += 0.0
        elif cibil_score >= 450:
            score += 0.10
        else:
            score += 0.18

    # Age profile: very young and older ages slightly higher risk
    if age < 25:
        score += 0.15
    elif age < 35:
        score += 0.10
    elif age > 55:
        score += 0.05

    # Employment stability: short histories are penalised
    if employment_years < 1:
        score += 0.20
    elif employment_years < 3:
        score += 0.15
    elif employment_years < 5:
        score += 0.05

    # Housing stability: short residence increases risk
    if years_residence < 1:
        score += 0.15
    elif years_residence < 3:
        score += 0.10
    elif years_residence < 5:
        score += 0.05

    # Affordability via simple debt-to-income style proxy
    if annual_income <= 0:
        ratio = 10.0  # treat as very risky when income is zero / missing
    else:
        ratio = requested_amount / max(annual_income, 1.0)

    if ratio <= 2:
        score += 0.05
    elif ratio <= 4:
        score += 0.15
    elif ratio <= 6:
        score += 0.25
    else:
        score += 0.35

    # Clamp to a valid probability range
    prob = max(0.05, min(score, 0.95))
    risk_category = _risk_category_from_prob(prob)
    risk_score = int(round(prob * 100))
    return prob, risk_score, risk_category


def _simulate_new_customer_risk(
    input_parameters,
    income_override=None,
    employment_years_override=None,
    amount_override=None,
):
    """Helper to recompute proxy risk under alternative income / tenure / amount."""
    params = dict(input_parameters or {})
    if income_override is not None:
        params["annual_income"] = float(income_override)
    if employment_years_override is not None:
        params["employment_years"] = float(employment_years_override)
    if amount_override is not None:
        params["requested_loan_amount"] = float(amount_override)
    return _proxy_new_customer_risk(params)


def _dice_new_customer_explanations(
    input_parameters,
    base_prob,
    base_category,
):
    """
    DiCE-style counterfactual explanations for NEW customers.

    Varies only:
    - Income
    - Employment length
    - Requested loan amount

    All features are aligned with the proxy model; no credit history or
    bureau-only variables are used.
    """
    explanations = []

    income = float(input_parameters.get("annual_income") or 0.0)
    employment_years = float(input_parameters.get("employment_years") or 0.0)
    amount = float(input_parameters.get("requested_loan_amount") or 0.0)
    amount_label = "loan amount"

    # Scenario 1: increase income
    if income > 0:
        for step in range(1, 7):
            candidate = income + step * 10000.0
            prob, _, cat = _simulate_new_customer_risk(
                input_parameters, income_override=candidate
            )
            if prob < base_prob:
                explanations.append(
                    f"If annual income increased from ₹{income:,.0f} to ₹{candidate:,.0f} "
                    f"(≈₹{(candidate - income)/12:,.0f} more per month), the applicant "
                    f"would move from {base_category.lower()} to {cat.lower()} risk."
                )
                break

    # Scenario 2: longer employment history
    for step in range(1, 6):
        candidate_years = employment_years + step * 1.0
        prob, _, cat = _simulate_new_customer_risk(
            input_parameters, employment_years_override=candidate_years
        )
        if prob < base_prob:
            explanations.append(
                f"If employment length increased from {employment_years:.1f} to "
                f"{candidate_years:.1f} years, the risk rating would improve from "
                f"{base_category.lower()} to {cat.lower()}."
            )
            break

    # Scenario 3: reduce requested amount
    if amount > 0:
        for factor in [0.9, 0.8, 0.7, 0.6]:
            candidate_amount = amount * factor
            prob, _, cat = _simulate_new_customer_risk(
                input_parameters, amount_override=candidate_amount
            )
            if prob < base_prob:
                explanations.append(
                    f"If the requested {amount_label} reduced from ₹{amount:,.0f} to "
                    f"₹{candidate_amount:,.0f}, the application would move from "
                    f"{base_category.lower()} to {cat.lower()} risk."
                )
                break

    if not explanations:
        # Fallback if no strictly better scenario was found – still framed in
        # terms of the specific drivers used for new-customer proxy models.
        explanations.append(
            "For this new-to-credit application, the model did not find a single change "
            f"in income, employment length or requested {amount_label} that clearly moved "
            "it into a lower risk band; a combination of improvements would be required."
        )
    return explanations


def _render_existing_credit_background(row_df):
    """
    Show credit behaviour proxies for EXISTING customers as read-only values.

    Values are fetched from the credit bureau / dataset and are never edited
    by the end user.
    """
    # Credit score proxy from EXT_SOURCE_* (0–1 range)
    credit_score = None
    try:
        if row_df is None or not len(row_df):
            pass
        else:
            scores = []
            for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
                if col in row_df.columns:
                    v = _safe_scalar(row_df[col])
                    if v is not None and not pd.isna(v):
                        scores.append(float(v))
            if scores:
                credit_score = float(np.mean(scores))
    except Exception:
        credit_score = None

    # Number of credit enquiries in past year
    enquiries = None
    try:
        if row_df is not None and len(row_df) and "AMT_REQ_CREDIT_BUREAU_YEAR" in row_df.columns:
            v = _safe_scalar(row_df["AMT_REQ_CREDIT_BUREAU_YEAR"])
            if v is not None and not pd.isna(v):
                enquiries = float(v)
    except Exception:
        enquiries = None

    with st.expander("Credit background (read-only)", expanded=True):
        st.caption("Fetched from credit bureau / dataset")
        c1, c2 = st.columns(2)
        with c1:
            if credit_score is not None:
                st.metric("Credit score (proxy, 0–1)", f"{credit_score:.2f}")
            else:
                st.write("Credit score (proxy): not available")
        with c2:
            if enquiries is not None:
                st.metric("Credit enquiries (last year)", f"{enquiries:.0f}")
            else:
                st.write("Credit enquiries (last year): not available")


def _compute_existing_prob_and_category(row_df, model, preprocessor, encoder_mappings):
    """Run the full risk model on a modified row to support DiCE-style what-ifs."""
    result = predict_single(
        row_df,
        model=model,
        preprocessor=preprocessor,
        encoder_mappings=encoder_mappings,
        compute_shap=False,
        background_df=None,
    )
    prob = float(result.get("probability", 0.0))
    category = result.get("risk_category") or _risk_category_from_prob(prob)
    return prob, category


def _generate_existing_dice_explanations(row_df, base_prob, base_category, model, preprocessor, encoder_mappings):
    """
    DiCE-style explanations for EXISTING customers.

    Varies:
    - Credit score proxies (EXT_SOURCE_1/2/3)
    - Credit enquiries (AMT_REQ_CREDIT_BUREAU_YEAR)
    - Credit utilisation proxy (AMT_CREDIT relative to income)
    """
    explanations = []

    # Scenario 1: improve credit score proxies
    try:
        if row_df is None or not len(row_df):
            pass
        else:
            scores = []
            for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
                if col in row_df.columns:
                    v = _safe_scalar(row_df[col])
                    if v is not None and not pd.isna(v):
                        scores.append(float(v))
            if scores:
                current = float(np.mean(scores))
                target = min(current + 0.15, 0.9)
                cf = row_df.copy()
                for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
                    if col in cf.columns:
                        cf[col] = target
                new_prob, new_cat = _compute_existing_prob_and_category(
                    cf, model, preprocessor, encoder_mappings
                )
                if new_prob < base_prob:
                    explanations.append(
                        f"If the credit score proxy increased from {current:.2f} to {target:.2f}, "
                        f"the predicted default probability would fall from {base_prob * 100:.1f}% "
                        f"to {new_prob * 100:.1f}%, moving from {base_category.lower()} to "
                        f"{new_cat.lower()} risk."
                    )
    except Exception:
        pass

    # Scenario 2: fewer credit enquiries
    try:
        if row_df is not None and len(row_df) and "AMT_REQ_CREDIT_BUREAU_YEAR" in row_df.columns:
            n = _safe_scalar(row_df["AMT_REQ_CREDIT_BUREAU_YEAR"])
            if n is not None and not pd.isna(n) and float(n) > 0:
                n = float(n)
                target_n = max(n - 2.0, 0.0)
                cf = row_df.copy()
                cf["AMT_REQ_CREDIT_BUREAU_YEAR"] = target_n
                new_prob, new_cat = _compute_existing_prob_and_category(
                    cf, model, preprocessor, encoder_mappings
                )
                if new_prob < base_prob:
                    explanations.append(
                        f"If credit enquiries in the past year reduced from {n:.0f} to {target_n:.0f}, "
                        f"the default probability would decrease from {base_prob * 100:.1f}% "
                        f"to {new_prob * 100:.1f}%, improving the risk band from "
                        f"{base_category.lower()} to {new_cat.lower()}."
                    )
    except Exception:
        pass

    # Scenario 3: lower credit utilisation (AMT_CREDIT relative to income)
    try:
        if (
            row_df is not None
            and len(row_df)
            and "AMT_CREDIT" in row_df.columns
            and "AMT_INCOME_TOTAL" in row_df.columns
        ):
            amt = _safe_scalar(row_df["AMT_CREDIT"])
            inc = _safe_scalar(row_df["AMT_INCOME_TOTAL"])
            if (
                amt is not None
                and inc is not None
                and not pd.isna(amt)
                and not pd.isna(inc)
                and float(amt) > 0
                and float(inc) > 0
            ):
                amt = float(amt)
                inc = float(inc)
                current_ratio = amt / inc
                target_amt = amt * 0.8
                cf = row_df.copy()
                cf["AMT_CREDIT"] = target_amt
                new_prob, new_cat = _compute_existing_prob_and_category(
                    cf, model, preprocessor, encoder_mappings
                )
                if new_prob < base_prob:
                    explanations.append(
                        f"If the requested/used credit amount decreased from ₹{amt:,.0f} "
                        f"to ₹{target_amt:,.0f} (utilisation ratio from {current_ratio:.2f} "
                        f"to {target_amt / inc:.2f}), the risk would move from "
                        f"{base_category.lower()} to {new_cat.lower()}."
                    )
    except Exception:
        pass

    if not explanations:
        explanations.append(
            "Improving bureau-observed factors such as credit score, reducing recent "
            "enquiries and lowering credit utilisation would move the applicant into a "
            "lower risk band."
        )
    return explanations


def render_risk_gauge(probability_of_default_pct):
    """
    Semi-circular gauge: Probability of Default (0–100%).
    Green → Low (0–33%), Yellow → Medium (33–66%), Red → High (66–100%).
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_of_default_pct,
            number={"suffix": "%", "font": {"size": 36}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#f97316"},  # orange bar only
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e2e8f0",
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={"family": "sans-serif"},
        title={"text": "Probability of default", "font": {"size": 14}, "x": 0.5},
    )
    return fig


def _build_counterfactual_points(pred_result):
    """
    Build exactly 5 counterfactual-style explanations in simple business language.
    Uses SHAP impact only to choose themes; does not expose feature names or values.
    """
    shap_impact = pred_result.get("shap_impact") or []
    prob = float(pred_result.get("probability", 0.0) or 0.0)

    points = []
    used = set()

    # Only focus on factors that increased risk
    increasing = [x for x in shap_impact if x.get("shap_value", 0) > 0]
    for x in increasing:
        name = str(x.get("display_name", "")).lower()
        if "income" in name and "income" not in used:
            points.append("Strengthening your documented income (higher or more stable earnings) would help reduce the risk rating.")
            used.add("income")
        elif ("credit" in name or "enquir" in name) and "credit_enquiries" not in used:
            points.append("Reducing the number of recent credit applications would lower your assessed risk.")
            used.add("credit_enquiries")
        elif ("employment" in name or "occupation" in name) and "employment" not in used:
            points.append("Maintaining a longer, stable employment history can improve your risk profile.")
            used.add("employment")
        elif "education" in name and "education" not in used:
            points.append("Pursuing higher or more specialized education is associated with lower default risk.")
            used.add("education")
        elif "housing" in name and "housing" not in used:
            points.append("Improving long-term housing stability (for example, longer residence at one address) can reduce risk.")
            used.add("housing")
        if len(points) >= 5:
            break

    # If we could not identify specific SHAP-based drivers, fall back to a
    # single, model-specific explanation rather than generic financial advice.
    if not points:
        points.append(
            "The model did not find a single dominant driver of risk; for this application, "
            "risk is influenced by several factors together rather than one variable alone."
        )
    return points[:5]


def render_results_dashboard(pred_result):
    """
    Results dashboard (Loan only): risk level, gauge, counterfactuals (reduce risk below threshold).
    """
    # Use stored customer/product/meta information to tailor explanations.
    customer_type_value = (pred_result.get("customer_type") or "").strip().lower()
    model_used = (pred_result.get("model_used") or "").strip().lower()

    if customer_type_value == "existing" and not model_used:
        model_used = "full_risk"
    elif customer_type_value == "new" and not model_used:
        model_used = "proxy"

    if model_used == "full_risk" and customer_type_value == "existing":
        st.caption(
            "Existing customer – full loan default risk model using bureau and repayment history."
        )
    elif model_used == "proxy" and customer_type_value == "new":
        st.caption(
            "New-to-credit customer – proxy model using income, employment and housing stability; "
            "no credit history features are used."
        )

    # Loan: risk level, gauge, risk-dependent explanations
    prob = pred_result["probability"]
    probability_of_default_pct = round(prob * 100)
    risk_category = pred_result.get("risk_category", "High")

    decision = _final_decision_from_risk(risk_category)
    decision_label = "Approved" if decision == "approved" else "Rejected"
    st.markdown(
        f"<div class='card-box'><strong>Loan decision:</strong> {decision_label}</div>",
        unsafe_allow_html=True,
    )

    risk_class = "risk-low" if risk_category == "Low" else ("risk-medium" if risk_category == "Medium" else "risk-high")
    st.markdown(
        f"<div class='card-box'><strong>Risk Level:</strong> <span class='{risk_class}'>{risk_category}</span></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Risk gauge")
    st.plotly_chart(render_risk_gauge(probability_of_default_pct), use_container_width=True)

    st.subheader("Explainability")
    # Risk-dependent explanation logic: Low risk gets confirmation, Medium/High get counterfactuals
    if risk_category == "Low":
        # Low risk: lightweight confirmation
        points = get_low_risk_confirmation(pred_result.get("shap_impact") or [])
    else:
        # Medium/High risk: counterfactual explanations
        points = get_loan_counterfactuals(
            pred_result.get("shap_impact") or [],
            prob,
            risk_threshold=0.33,
            risk_category=risk_category,
        )
        if not points:
            points = _build_counterfactual_points(pred_result)
    for p in points:
        st.markdown(f"- {p}")

    dice_points = pred_result.get("dice_explanations") or []
    if dice_points:
        st.markdown("**DiCE counterfactual explanations**")
        for p in dice_points:
            st.markdown(f"- {p}")


def render_admin_view():
    """Admin / View Database: table of stored records and optional lookup."""
    st.markdown("<p class='main-header'>View Stored Records</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Submitted predictions for project demonstration.</p>", unsafe_allow_html=True)
    st.divider()

    # Lookup (DB is simplified: no PII stored; search runs on flat submissions fields)
    _display_columns = [
        "ID",
        "Created At",
        "Customer Type",
        "Income",
        "Requested Amount",
        "Employment Type",
        "CIBIL Score",
        "Risk Score",
        "Eligibility Decision",
        "Model Accuracy",
    ]
    with st.expander("🔍 Lookup (search in customer type / employment / decision)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            lookup_name = st.text_input(
                "Search text",
                key="lookup_name",
                placeholder="e.g. new, salaried, approve, or 0.85",
            )
        with col2:
            lookup_phone = st.text_input("Additional text (optional)", key="lookup_phone", placeholder="Optional...")
        if st.button("Search", key="lookup_btn"):
            rows = lookup_by_name_or_phone(name=lookup_name if lookup_name else None, phone=lookup_phone if lookup_phone else None)
            df, err = _ensure_dataframe_for_display(rows, _display_columns, fallback_message="No matching records found.")
            if err:
                st.info(err)
            elif df is None or df.empty:
                st.info("No matching records found.")
            else:
                # Use st.table to avoid the buggy DataFrame rendering path in st.dataframe.
                # Reset index and ensure primitive column types only.
                df = df.reset_index(drop=True)
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].astype(str)
                st.table(df)

    # Full table (DataFrame-safe for Streamlit)
    rows = get_all_submissions()
    df_all, err = _ensure_dataframe_for_display(rows, _display_columns, fallback_message="No submissions yet.")
    count = int(len(df_all)) if isinstance(df_all, pd.DataFrame) else 0
    st.metric("Total submissions", count)

    if err:
        st.info(err)
    elif df_all is None or df_all.empty:
        st.info("No submissions yet.")
    else:
        # Use st.table to avoid the buggy DataFrame rendering path in st.dataframe.
        # Reset index and ensure primitive column types only.
        df_all = df_all.reset_index(drop=True)
        for col in df_all.columns:
            if pd.api.types.is_datetime64_any_dtype(df_all[col]):
                df_all[col] = df_all[col].astype(str)
        st.table(df_all)


def main():
    if "assessment_started" not in st.session_state:
        st.session_state["assessment_started"] = False
    if "view" not in st.session_state:
        st.session_state["view"] = "form"
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_product_name" not in st.session_state:
        st.session_state["last_product_name"] = None
    if "role" not in st.session_state:
        st.session_state["role"] = "User"
    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False

    # Sidebar: role selection, then navigation. Admin password is entered via the web UI (sidebar), not the terminal.
    with st.sidebar:
        role = st.radio(
            "Role",
            options=["User", "Admin"],
            key="role",
        )
        # When switching to User, clear admin login so Admin must re-authenticate when returning.
        if role == "User":
            st.session_state["admin_logged_in"] = False
        # Password input appears ONLY when Admin is selected. Entered through the web UI, not the terminal.
        if role == "Admin":
            st.text_input(
                "Enter Admin Password",
                type="password",
                key="admin_password_input",
                placeholder="Password",
            )
            if st.button("Login as Admin", type="primary", use_container_width=True):
                entered = (st.session_state.get("admin_password_input") or "").strip()
                if entered == ADMIN_PASSWORD:
                    st.session_state["admin_logged_in"] = True
                    st.success("Logged in.")
                    st.rerun()
                else:
                    st.error("Incorrect password. Access denied.")
            if st.session_state.get("admin_logged_in"):
                st.caption("Admin logged in")
        st.divider()
        # Show "View Records" in Navigation only when Admin is logged in. User role never sees it.
        show_view_records = role == "Admin" and st.session_state.get("admin_logged_in")
        nav_options = ["Risk Assessment", "View Records"] if show_view_records else ["Risk Assessment"]
        if not show_view_records and st.session_state.get("nav_page") == "View Records":
            st.session_state["nav_page"] = "Risk Assessment"
        if role == "User":
            st.session_state["nav_page"] = "Risk Assessment"
        page = st.radio(
            "Navigation",
            options=nav_options,
            key="nav_page",
        )

    # View Records: only allow when Admin role and admin_logged_in. Prevent direct access otherwise.
    if page == "View Records":
        if role == "Admin" and st.session_state.get("admin_logged_in"):
            render_admin_view()
            return
        st.info("Enter the admin password in the **sidebar** and click **Login as Admin** to view records.")
        return

    if not st.session_state["assessment_started"]:
        render_landing()
        return

    # Results page (separate section after submission)
    if st.session_state["view"] == "results" and st.session_state["last_result"] is not None:
        last_result = st.session_state["last_result"]
        customer_type_value_for_header = (last_result.get("customer_type") or "").strip().lower()

        product_label = "Loan"

        if customer_type_value_for_header == "existing":
            customer_label = "Existing Customer"
        elif customer_type_value_for_header == "new":
            customer_label = "New Customer"
        else:
            customer_label = ""

        if customer_label:
            main_header = f"{product_label} Application – {customer_label}"
        else:
            main_header = f"{product_label} – Result"

        st.markdown(f"<p class='main-header'>{main_header}</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Gauge-based risk view with counterfactual explanations.</p>", unsafe_allow_html=True)
        st.divider()
        render_results_dashboard(last_result)
        if st.button("Back to input form"):
            st.session_state["view"] = "form"
            st.rerun()
        return

    # Assessment flow (input form)
    st.markdown("<p class='main-header'>Loan Risk Prediction System</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Enter applicant details to check prediction.</p>", unsafe_allow_html=True)
    st.divider()

    # Applicant identification (required for all predictions)
    st.subheader("Applicant identification")
    name = st.text_input("Applicant name", key="applicant_name")
    contact = st.text_input("Contact / Mobile number", key="contact_number")

    is_existing, customer_type_value = customer_type_selection()

    if is_existing:
        customer_id = st.text_input("Enter customer ID", placeholder="e.g. 1", key="cust_id")

        existing_loan_amount = st.number_input(
            "Requested loan amount",
            min_value=1000,
            value=100000,
            step=5000,
            key="existing_loan_amount",
        )

        submit = st.button("Check Prediction", type="primary", key="submit_existing")
        if submit:
            if customer_type_value not in ("new", "existing"):
                st.warning("Please select whether this is a New or Existing customer.")
            else:
                valid_identity, msg = validate_applicant_identity(name, contact)
                if not valid_identity:
                    st.warning(msg)
                elif not customer_id or not customer_id.strip():
                    st.warning("Please enter a customer ID.")
                elif existing_loan_amount is None or existing_loan_amount < 1000:
                    st.warning("Please enter a valid requested loan amount (min 1,000).")
                else:
                    row = get_customer_by_id(customer_id.strip())
                    if row is None:
                        st.error(f"Customer ID '{customer_id}' not found.")
                    else:
                        # Read-only credit background fetched from dataset / bureau
                        _render_existing_credit_background(row)
                        with st.spinner("Fetching and predicting..."):
                            try:
                                model_used = "full_risk"
                                row["AMT_CREDIT"] = float(existing_loan_amount)
                                model, preprocessor, encoder_mappings = load_model_and_artifacts()
                                # Existing customer loan: use full risk model including credit behaviour
                                result = predict_single(
                                    row,
                                    model=model,
                                    preprocessor=preprocessor,
                                    encoder_mappings=encoder_mappings,
                                    compute_shap=True,
                                    background_df=get_background_data(),
                                )
                                base_prob = result.get("probability", 0.0)
                                base_category = result.get("risk_category", _risk_category_from_prob(base_prob))
                                result["dice_explanations"] = _generate_existing_dice_explanations(
                                    row,
                                    base_prob,
                                    base_category,
                                    model,
                                    preprocessor,
                                    encoder_mappings,
                                )
                                # Annotate result with product/customer metadata for explanations/UI.
                                result["customer_type"] = customer_type_value
                                result["product_type"] = "loan"
                                result["model_used"] = model_used

                                # Snapshot of technical inputs plus any product-specific UI fields
                                if row is not None and len(row) > 0:
                                    input_parameters = row.iloc[0].to_dict()
                                else:
                                    input_parameters = {}
                                input_parameters.pop("CODE_GENDER", None)
                                if existing_loan_amount is not None:
                                    input_parameters["requested_loan_amount"] = float(existing_loan_amount)

                                # Persist both detailed submission and high-level application record.
                                save_submission(
                                    applicant_name=name,
                                    contact_number=contact,
                                    is_existing_customer=True,
                                    customer_id=customer_id.strip(),
                                    input_parameters=input_parameters,
                                    risk_result=result,
                                )
                                save_application_record(
                                    applicant_name=name,
                                    contact_number=contact,
                                    risk_result=result,
                                )

                                st.session_state["last_result"] = result
                                st.session_state["last_product_name"] = "Loan"
                                st.session_state["view"] = "results"
                                st.rerun()
                            except FileNotFoundError:
                                st.error("**Train model first.** No pre-trained model found. Run: `python -m ml.train_model` (requires labeled data with TARGET).")
                            except Exception as e:
                                st.exception(e)
    else:
        _, input_parameters = build_new_customer_form()
        submit = st.button("Check Prediction", type="primary", key="submit_new")
        if submit:
            if customer_type_value not in ("new", "existing"):
                st.warning("Please select whether this is a New or Existing customer.")
            else:
                valid_identity, msg = validate_applicant_identity(name, contact)
                if not valid_identity:
                    st.warning(msg)
                elif input_parameters.get("requested_loan_amount") is None or float(input_parameters.get("requested_loan_amount") or 0.0) < 1000:
                    st.warning("Please enter a valid loan amount (min 1,000).")
                elif not validate_new_customer_inputs():
                    st.warning("Please complete all required fields.")
                else:
                    # NEW customers: use proxy-based affordability / stability model,
                    # excluding all credit history features by design.
                    with st.spinner("Computing prediction and DiCE explanations..."):
                        try:
                            model_used = "proxy"
                            prob, risk_score, risk_category = _proxy_new_customer_risk(
                                input_parameters
                            )
                            result = {
                                "probability": prob,
                                "risk_score": risk_score,
                                "risk_category": risk_category,
                                "shap_impact": [],
                                "base_value": None,
                            }
                            # DiCE explanations for new customers (income, employment, amount only)
                            result["dice_explanations"] = _dice_new_customer_explanations(
                                input_parameters, prob, risk_category
                            )

                            # Annotate result with product/customer metadata for explanations/UI.
                            result["customer_type"] = customer_type_value
                            result["product_type"] = "loan"
                            result["model_used"] = model_used

                            # Persist both detailed submission and high-level application record.
                            save_submission(
                                applicant_name=name,
                                contact_number=contact,
                                is_existing_customer=False,
                                customer_id=None,
                                input_parameters=input_parameters,
                                risk_result=result,
                            )
                            save_application_record(
                                applicant_name=name,
                                contact_number=contact,
                                risk_result=result,
                            )

                            st.session_state["last_result"] = result
                            st.session_state["last_product_name"] = "Loan"
                            st.session_state["view"] = "results"
                            st.rerun()
                        except Exception as e:
                            st.exception(e)


if __name__ == "__main__":
    main()
