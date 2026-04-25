"""
Explainability module: SHAP feature contributions and optional DiCE counterfactuals.
Converts technical feature names to business-friendly explanations.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_DISPLAY_NAMES, FEATURE_COLUMNS


def get_display_name(feature_name):
    """Map technical feature name to business-friendly label."""
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)


def compute_shap_values(model, X_background, X_explain, feature_names=None):
    """
    Compute SHAP values for the given instance(s).
    Returns base_value, shap_values, and optional feature names.
    """
    try:
        import shap
    except ImportError:
        return None, None, None
    
    if feature_names is None:
        feature_names = list(FEATURE_COLUMNS) if hasattr(FEATURE_COLUMNS, "__iter__") and not isinstance(FEATURE_COLUMNS, str) else []
    
    X_background = np.asarray(X_background)
    X_explain = np.asarray(X_explain)
    if len(X_background) > 100:
        X_background = X_background[:100]
    if len(X_explain.shape) == 1:
        X_explain = X_explain.reshape(1, -1)
    
    explainer = None
    try:
        if _is_tree_model(model):
            explainer = shap.TreeExplainer(model, X_background)
        else:
            # LinearExplainer for LogisticRegression
            explainer = shap.LinearExplainer(model, X_background)
    except Exception:
        pass
    if explainer is None:
        try:
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
        except Exception:
            return None, None, None
    
    shap_values = explainer.shap_values(X_explain)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
    else:
        base_value = float(base_value)
    return base_value, shap_values, feature_names


def _is_tree_model(model):
    name = type(model).__name__
    return "Tree" in name or "Forest" in name or "XGB" in name or "Gradient" in name


def shap_to_feature_impact(shap_values_one, feature_names, base_value, prediction_prob):
    """
    Convert SHAP output to a list of dicts with display names and direction.
    shap_values_one: 1D array of SHAP values for one instance.
    """
    if shap_values_one is None or feature_names is None:
        return []
    feature_names = list(feature_names)[: len(shap_values_one)]
    impact = []
    for i, (name, val) in enumerate(zip(feature_names, shap_values_one)):
        impact.append({
            "feature": name,
            "display_name": get_display_name(name),
            "shap_value": float(val),
            "direction": "increases risk" if val > 0 else "decreases risk",
        })
    impact.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return impact


def shap_to_plain_language(shap_impact, top_n=10):
    """
    Convert SHAP feature impact into business-friendly plain-language sentences.
    Returns list of strings, e.g. "High credit utilization increased your risk."
    """
    if not shap_impact:
        return []
    sentences = []
    for x in shap_impact[:top_n]:
        name = x["display_name"]
        direction = x["direction"]
        if direction == "increases risk":
            if "Income" in name:
                sentences.append("Lower or less stable income increased your risk.")
            elif "Credit" in name and "Enquiries" in name:
                sentences.append("More credit enquiries in the past year increased your risk.")
            elif "Credit Score" in name:
                sentences.append("Lower credit score increased your risk.")
            elif "Employment" in name or "Occupation" in name:
                sentences.append("Shorter employment or certain occupation types increased your risk.")
            elif "Education" in name:
                sentences.append("Lower education level increased your risk.")
            elif "Children" in name:
                sentences.append("Having more dependants increased your risk.")
            elif "Region" in name:
                sentences.append("Region rating contributed to higher risk.")
            elif "Housing" in name:
                sentences.append("Housing type contributed to higher risk.")
            else:
                sentences.append(f"**{name}** increased your risk.")
        else:
            if "Income" in name:
                sentences.append("Higher or more stable income reduced your risk.")
            elif "Credit" in name and "Enquiries" in name:
                sentences.append("Fewer credit enquiries reduced your risk.")
            elif "Credit Score" in name:
                sentences.append("Higher credit score reduced your risk.")
            elif "Employment" in name or "Occupation" in name:
                sentences.append("Longer employment or stable occupation reduced your risk.")
            elif "Education" in name:
                sentences.append("Higher education level reduced your risk.")
            elif "Children" in name:
                sentences.append("Fewer dependants reduced your risk.")
            elif "Region" in name:
                sentences.append("Favorable region rating reduced your risk.")
            elif "Housing" in name:
                sentences.append("Housing type contributed to lower risk.")
            else:
                sentences.append(f"**{name}** reduced your risk.")
    return sentences


def get_low_risk_confirmation(shap_impact):
    """
    Generate a lightweight confirmation message for Low risk outcomes.
    Simple, customer-friendly, and acceptable.
    """
    return ["You are classified as a low-risk customer based on your strong credit profile and affordable repayment capacity."]


def get_counterfactual_suggestion(shap_impact, current_prob, target_lower=True):
    """
    Generate a simple counterfactual suggestion from top positive SHAP features.
    Suggests minimal changes to reduce risk (e.g. improve income, reduce enquiries).
    """
    if not shap_impact or current_prob < 0.5:
        return None
    # Top factors increasing risk
    increasing = [x for x in shap_impact if x["shap_value"] > 0][:3]
    if not increasing:
        return None
    suggestions = []
    for x in increasing:
        name = x["display_name"]
        if "Income" in name:
            suggestions.append("Consider showing higher or more stable annual income.")
        elif "Credit" in name or "Enquiries" in name:
            suggestions.append("Reduce new credit applications; fewer enquiries can lower risk.")
        elif "Employment" in name or "Occupation" in name:
            suggestions.append("Longer employment history or a more stable occupation type may help.")
        elif "Education" in name:
            suggestions.append("Higher education level is associated with lower risk.")
        elif "Children" in name:
            suggestions.append("Fewer dependants can slightly reduce perceived risk.")
        elif "Region" in name:
            suggestions.append("Region rating affects risk; urban regions may be favored.")
        else:
            suggestions.append(f"Improving '{name}' could reduce default risk.")
    return " ".join(suggestions[:2]) if suggestions else None


def get_loan_counterfactuals(shap_impact, current_prob, risk_threshold=0.33, risk_category=None):
    """
    Model-agnostic, user-readable counterfactuals for loan risk.
    Answers: "What minimal change would reduce the risk below the threshold?"
    Focus: loan amount, income, other high-impact features.
    
    For Low risk cases, returns empty list (counterfactuals should not be generated).
    For Medium/High risk cases, generates actionable counterfactual explanations.
    """
    # Do not generate counterfactuals for Low risk cases
    if risk_category == "Low":
        return []
    
    if not shap_impact:
        return []
    points = []
    used = set()
    increasing = [x for x in shap_impact if x.get("shap_value", 0) > 0]

    for x in increasing:
        name = str(x.get("display_name", "")).lower()
        if "loan amount" in name and "loan_amount" not in used:
            points.append(
                "Reducing the requested loan amount would lower your default risk; a smaller loan "
                "keeps risk below the threshold for most profiles."
            )
            used.add("loan_amount")
        elif "income" in name and "income" not in used:
            points.append(
                "Increasing your documented annual income (or showing more stable earnings) would "
                "help bring your risk below the threshold."
            )
            used.add("income")
        elif ("credit" in name or "enquir" in name) and "credit_enquiries" not in used:
            points.append(
                "Reducing recent credit applications would lower your assessed risk and help you "
                "stay below the risk threshold."
            )
            used.add("credit_enquiries")
        elif ("employment" in name or "occupation" in name) and "employment" not in used:
            points.append(
                "A longer, more stable employment history would improve your risk profile and "
                "could bring risk below the threshold."
            )
            used.add("employment")
        elif "education" in name and "education" not in used:
            points.append(
                "Higher or more specialized education is associated with lower default risk and "
                "could help you meet the threshold."
            )
            used.add("education")
        elif "housing" in name and "housing" not in used:
            points.append(
                "Stronger housing stability (e.g. longer residence at one address) can reduce "
                "perceived risk and help you stay below the threshold."
            )
            used.add("housing")
        if len(points) >= 5:
            break

    fallback = [
        "Keeping all repayments on time will steadily improve your risk rating.",
        "Avoiding unnecessary short-term borrowing can improve your risk over time.",
        "Building a larger savings buffer relative to monthly commitments strengthens your profile.",
        "Limiting new credit applications can reduce perceived risk.",
        "Updating your financial records with the lender helps present an accurate, lower-risk picture.",
    ]
    for p in fallback:
        if len(points) >= 5:
            break
        if p not in points:
            points.append(p)
    return points[:5]


def dice_counterfactual(model, X_train_df, instance_df, target_class=0, num_cf=2):
    """
    Optional: Use DiCE to generate counterfactual examples.
    Returns list of counterfactual rows (DataFrames) or None if DiCE not used.
    """
    try:
        import dice_ml
        from dice_ml import Dice
    except ImportError:
        return None
    
    try:
        d = dice_ml.Data(dataframe=X_train_df, continuous_features=[], outcome_name="target")
        # We need a wrapper that uses our model; DiCE expects dataframe with outcome
        # For simplicity we skip full DiCE integration if data interface is complex
        return None
    except Exception:
        return None
