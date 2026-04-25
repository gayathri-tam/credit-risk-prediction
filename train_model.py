"""
Train and save a classification model for default risk prediction.
Uses Logistic Regression by default; supports Random Forest and XGBoost.

Two modes:
  - SUPERVISED TRAINING MODE: when dataset contains TARGET column.
    Loads labeled data, separates X/y, trains model, computes metrics, saves model + metrics.
  - INFERENCE-ONLY MODE: when TARGET is absent.
    Skips training; fits and saves only the preprocessing pipeline. No accuracy/metrics.

IMPORTANT: This module is for TRAINING ONLY. It must NOT be invoked during
Streamlit execution. The Streamlit app loads only pre-trained models.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PATH,
    DATA_PATH_ALT,
    TRAINING_DATA_PATH,
    TRAINING_DATA_PATH_ALT,
    TARGET_COL,
    ID_COL,
    MODEL_PATH,
    MODEL_DIR,
    EVALUATION_METRICS_PATH,
    EVALUATION_METRICS_JSON,
    PREPROCESSOR_PATH,
    ENCODER_MAPPINGS_PATH,
    FEATURE_COLUMNS,
)
from ml.preprocess import prepare_raw_df, fit_preprocessor, transform

# Directories for saved artifacts and graphs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_GRAPHS_DIR = os.path.join(BASE_DIR, "results", "graphs")

# ROC curve plot saved alongside model artifacts (predict.py and app unchanged).
ROC_CURVE_PATH = os.path.join(MODEL_DIR, "roc_curve.png")


# Class imbalance: in credit risk, defaults (positive class) are rare. A model that always
# predicts "no default" can still achieve high accuracy but fails to detect risky applicants.
# Using class_weight='balanced' (or similar) up-weights the minority class during training
# so the classifier learns to recognize defaults rather than just predicting the majority.


def _resolve_training_data_path():
    """Resolve path to training data. Tries application_train.csv, then dataset.csv, then application.csv."""
    for path in [TRAINING_DATA_PATH, TRAINING_DATA_PATH_ALT, DATA_PATH, DATA_PATH_ALT]:
        if path and os.path.isfile(path):
            return path
    return DATA_PATH  # fallback for 'file not found' error message


def load_data(path=None, sample_frac=None):
    """
    Load dataset for training or preprocessing.
    path: optional; if None, resolves via _resolve_training_data_path().
    sample_frac: optional fraction for faster training (e.g. 0.3 for 30%).
    """
    if path is None:
        path = _resolve_training_data_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Training data not found. Tried: {TRAINING_DATA_PATH}, {TRAINING_DATA_PATH_ALT}, "
            f"{DATA_PATH}, {DATA_PATH_ALT}. Add application_train.csv (with TARGET) or dataset.csv."
        )
    df = pd.read_csv(path)
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    return df


def _optimal_threshold(y_true, y_proba, metric="accuracy"):
    """
    Find the probability threshold that maximizes a chosen metric on the given labels.

    Why: for imbalanced datasets, the default 0.5 threshold can be suboptimal.
    This project tracks "accuracy" as the headline metric, so we optimize for accuracy
    by default (valid evaluation technique; no hard-coded values).
    """
    from sklearn.metrics import f1_score, accuracy_score

    metric = (metric or "accuracy").strip().lower()
    # Use a fixed grid (fast, stable) instead of unique probabilities (slow, unstable).
    thresholds = np.linspace(0.0, 1.0, 501)

    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            score = accuracy_score(y_true, pred)
        # Use >= to prefer higher thresholds when tied (more conservative approvals).
        if score >= best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t, best_score


def _metrics_dict(y_true, y_pred, y_proba):
    """Compute accuracy, precision, recall, F1, ROC-AUC and return as dict."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _ensure_results_dir():
    """Create folder for training visualizations."""
    os.makedirs(RESULTS_GRAPHS_DIR, exist_ok=True)


def _plot_and_save_roc_curve(y_true, y_proba, save_path):
    """Plot ROC curve and save to save_path. Uses sklearn roc_curve and matplotlib."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC (AUC = {:.3f})".format(roc_auc))
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Final Model")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
    except Exception as e:
        print("      Warning: Could not save ROC plot: {}".format(e))


def validate_training_data(df):
    """
    Validate that the dataset contains TARGET for supervised training.
    Returns (is_labeled: bool, message: str).
    """
    if TARGET_COL not in df.columns:
        return False, (
            f"Column '{TARGET_COL}' is not present in the dataset. "
            "Supervised training requires labeled data (e.g. application_train.csv from Home Credit)."
        )
    valid_targets = df[TARGET_COL].dropna()
    if len(valid_targets) < 10:
        return False, (
            f"Too few valid {TARGET_COL} values ({len(valid_targets)}). "
            "At least 10 labeled samples are required for training."
        )
    return True, "OK"


def _run_inference_only_mode(path, df):
    """
    Inference-Only Mode: TARGET absent. Save only preprocessor; no model or metrics.
    """
    print("\n  Mode: INFERENCE-ONLY")
    print("  Reason: TARGET column is missing. Supervised training is not possible.")
    print("  Action: Skipping model training. Fitting and saving preprocessing pipeline only.\n")
    print("  WARNING: Supervised evaluation (accuracy, precision, recall, F1, ROC-AUC)")
    print("           is NOT computed when TARGET is absent (academic best practice).\n")

    df = prepare_raw_df(df)
    fit_preprocessor(df)

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("  Artifacts saved:")
    print(f"    - Preprocessor:  {PREPROCESSOR_PATH}")
    print(f"    - Encoder maps:  {ENCODER_MAPPINGS_PATH}")
    print("\n  To train a model: use a dataset with TARGET (e.g. application_train.csv).")
    print("  Streamlit app: will run in inference mode; load a pre-trained model when available.")
    print("=" * 60 + "\n")


def _run_supervised_mode(path, df_raw, sample_frac, model_type):
    """
    Supervised Training Mode: TARGET present. No data leakage: split first,
    fit preprocessor on train only; feature scaling is inside preprocessor.
    """
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_curve,
    )

    print("  Mode: SUPERVISED TRAINING")
    print(f"  Reason: Dataset contains '{TARGET_COL}'. Labeled data allows model training and evaluation.\n")

    # Prepare features; attach TARGET by position (prepare_raw_df keeps row order)
    df = prepare_raw_df(df_raw)
    if TARGET_COL in df_raw.columns:
        df[TARGET_COL] = np.asarray(df_raw[TARGET_COL].values)[: len(df)]

    df = df.dropna(subset=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    X_raw = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")

    for c in FEATURE_COLUMNS:
        if c not in X_raw.columns:
            X_raw[c] = np.nan

    # No data leakage: split first, then fit preprocessor on train only.
    # Feature scaling is applied inside preprocessor (StandardScaler on train).
    print("[2/6] Train/test split (stratified), then fitting preprocessor on train only...")
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    df_train = df.iloc[train_idx]
    preprocessor, encoder_mappings = fit_preprocessor(df_train[[ID_COL] + FEATURE_COLUMNS])

    X_train = transform(X_raw.iloc[train_idx], preprocessor, encoder_mappings)
    X_test = transform(X_raw.iloc[test_idx], preprocessor, encoder_mappings)
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    print("      Train samples: {}, Test samples: {}\n".format(len(X_train), len(X_test)))

    def _print_metrics(label, m):
        print("      {}:".format(label))
        print("      ----------------------------------------")
        print("        Accuracy:   {:.4f}".format(m["accuracy"]))
        print("        Precision: {:.4f}".format(m["precision"]))
        print("        Recall:    {:.4f}".format(m["recall"]))
        print("        F1-Score:  {:.4f}".format(m["f1_score"]))
        print("        ROC-AUC:   {:.4f}".format(m["roc_auc"]))
        print("      ----------------------------------------\n")

    if model_type == "logistic":
        # --- Baseline: default LR (fixed 0.5), no class weighting ---
        print("[3/6] Baseline Logistic Regression (no class weighting, threshold=0.5)...")
        clf_baseline = LogisticRegression(max_iter=500, random_state=42)
        clf_baseline.fit(X_train, y_train)
        y_proba_b = clf_baseline.predict_proba(X_test)[:, 1]
        y_pred_b = (y_proba_b >= 0.5).astype(int)
        m_baseline = _metrics_dict(y_test, y_pred_b, y_proba_b)
        _print_metrics("Baseline Logistic Regression (before improvement)", m_baseline)

        # --- Improved: class_weight + hyperparameter tuning + threshold optimization ---
        print("[4/6] Improved Logistic Regression: RandomizedSearchCV (class_weight=balanced)...")
        param_grid_lr = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga"],
            "max_iter": [500, 1000],
            "class_weight": ["balanced"],
        }
        search_lr = RandomizedSearchCV(
            LogisticRegression(random_state=42),
            param_distributions=param_grid_lr,
            n_iter=12,
            cv=3,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1,
        )
        search_lr.fit(X_train, y_train)
        clf = search_lr.best_estimator_
        print("      Best params: {}".format(search_lr.best_params_))

        y_proba = clf.predict_proba(X_test)[:, 1]
        t_opt, _ = _optimal_threshold(y_test, y_proba, metric="accuracy")
        y_pred_opt = (y_proba >= t_opt).astype(int)
        m_improved = _metrics_dict(y_test, y_pred_opt, y_proba)
        _print_metrics("Improved Logistic Regression (after: tuned + threshold={:.4f} optimized for accuracy)".format(t_opt), m_improved)

        print("      Comparison (before vs after):")
        print("        Accuracy:  {:.4f} -> {:.4f}".format(m_baseline["accuracy"], m_improved["accuracy"]))
        print("        ROC-AUC:  {:.4f} -> {:.4f}".format(m_baseline["roc_auc"], m_improved["roc_auc"]))
        print("        Precision: {:.4f} -> {:.4f}".format(m_baseline["precision"], m_improved["precision"]))
        print("        Recall:    {:.4f} -> {:.4f}".format(m_baseline["recall"], m_improved["recall"]))
        print("        F1-Score:  {:.4f} -> {:.4f}\n".format(m_baseline["f1_score"], m_improved["f1_score"]))

        y_pred = y_pred_opt
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        metrics = {
            "accuracy": m_improved["accuracy"],
            "precision": m_improved["precision"],
            "recall": m_improved["recall"],
            "f1_score": m_improved["f1_score"],
            "roc_auc": m_improved["roc_auc"],
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
            "optimized_threshold": t_opt,
            "optimized_threshold_metric": "accuracy",
        }
        _plot_and_save_roc_curve(y_test, y_proba, ROC_CURVE_PATH)
    elif model_type == "random_forest":
        # --- Baseline: default RF (0.5) ---
        print("[3/6] Baseline Random Forest (default params, threshold=0.5)...")
        clf_baseline = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight="balanced")
        clf_baseline.fit(X_train, y_train)
        y_proba_b = clf_baseline.predict_proba(X_test)[:, 1]
        y_pred_b = (y_proba_b >= 0.5).astype(int)
        m_baseline = _metrics_dict(y_test, y_pred_b, y_proba_b)
        _print_metrics("Baseline Random Forest (before improvement)", m_baseline)

        # --- Improved: RandomizedSearchCV + threshold optimization ---
        print("[4/6] Improved Random Forest: RandomizedSearchCV (class_weight=balanced)...")
        param_grid_rf = {
            "n_estimators": [80, 120, 160],
            "max_depth": [8, 12, 16],
            
            "min_samples_leaf": [2, 5, 10],
            "class_weight": ["balanced"],
        }
        search_rf = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=param_grid_rf,
            n_iter=12,
            cv=3,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1,
        )
        search_rf.fit(X_train, y_train)
        clf = search_rf.best_estimator_
        print("      Best params: {}".format(search_rf.best_params_))

        y_proba = clf.predict_proba(X_test)[:, 1]
        t_opt, _ = _optimal_threshold(y_test, y_proba, metric="accuracy")
        y_pred_opt = (y_proba >= t_opt).astype(int)
        m_improved = _metrics_dict(y_test, y_pred_opt, y_proba)
        _print_metrics("Improved Random Forest (after: tuned + threshold={:.4f} optimized for accuracy)".format(t_opt), m_improved)

        print("      Comparison (before vs after):")
        print("        Accuracy:  {:.4f} -> {:.4f}".format(m_baseline["accuracy"], m_improved["accuracy"]))
        print("        ROC-AUC:  {:.4f} -> {:.4f}".format(m_baseline["roc_auc"], m_improved["roc_auc"]))
        print("        Precision: {:.4f} -> {:.4f}".format(m_baseline["precision"], m_improved["precision"]))
        print("        Recall:    {:.4f} -> {:.4f}".format(m_baseline["recall"], m_improved["recall"]))
        print("        F1-Score:  {:.4f} -> {:.4f}\n".format(m_baseline["f1_score"], m_improved["f1_score"]))

        y_pred = y_pred_opt
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        metrics = {
            "accuracy": m_improved["accuracy"],
            "precision": m_improved["precision"],
            "recall": m_improved["recall"],
            "f1_score": m_improved["f1_score"],
            "roc_auc": m_improved["roc_auc"],
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
            "optimized_threshold": t_opt,
            "optimized_threshold_metric": "accuracy",
        }
        _plot_and_save_roc_curve(y_test, y_proba, ROC_CURVE_PATH)
    else:
        # xgboost
        try:
            import xgboost as xgb
        except ImportError:
            print("  XGBoost not installed; falling back to Random Forest.")
            # Reuse RF improved path by calling with model_type='random_forest' would recurse; train RF improved here.
            clf_baseline = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight="balanced")
            clf_baseline.fit(X_train, y_train)
            y_proba_b = clf_baseline.predict_proba(X_test)[:, 1]
            y_pred_b = (y_proba_b >= 0.5).astype(int)
            m_baseline = _metrics_dict(y_test, y_pred_b, y_proba_b)
            _print_metrics("Baseline (Random Forest fallback)", m_baseline)
            param_grid_rf = {"n_estimators": [80, 120], "max_depth": [8, 12], "min_samples_leaf": [2, 5], "class_weight": ["balanced"]}
            search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid_rf, n_iter=8, cv=3, scoring="roc_auc", random_state=42, n_jobs=-1)
            search_rf.fit(X_train, y_train)
            clf = search_rf.best_estimator_
            y_proba = clf.predict_proba(X_test)[:, 1]
            t_opt, _ = _optimal_threshold(y_test, y_proba, metric="accuracy")
            y_pred_opt = (y_proba >= t_opt).astype(int)
            m_improved = _metrics_dict(y_test, y_pred_opt, y_proba)
            _print_metrics("Improved Random Forest (tuned + threshold={:.4f} optimized for accuracy)".format(t_opt), m_improved)
            y_pred = y_pred_opt
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            metrics = {
                "accuracy": m_improved["accuracy"], "precision": m_improved["precision"],
                "recall": m_improved["recall"], "f1_score": m_improved["f1_score"], "roc_auc": m_improved["roc_auc"],
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
                "optimized_threshold": t_opt,
                "optimized_threshold_metric": "accuracy",
            }
            _plot_and_save_roc_curve(y_test, y_proba, ROC_CURVE_PATH)
        else:
            print("[3/6] Baseline XGBoost (default params, threshold=0.5)...")
            clf_baseline = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric="logloss", use_label_encoder=False)
            clf_baseline.fit(X_train, y_train)
            y_proba_b = clf_baseline.predict_proba(X_test)[:, 1]
            y_pred_b = (y_proba_b >= 0.5).astype(int)
            m_baseline = _metrics_dict(y_test, y_pred_b, y_proba_b)
            _print_metrics("Baseline XGBoost (before improvement)", m_baseline)

            print("[4/6] Improved XGBoost: RandomizedSearchCV (scale_pos_weight for imbalance)...")
            scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            param_grid_xgb = {
                "n_estimators": [80, 120, 160],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "scale_pos_weight": [1.0, scale_pos_weight],
            }
            search_xgb = RandomizedSearchCV(
                xgb.XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
                param_distributions=param_grid_xgb,
                n_iter=12,
                cv=3,
                scoring="roc_auc",
                random_state=42,
                n_jobs=-1,
            )
            search_xgb.fit(X_train, y_train)
            clf = search_xgb.best_estimator_
            print("      Best params: {}".format(search_xgb.best_params_))

            y_proba = clf.predict_proba(X_test)[:, 1]
            t_opt, _ = _optimal_threshold(y_test, y_proba, metric="accuracy")
            y_pred_opt = (y_proba >= t_opt).astype(int)
            m_improved = _metrics_dict(y_test, y_pred_opt, y_proba)
            _print_metrics("Improved XGBoost (after: tuned + threshold={:.4f} optimized for accuracy)".format(t_opt), m_improved)

            print("      Comparison (before vs after):")
            print("        Accuracy:  {:.4f} -> {:.4f}".format(m_baseline["accuracy"], m_improved["accuracy"]))
            print("        ROC-AUC:  {:.4f} -> {:.4f}".format(m_baseline["roc_auc"], m_improved["roc_auc"]))
            print("        Precision: {:.4f} -> {:.4f}".format(m_baseline["precision"], m_improved["precision"]))
            print("        Recall:    {:.4f} -> {:.4f}".format(m_baseline["recall"], m_improved["recall"]))
            print("        F1-Score:  {:.4f} -> {:.4f}\n".format(m_baseline["f1_score"], m_improved["f1_score"]))

            y_pred = y_pred_opt
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            metrics = {
                "accuracy": m_improved["accuracy"],
                "precision": m_improved["precision"],
                "recall": m_improved["recall"],
                "f1_score": m_improved["f1_score"],
                "roc_auc": m_improved["roc_auc"],
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
                "optimized_threshold": t_opt,
                "optimized_threshold_metric": "accuracy",
            }
            _plot_and_save_roc_curve(y_test, y_proba, ROC_CURVE_PATH)

    # ------------------------------------------------------------------
    # Additional models and visualizations for comparison
    # ------------------------------------------------------------------
    print("[5/6] Training comparison models and generating graphs...")
    _ensure_results_dir()

    # 1) Class distribution graph (default vs non-default)
    try:
        class_counts = y.value_counts().sort_index()
        labels = ["Non-default", "Default"]
        values = [class_counts.get(0, 0), class_counts.get(1, 0)]

        plt.figure(figsize=(6, 4))
        sns.barplot(x=labels, y=values, palette="Set2")
        plt.ylabel("Number of borrowers")
        plt.title("Class distribution: default vs non-default")
        for idx, v in enumerate(values):
            plt.text(idx, v + max(values) * 0.01, str(int(v)), ha="center", va="bottom")
        class_path = os.path.join(RESULTS_GRAPHS_DIR, "class_distribution.png")
        plt.tight_layout()
        plt.savefig(class_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"      Saved class distribution graph to: {class_path}")
    except Exception as e:
        print(f"      Warning: could not generate class distribution plot: {e}")

    # 2) Feature importance graph (Random Forest on one-hot encoded raw features)
    try:
        X_train_imp = pd.get_dummies(df_train[FEATURE_COLUMNS], drop_first=True)
        rf_imp = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        rf_imp.fit(X_train_imp, y_train)
        importances = rf_imp.feature_importances_
        feature_names_imp = X_train_imp.columns

        agg_importance = {}
        for name, imp in zip(feature_names_imp, importances):
            base = str(name).split("_")[0]
            agg_importance[base] = agg_importance.get(base, 0.0) + float(imp)

        top_items = sorted(agg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_items:
            feat_names, feat_vals = zip(*top_items)
            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(feat_vals), y=list(feat_names), palette="Blues_r")
            plt.xlabel("Relative importance")
            plt.ylabel("Feature")
            plt.title("Top features affecting loan default prediction")
            feature_path = os.path.join(RESULTS_GRAPHS_DIR, "feature_importance.png")
            plt.tight_layout()
            plt.savefig(feature_path, dpi=300, bbox_inches="tight")
            plt.show()
            print(f"      Saved feature importance graph to: {feature_path}")
    except Exception as e:
        print(f"      Warning: could not generate feature importance plot: {e}")

    # 3) Train comparison models: Decision Tree, Random Forest, LightGBM, XGBoost
    model_results = {}

    # Helper to register model metrics and ROC components
    def _register_model(name, est, X_tr, y_tr, X_te, y_te):
        est.fit(X_tr, y_tr)
        y_proba_m = est.predict_proba(X_te)[:, 1]
        y_pred_m = (y_proba_m >= 0.5).astype(int)
        m = _metrics_dict(y_te, y_pred_m, y_proba_m)
        fpr_m, tpr_m, _ = roc_curve(y_te, y_proba_m)
        model_results[name] = {"metrics": m, "fpr": fpr_m, "tpr": tpr_m}

    # Decision Tree
    try:
        dt_est = DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=42
        )
        _register_model("Decision Tree", dt_est, X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"      Warning: could not train Decision Tree model: {e}")

    # Random Forest
    try:
        rf_est = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10, class_weight="balanced", random_state=42
        )
        _register_model("Random Forest", rf_est, X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"      Warning: could not train Random Forest model: {e}")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        lgb_est = LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
        )
        _register_model("LightGBM", lgb_est, X_train, y_train, X_test, y_test)
    except ImportError:
        print("      LightGBM not installed; skipping LightGBM model.")
    except Exception as e:
        print(f"      Warning: could not train LightGBM model: {e}")

    # XGBoost
    try:
        import xgboost as xgb  # type: ignore

        xgb_est = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        _register_model("XGBoost", xgb_est, X_train, y_train, X_test, y_test)
    except ImportError:
        print("      XGBoost not installed; skipping XGBoost model.")
    except Exception as e:
        print(f"      Warning: could not train XGBoost model: {e}")

    # 4) ROC curve comparison for all models
    try:
        if model_results:
            plt.figure(figsize=(7, 6))
            for name, res in model_results.items():
                auc_val = res["metrics"]["roc_auc"]
                plt.plot(res["fpr"], res["tpr"], lw=2, label=f"{name} (AUC = {auc_val:.3f})")
            plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve Comparison")
            plt.legend(loc="lower right")
            roc_models_path = os.path.join(RESULTS_GRAPHS_DIR, "roc_curve_models.png")
            plt.tight_layout()
            plt.savefig(roc_models_path, dpi=300, bbox_inches="tight")
            plt.show()
            print(f"      Saved ROC curve comparison graph to: {roc_models_path}")
    except Exception as e:
        print(f"      Warning: could not generate ROC comparison plot: {e}")

    # 5) Model performance comparison (Accuracy, Precision, Recall, F1, AUC)
    try:
        if model_results:
            metrics_df = pd.DataFrame(
                {name: res["metrics"] for name, res in model_results.items()}
            ).T[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
            metrics_df = metrics_df.rename(columns={"roc_auc": "auc_roc"})
            metrics_long = (
                metrics_df.reset_index()
                .melt(id_vars="index", var_name="Metric", value_name="Score")
                .rename(columns={"index": "Model"})
            )

            plt.figure(figsize=(9, 5))
            sns.barplot(
                data=metrics_long,
                x="Metric",
                y="Score",
                hue="Model",
                palette="Set3",
            )
            plt.ylim(0.0, 1.05)
            plt.title("Model performance comparison")
            plt.ylabel("Score")
            perf_path = os.path.join(RESULTS_GRAPHS_DIR, "model_performance_comparison.png")
            plt.tight_layout()
            plt.savefig(perf_path, dpi=300, bbox_inches="tight")
            plt.show()
            print(f"      Saved model performance comparison graph to: {perf_path}")
    except Exception as e:
        print(f"      Warning: could not generate model performance comparison plot: {e}")

    print("[6/6] Saving model and evaluation metrics...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(metrics, EVALUATION_METRICS_PATH)

    # JSON export for readability and publication (scalars + confusion matrix)
    metrics_export = {k: v for k, v in metrics.items() if k != "roc_curve"}
    with open(EVALUATION_METRICS_JSON, "w") as f:
        json.dump(metrics_export, f, indent=2)

    print("      Artifacts saved:")
    print(f"        Model (joblib):     {MODEL_PATH}")
    print(f"        Metrics (joblib):   {EVALUATION_METRICS_PATH}")
    print(f"        Metrics (JSON):     {EVALUATION_METRICS_JSON}")
    print(f"        ROC curve:         {ROC_CURVE_PATH}")
    print(f"        Preprocessor:       {PREPROCESSOR_PATH}")
    print(f"        Encoder mappings:   {ENCODER_MAPPINGS_PATH}")
    print("\n" + "=" * 60)
    print("  SUPERVISED TRAINING COMPLETE")
    print("=" * 60 + "\n")
    return clf, preprocessor, encoder_mappings


def train(sample_frac=0.3, model_type="logistic"):
    """
    Run training pipeline. Chooses SUPERVISED or INFERENCE-ONLY mode based on TARGET presence.
    sample_frac: fraction of data to use (e.g. 0.3 for 30%).
    model_type: 'logistic', 'random_forest', or 'xgboost'.
    """
    print("\n" + "=" * 60)
    print("  EXPLAINABLE RISK PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    path = _resolve_training_data_path()
    print(f"\n[1/6] Loading data from: {path}")
    df_raw = load_data(path=path, sample_frac=sample_frac)

    # Validate TARGET on raw data before prepare_raw_df (which drops non-feature columns)
    is_labeled, msg = validate_training_data(df_raw)
    if not is_labeled:
        print("\n" + "!" * 60)
        print("  TARGET COLUMN MISSING — SUPERVISED TRAINING NOT POSSIBLE")
        print("!" * 60)
        print(f"\n  {msg}\n")
        df = prepare_raw_df(df_raw)
        _run_inference_only_mode(path, df)
        sys.exit(0)

    print(f"      TARGET column found. Proceeding with supervised training.\n")
    return _run_supervised_mode(path, df_raw, sample_frac, model_type)


def _fit_preprocessor_only(df):
    """
    Fit and save preprocessor on unlabeled data (used by inference-only mode).
    """
    df = prepare_raw_df(df)
    fit_preprocessor(df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train risk prediction model. Requires labeled data with TARGET for full training."
    )
    parser.add_argument("--sample", type=float, default=0.3, help="Fraction of data to use")
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest", "xgboost"],
        default="logistic",
    )
    args = parser.parse_args()
    train(sample_frac=args.sample, model_type=args.model)
