# Explainable Risk Prediction System

Machine Learning–based **Explainable Risk Prediction System** using the Home Credit Default Risk dataset. The web app assesses financial default risk for **Loan** or **Credit Card** applications and provides **SHAP-based explanations** suitable for academic evaluation.

## Features

- **User flow**: Landing → Product (Loan / Credit Card) → Customer type (New / Existing)
- **New customers**: Manual input form aligned with `application_train.csv` (Personal, Employment, Housing, Credit background)
- **Existing customers**: Enter Customer ID → fetch historical features → predict
- **Output**: Probability of default, risk score (0–100), risk category (Low / Medium / High)
- **Explainability**: SHAP (base value vs final prediction, directional feature impact, top contributors), business-friendly labels, optional counterfactual suggestion

## Tech stack

- **Streamlit** (UI)
- **Python** (ML logic)
- **scikit-learn** (model: Logistic Regression / Random Forest / XGBoost)
- **SHAP** (feature contribution analysis)
- **DiCE** (optional counterfactuals; app includes rule-based counterfactual from SHAP)

## Setup

1. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (required before running the app)

   **Training vs inference separation** (important for supervised learning):
   - **Training** requires labeled data with a `TARGET` column (e.g. `application_train.csv` from Home Credit).
   - **Inference** uses unlabeled data (e.g. `application.csv`) for customer lookup and SHAP background—no `TARGET` needed.

   ```bash
   python -m ml.train_model --sample 0.3 --model logistic
   ```

   - If `TARGET` is **present**: full supervised training, evaluation metrics (ROC-AUC, accuracy, etc.), and model saved.
   - If `TARGET` is **missing**: training is skipped with a clear message; only the preprocessor is fitted (inference-only mode). Obtain labeled data to train a model.

   Options:
   - `--sample 0.3`  Use 30% of data (increase for better accuracy, e.g. `0.5` or `1.0`)
   - `--model logistic` | `random_forest` | `xgboost`

   This creates the `models/` folder with:
   - `risk_model.joblib` (only when training succeeds with labeled data)
   - `preprocessor.joblib`
   - `encoder_mappings.joblib`
   - `evaluation_metrics.joblib` (only when training succeeds)

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Open the URL shown in the terminal (e.g. http://localhost:8501).

## Dataset

- **For training** (supervised): Use **`application_train.csv`** (Home Credit) with a `TARGET` column. Falls back to `dataset.csv` or `application.csv` if present.
- **For inference** (Streamlit app): Use **`dataset.csv`** or **`application.csv`** (unlabeled). The app needs this for:
  - SHAP background sample
  - Existing-customer lookup by Customer ID

- Supported ID columns: `Customer_ID` or `SK_ID_CURR` (Home Credit).
- Feature columns: see `config.py` (e.g. `CODE_GENDER`, `AMT_INCOME_TOTAL`, `NAME_EDUCATION_TYPE`, `DAYS_BIRTH`, `EXT_SOURCE_1`, etc.).

**Important**: Training never runs during Streamlit execution. Run `python -m ml.train_model` separately.

## Project structure

```
Financial/
├── app.py              # Streamlit UI
├── config.py           # Feature mapping, paths, risk categories
├── dataset.csv         # Home Credit–style data
├── requirements.txt
├── README.md
└── ml/
    ├── __init__.py
    ├── preprocess.py   # Encoding, scaling, imputation
    ├── train_model.py  # Train and save model
    ├── predict.py     # Load model, predict, SHAP
    └── explain.py      # SHAP + counterfactual text
```

## Input features (new customers)

Aligned with `application_train.csv`:

- **Personal & demographic**: Age, Gender, Marital status, Education level, Number of children  
- **Employment & income**: Employment type, Employment length, Annual income, Income type, Occupation type  
- **Housing & stability**: Housing type, Years at current residence, Region rating  
- **Credit background**: Credit score (0–1), Credit history length, Existing credit enquiries  

Existing customers need only **Customer ID**; no manual inputs.

## Explainability

- **SHAP**: Base value vs final prediction, directional impact (+/−) per feature, top contributing features, technical names mapped to business-friendly labels in the UI.
- **Counterfactual**: A short text suggestion (e.g. “Consider showing higher or more stable annual income”) derived from top risk-increasing SHAP factors. Full DiCE integration can be added in `ml/explain.py` for minimal-change counterfactuals.

## Visualizations

- **Gauge**: Overall risk score (0–100) with Low / Medium / High bands.
- **Bar chart**: Feature impact (SHAP) with clear indication of factors increasing (red) vs decreasing (green) risk.
