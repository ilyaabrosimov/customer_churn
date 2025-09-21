# Customer Churn — EDA, Modeling, and Report

A reproducible capstone-style project that explores **customer churn** with thorough **EDA**, feature engineering, and baseline **ML models**.  
Outputs include rich visualizations (KDE grids, heatmaps, stacked bars, Sankey, etc.), model comparison with class-imbalance strategies, and **Word report** for non-technical audiences.

---

## 🔧 Quickstart

**Python**: 3.9–3.12

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -U pip
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn \
            statsmodels plotly python-docx jupyter
jupyter lab
```

Place your CSV in `data/raw/` (or edit `DATA_PATH` in the notebook).  
The project expects a target column named **`Churn`** (values like Yes/No).  
If your file uses a different name, adjust the `TARGET` detection cell.

---

## 🎯 Objectives

- Clean and understand the dataset (missing values, dtypes, duplicates, outliers).
- Explore relationships between features and **churn** with compelling visuals.
- Engineer simple, leakage-safe features (e.g., `tenure_bucket`, `services_count`).
- Train **baseline classifiers** with **class imbalance** handling:
  - Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, SVM
  - Strategies: no handling, **class weights**, **RandomOverSampler**, **SMOTE**, **RandomUnderSampler**
- Evaluate with **ROC AUC**, **Precision**, **Recall**, **F1**, plus ROC curves.
- Export clean data, metrics, and a polished **Word summary**.

---

## 🧹 Data cleaning & feature engineering

- Trim strings, cast numerics (e.g., `TotalCharges`), convert booleans/Yes–No.
- Remove exact duplicates; simple imputation for EDA.
- IQR capping for numeric outliers.
- Features:
  - `tenure_bucket` (<=6m, 7–12m, …)
  - `services_count` (sum of yes/no service flags)
  - Binary convenience flags (`PaperlessBilling_bin`, `Partner_bin`, `Dependents_bin`)

---

## 📊 Visualizations (saved in `figures/`)

- **Class balance** bar.
- **Numeric**: histograms, **boxplots vs churn**, **KDE grids** (optionally overlay Churn/Not churn).
- **Correlations**: numeric heatmap.
- **Categorical vs churn**:
  - 100% **stacked bars** (churn share by category)
  - **Cat×Cat heatmaps** (churn rate by pair of categories)
  - **Facet** churn-rate grids (bar within row panels)
  - **Risk deviation** (dot/lollipop vs base churn)
  - **Mosaic** plots (requires `statsmodels`)
  - **Sankey** flows (requires `plotly`) for category → churn paths

Tune speed/readability with constants like `MAX_LEVELS`, `MIN_COUNT`, `TOP_K`.

---

## 🤖 Modeling & class imbalance

**Preprocessing (ColumnTransformer)**
- Numeric: median imputation + `StandardScaler`
- Categorical: most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`

**Models**
- `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`,
  `HistGradientBoostingClassifier`, `SVC(probability=True)`

**Imbalance strategies**
- none, **class_weight**, **oversample**, **SMOTE**, **undersample**

Each (model × strategy) is evaluated; results are stored in:
- `reports/model_results.csv`
- `reports/metrics.json`

A helper block **rebuilds the best pipeline**, plots the **confusion matrix**, and prints a **classification report**.

---

## ✅ What to expect in the results

Typical (Telco-like) risk drivers surfaced by EDA:

- **Contract**: **Month-to-month** churns much more than **Two year**.
- **OnlineSecurity / TechSupport**: **No** → high churn; “No internet service” → low churn.
- **InternetService**: **Fiber optic** users churn more (price/quality sensitive).
- **PaymentMethod**: **Electronic check** churns more than **Credit card (automatic)**.
- **PaperlessBilling**: **Yes** slightly increases churn vs **No** (co-varies with other factors).

Use the provided **dot/lollipop** and **stacked bars** to communicate which categories are above/below base churn.

---

## 🔁 Reproducibility

- Random seeds set to `42` in split/models where applicable.
- All file paths are created automatically; change `PROJECT_ROOT`/`DATA_PATH` as needed.

---

## 📌 Mapping to rubric (what graders look for)

- **Project organization** → created folder structure, clean notebook, saved artifacts.
- **Syntax & code quality** → modular helpers, pipelines, comments, no long monoliths.
- **Visualizations** → readable titles/labels, appropriate plots for data types, facetting.
- **Data cleaning & EDA** → imputation, duplicates, outliers, feature engineering.
- **Modeling** → appropriate classifier(s), clear metric choice & interpretation (ROC AUC, PR/F1), rationale for imbalance handling.

---

## 📄 Dataset

This project assumes a Telco-style churn dataset with a binary `Churn` column and service/billing features.  
If you use a different schema:
- Rename/match the target, update yes/no mappings, and adjust the feature engineering list.

---

## Business recommendations
- Segment customers by churn risk (e.g., p ≥ 0.7 / 0.4–0.7 / < 0.4).
- Targeted retention offers for high-risk: discounts/bundles, switch to annual.
- Proactive support outreach for high-risk.
- Plan A/B tests based on top feature importances (pricing, contract type, tenure).

---

## Next steps
- Establish baseline (churn, CAC/LTV, NPS).
- Choose decision threshold by FP/FN costs.
- Monthly drift monitoring & retraining; dashboard with uplift & model metrics.

