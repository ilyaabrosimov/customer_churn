# Capstone Project  
**Customer Churn Prediction — Final Report**

- 📓 Notebook: `customer_churn_analysis.ipynb`  
- 📁 Figures: `figures/` (все графики, упомянутые ниже)  
- 📁 Reports: `reports/` (включая `model_results.csv`, `gridsearch_results.csv`)

---

## 1) Define the Problem Statement
We aim to predict **customer churn** (whether a customer will leave the service in the near future) to focus retention actions on at-risk users.  
**Goals:** reduce churn rate, prioritize outreach to high-risk customers, and improve ROI of retention campaigns.  
**Challenges:** class imbalance (fewer churners), heterogeneous feature types (demographics, services, billing), and the need for interpretable insights for business teams.  
**Business benefits:** early identification of at-risk customers enables targeted offers (contract upgrades, bundles, auto-pay incentives) and improves customer lifetime value (LTV).

---

## 2) Model Outcomes or Predictions
- **Learning type:** supervised **binary classification**.  
- **Model output:** probability of churn (`P(churn)` from 0 to 1) and a class label via a chosen threshold.  
- **Usage:** rank customers by risk for retention campaigns; threshold can be tuned by business costs (FP/FN).

---

## 3) Data Acquisition
- **Primary source:** Kaggle “Customer Churn Dataset” (Telco-style), with target **`Churn`** (Yes/No) and features covering demographics, services, and billing.
- **Data coverage:** ~7k customers, ~20+ columns (contract type, internet service, add-on services, charges, tenure, payment method, etc.).
- **Sanity checks:** class balance inspection, feature completeness, and basic distribution checks.
- **Visuals to assess fitness:**  
  - Class balance bar plot (`figures/class_balance.png`)  
  - Numeric distributions & boxplots by churn (`figures/num_distributions_*.png`)  
  - Categorical churn rates (`figures/cat_churn_rate__*.png`)  
  - Correlation heatmap for numeric variables (`figures/corr_heatmap.png`)

> If additional internal data is available (support tickets, payment delays, NPS/CSAT), it can further improve the model in future iterations.

---

## 4) Data Preprocessing / Preparation
**a. Missing values & inconsistencies**
- Trimmed strings and unified Yes/No values.
- Casted numerics (e.g., `TotalCharges`) and handled missing values (simple imputation in EDA; pipeline imputation in modeling).
- Removed duplicates; capped numeric outliers via IQR where relevant for EDA.

**b. Train/test split**
- Stratified split into **train/test** to preserve churn ratio (e.g., 80/20).  
- Random seed fixed for reproducibility.

**c. Encoding & pipelines**
- **ColumnTransformer** with two branches:
  - **Numeric:** median imputation → `StandardScaler`.
  - **Categorical:** most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`.
- Feature engineering (leakage-safe): e.g., `tenure_bucket`, `services_count`, binaries for convenience.
- Class imbalance strategies compared in baseline loop: none, `class_weight="balanced"`, **SMOTE/oversampling**, **undersampling**.

---

## 5) Modeling
We trained multiple classification models:
- **Baseline sweep:** `LogisticRegression`, `DecisionTree`, `RandomForest`, `HistGradientBoosting`, `SVC(probability=True)`  
  × imbalance strategies (none / class_weight / over/under-sampling / SMOTE).  
- **Hyperparameter tuning:** **GridSearchCV** with **StratifiedKFold (n=5)** for  
  `LogisticRegression`, `DecisionTree`, `RandomForest`, `GradientBoosting`, `HistGradientBoosting`, `SVC`, `KNN`  
  (optionally XGBoost/LightGBM if installed).  
- **Primary selection metric:** **ROC AUC** (robust under class imbalance). We also report **F1**, **Precision**, **Recall**, **Accuracy** on the test set.

---

## 6) Model Evaluation
- **Why ROC AUC:** ranks positive class well even with imbalance; independent of a single threshold.  
- **Additional metrics:** F1, Precision, Recall, Accuracy at default threshold 0.5 for operational view.

**Summary of tuned models (from GridSearchCV, test set):**
- **Best model:** `GradientBoosting` — Test **ROC AUC ≈ 0.847**, **F1 ≈ 0.588**, **Accuracy ≈ 0.805**.  
  **Best params:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=2`, `subsample=0.8`.  
- Full comparison table and bar charts are generated in the notebook and saved to:
  - `reports/gridsearch_results.csv`
  - `figures/model_compare__test_auc.png`, `figures/model_compare__test_f1.png`

**Interpretation & business insights (from model importances/coefficients):**
- `Contract = Month-to-month`, `InternetService = Fiber optic`, low `tenure`, high `MonthlyCharges`, and gaps in add-on services (`OnlineSecurity = No`, `TechSupport = No`) **increase** churn risk.  
- `One/Two-year` contracts and longer `tenure` **reduce** churn risk.  
- Actionable levers: promote longer-term contracts, bundle OnlineSecurity/TechSupport, and incentivize auto-pay instead of electronic check.

> Next improvements: select decision threshold by **business cost** (FP/FN), calibrate probabilities (Platt/Isotonic), monitor drift monthly, expand features (support/NPS/payments), and A/B test retention offers on top-risk segments.

## 🥇 Top Model (from GridSearchCV)

> This section summarizes the best model by **Test ROC AUC** (tie-break by **Test F1**).  
> It is auto-generated from `gridsearch_results.csv` in the notebook.

| Model | CV AUC | Test AUC | Test F1 | Precision | Recall | Accuracy |
|:------|-------:|---------:|--------:|----------:|-------:|---------:|
| GradientBoosting | 0.851 | 0.847 | 0.588 | 0.669 | 0.524 | 0.805 |

**Best hyperparameters:**

  "clf__learning_rate": 0.05
  "clf__max_depth": 2
  "clf__n_estimators": 200
  "clf__subsample": 0.8

