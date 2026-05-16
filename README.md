# Credit Card Fraud Detection

> Applied Machine Learning group project for the **Machine Learning Foundations** course (BDBA2025DBA.2.M.A) at IE University.

This project develops an end-to-end machine-learning pipeline that flags fraudulent credit-card transactions. We train and compare seven models on the public ULB Credit Card Fraud dataset (284,807 transactions, 0.17% fraud rate) and select a neural network operating at a cost-tuned decision threshold that catches **85% of fraud on the held-out test set** while keeping false alarms manageable.

---




---

## Problem Statement

Credit-card fraud costs the global financial industry over USD 32 billion a year. Banks need models that can flag fraudulent transactions in near-real time, with two competing constraints: missed fraud is expensive (direct financial loss), and false alarms are also costly (operational review burden and customer friction). We frame this as a **binary classification** problem with an explicit cost-sensitive decision rule.

Key technical challenges:

- **Severe class imbalance** — fraud is ~0.17% of the data (≈1 in 578 transactions).
- **Anonymised features** — 28 of 30 predictors are PCA-transformed, limiting domain interpretation.
- **Cost asymmetry** — false negatives are roughly 100× more expensive than false positives.

---

## Dataset

| | |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Size | 284,807 transactions |
| Features | `Time`, `Amount`, `V1`–`V28` (PCA components) |
| Target | `Class` (0 = legitimate, 1 = fraud) |
| Fraud rate | 0.173% (492 fraud cases) |
| Missing values | None |



---

## Project Structure

```
ml_foundations_group_project/
├── README.md                              ← this file
├── MODEL_CARD.md                          ← summary of the selected model
├── requirements.txt                       ← Python dependencies
├── .gitignore                             ← files Git should not track
├── fraud_detection_ml_project.ipynb       ← main notebook (full pipeline)
├── creditcard.csv                         ← dataset (download separately)
└── report/
    ├── Credit_Card_Fraud_Detection_Report.docx
    └── figures/
        ├── chart_class_imbalance.png
        ├── chart_model_comparison.png
        ├── chart_confusion_matrices.png
        └── chart_permutation_importance.png
```

---

## Setup

The project uses Python 3.10+ and a small set of standard data-science libraries.

```bash
# 1. Clone the repository
git clone https://github.com/vvarshhh/ml_foundations_group_project.git
cd ml_foundations_group_project

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---


---

## Key Results

Final model: **MLP (neural network)** evaluated at the cost-sensitive threshold of 0.028.

| Setting | Recall | Precision | F1 | ROC-AUC | PR-AUC | Total cost |
|---|---:|---:|---:|---:|---:|---:|
| Default threshold (0.5) | 70.3% | 89.7% | 0.788 | 0.965 | 0.812 | 2,206 |
| **Cost-sensitive (0.028)** | **85.1%** | 43.4% | 0.575 | 0.965 | 0.812 | **1,182** |

Tuning the decision threshold for the 1:100 cost ratio **nearly halves the expected cost** and lifts fraud recall by 15 percentage points, catching 63 of 74 frauds on the held-out test set.

A full set of results — including validation tables for all seven models, ROC and PR curves, confusion matrices, and SHAP/permutation-importance plots — is included in the report under `report/`.

---

## Methods Summary

| Stage | Approach |
|---|---|
| Feature engineering | `log1p(Amount)` to reduce skew; cyclical encoding of hour-of-day from `Time` |
| Scaling | `StandardScaler` inside the pipeline (leakage-safe) |
| Imbalance handling | `class_weight='balanced'` and SMOTE (inside `imblearn` pipeline) |
| Data split | Stratified 70 / 15 / 15 → train / validation / test |
| Models | Dummy, Logistic Regression, Logistic + SMOTE, Random Forest, MLP, XGBoost |
| Tuning | `GridSearchCV` (3-fold stratified, average-precision scoring) |
| Selection | Validation PR-AUC + cost-sensitive threshold sweep |
| Interpretation | SHAP values + permutation importance |

---

## References

- Pozzolo, A. D., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating probability with undersampling for unbalanced classification.* IEEE Symposium on Computational Intelligence and Data Mining (CIDM).
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD '16.
- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
- Chawla, N. V., et al. (2002). *SMOTE: Synthetic minority over-sampling technique.* JAIR 16.
- scikit-learn, imbalanced-learn, XGBoost, SHAP, pandas, NumPy, Matplotlib, Seaborn.

---

*Course project — IE University, Spring 2026. Prof. Matteo Turilli.*
