# Fairness-Aware Loan Approval

> *Detecting and mitigating demographic bias in Indian bank loan decisions using CIBIL data and Fairlearn.*

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Fairness Objective](#fairness-objective)
4. [Folder Structure](#folder-structure)
5. [Pipeline Workflow](#pipeline-workflow)
6. [Setup & Installation](#setup--installation)
7. [Running the Pipeline](#running-the-pipeline)
8. [Team Collaboration Notes](#team-collaboration-notes)
9. [References](#references)

---

## Project Overview

This project investigates algorithmic fairness in automated loan approval systems used by a Leading Indian Bank. Using applicant credit profiles derived from the CIBIL (Credit Information Bureau India Limited) dataset, we:

- Build three baseline classifiers (Logistic Regression, Decision Tree, Random Forest).
- Audit each model for bias across protected demographic attributes (gender, age group, caste category, region).
- Apply Fairlearn-based mitigation strategies (ThresholdOptimizer, ExponentiatedGradient) to reduce unfair disparities without sacrificing model utility.

The project is structured for clean collaboration between **3 contributors** and follows reproducible ML engineering practices.

---

## Dataset Description

| Attribute | Details |
|---|---|
| **Source** | Leading Indian Bank & CIBIL dataset |
| **Task** | Binary classification — loan approved (1) / rejected (0) |
| **Key Features** | Credit score, loan amount, income, employment type, repayment history, LTV ratio, bureau enquiries, etc. |
| **Protected Attributes** | Gender, Age Group, Caste Category, Region |
| **Format** | CSV / Excel |

> ⚠️ **The dataset is NOT included in this repository.** Place the raw file(s) inside `data/raw/` on your local machine. See [Running the Pipeline](#running-the-pipeline) for the expected filename.

---

## Fairness Objective

Automated credit scoring can perpetuate or amplify historical societal inequities. Our goals are to:

1. **Measure** demographic disparity using Fairlearn's `MetricFrame` — selection rate, TPR, FPR, Demographic Parity Difference, and Equalized Odds Difference.
2. **Mitigate** identified disparities using two complementary approaches:
   - **Post-processing** — `ThresholdOptimizer` with a Demographic Parity constraint.
   - **In-processing** — `ExponentiatedGradient` with a Demographic Parity constraint.
3. **Report** the accuracy–fairness trade-off for each technique and each protected attribute.

---

## Folder Structure

```
fairness-aware-loan-approval/
│
├── data/
│   ├── raw/                    ← Place your dataset here (git-ignored)
│   └── processed/              ← Cleaned & feature-selected CSVs (git-ignored)
│
├── notebooks/
│   └── 01_data_exploration.ipynb   ← EDA, distributions, correlation analysis
│
├── src/
│   ├── data_processing/
│   │   ├── clean_data.py           ← Deduplication, imputation, encoding
│   │   └── feature_selection.py    ← Variance, correlation, SelectKBest
│   ├── modeling/
│   │   └── train_baseline_models.py← LR, DT, RF training & evaluation
│   └── fairness/
│       ├── fairness_audit.py       ← MetricFrame bias audit
│       └── fairness_mitigation.py  ← ThresholdOptimizer & ExpGradient
│
├── models/                     ← Serialised .joblib model files
├── results/                    ← Metrics CSVs, fairness plots
├── report/                     ← Final project report (PDF / DOCX)
├── tests/                      ← Unit tests (pytest)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline Workflow

```
data/raw/loan_data.csv
        │
        ▼
src/data_processing/clean_data.py
   → data/processed/cleaned_data.csv
        │
        ▼
src/data_processing/feature_selection.py
   → data/processed/features.csv
        │
        ▼
src/modeling/train_baseline_models.py
   → models/*.joblib
   → results/baseline_metrics.csv
        │
        ▼
src/fairness/fairness_audit.py
   → results/fairness_audit_report.csv
   → results/fairness_<model>_<attribute>.png
        │
        ▼
src/fairness/fairness_mitigation.py
   → models/threshold_optimizer_<attr>.joblib
   → models/exponentiated_gradient_<attr>.joblib
   → results/mitigation_comparison.csv
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- `pip`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/fairness-aware-loan-approval.git
cd fairness-aware-loan-approval

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your dataset
cp /path/to/your/loan_data.csv data/raw/loan_data.csv
```

---

## Running the Pipeline

Execute scripts in order from the **project root**:

```bash
# Step 1 — Clean data
python src/data_processing/clean_data.py

# Step 2 — Feature selection
python src/data_processing/feature_selection.py

# Step 3 — Train baseline models
python src/modeling/train_baseline_models.py

# Step 4 — Fairness audit
python src/fairness/fairness_audit.py

# Step 5 — Fairness mitigation
python src/fairness/fairness_mitigation.py
```

To run the exploratory notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

To run unit tests:

```bash
pytest tests/
```

---

## Team Collaboration Notes

| Contributor | Suggested Ownership |
|---|---|
| Member 1 | `src/data_processing/` + `notebooks/` |
| Member 2 | `src/modeling/` + `tests/` |
| Member 3 | `src/fairness/` + `report/` |

- Work on **feature branches**: `git checkout -b feature/<your-task>`
- Open a **pull request** into `main`; require at least **1 reviewer approval**.
- Never commit data files — `.gitignore` enforces this.
- Pin dependency versions in `requirements.txt` before merging to `main`.

---

## References

- Fairlearn documentation: https://fairlearn.org
- CIBIL: https://www.cibil.com
- Bird et al. (2020). *Fairlearn: A toolkit for assessing and improving fairness in AI*. Microsoft Research.
- Hardt, Price & Srebro (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS.
