import pandas as pd
import numpy as np
import pickle
import os
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    demographic_parity_ratio,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ── Paths ────────────────────────────────────────────────
splits_dir  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\splits"
sens_path   = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\cleaning02\sensitive_features.csv"
models_dir  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\models\baseline"
output_dir  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\results\fairness"
os.makedirs(output_dir, exist_ok=True)

# ── Load test data ───────────────────────────────────────
X_test   = pd.read_csv(os.path.join(splits_dir, "X_test.csv"))
y_test   = pd.read_csv(os.path.join(splits_dir, "y_test.csv")).squeeze()
ids_test = pd.read_csv(os.path.join(splits_dir, "X_test_ids.csv"))

# ── Load and join sensitive features ────────────────────
sens     = pd.read_csv(sens_path)
merged   = ids_test.merge(sens, on='PROSPECTID', how='left')

# ── AGE bins ─────────────────────────────────────────────
def bin_age(age):
    if age <= 30:   return "Young (18-30)"
    elif age <= 50: return "Mid (31-50)"
    else:           return "Senior (51+)"

age_groups        = merged['AGE'].apply(bin_age)
gender_groups     = merged['GENDER']
marital_groups    = merged['MARITALSTATUS']

print("=" * 55)
print("SENSITIVE FEATURE DISTRIBUTIONS IN TEST SET")
print("=" * 55)
print("\nAGE groups:")
print(age_groups.value_counts())
print("\nGENDER:")
print(gender_groups.value_counts())
print("\nMARITALSTATUS:")
print(marital_groups.value_counts())

# ── Metrics dict ─────────────────────────────────────────
metrics = {
    "accuracy"  : accuracy_score,
    "f1"        : f1_score,
    "precision" : precision_score,
    "recall"    : recall_score,
}

# ── Helper ───────────────────────────────────────────────
def run_metricframe(model_name, y_pred, sensitive_feature, feature_name):
    mf = MetricFrame(
        metrics            = metrics,
        y_true             = y_test,
        y_pred             = y_pred,
        sensitive_features = sensitive_feature,
    )

    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature)
    eod = equalized_odds_difference(y_test,     y_pred, sensitive_features=sensitive_feature)
    dpr = demographic_parity_ratio(y_test,      y_pred, sensitive_features=sensitive_feature)

    print(f"\n{'=' * 55}")
    print(f"  {model_name.upper()} — by {feature_name}")
    print(f"{'=' * 55}")
    print(f"\nOverall metrics:")
    print(mf.overall)
    print(f"\nBy {feature_name}:")
    print(mf.by_group)
    print(f"\nDemographic Parity Difference : {dpd:.4f}  (0 = fair)")
    print(f"Demographic Parity Ratio      : {dpr:.4f}  (1 = fair)")
    print(f"Equalized Odds Difference     : {eod:.4f}  (0 = fair)")

    # save by group breakdown
    mf.by_group.to_csv(
        os.path.join(output_dir, f"{model_name}_{feature_name}_bygroup.csv")
    )

    return {
        "model"                        : model_name,
        "sensitive_feature"            : feature_name,
        "overall_accuracy"             : mf.overall["accuracy"],
        "overall_f1"                   : mf.overall["f1"],
        "demographic_parity_difference": dpd,
        "demographic_parity_ratio"     : dpr,
        "equalized_odds_difference"    : eod,
    }

# ── Run for all models × all sensitive features ──────────
model_files = {
    "logistic_regression" : "logistic_regression.pkl",
    "decision_tree"       : "decision_tree.pkl",
    "random_forest"       : "random_forest.pkl",
}

sensitive_features = {
    "AGE"          : age_groups,
    "GENDER"       : gender_groups,
    "MARITALSTATUS": marital_groups,
}

all_results = []

for model_name, fname in model_files.items():
    fpath = os.path.join(models_dir, fname)
    if not os.path.exists(fpath):
        print(f"\nSKIPPING {fname} — not found in models/baseline/")
        continue

    with open(fpath, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    for feature_name, feature_groups in sensitive_features.items():
        result = run_metricframe(model_name, y_pred, feature_groups, feature_name)
        all_results.append(result)

# ── Save combined comparison table ───────────────────────
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(os.path.join(output_dir, "all_models_fairness_summary.csv"), index=False)

print("\n" + "=" * 55)
print("COMBINED FAIRNESS SUMMARY")
print("=" * 55)
print(summary_df.to_string(index=False))

# ── Identify best model for ThresholdOptimizer ───────────
print("\n" + "=" * 55)
print("BEST MODEL FOR THRESHOLDOPTIMIZER")
print("=" * 55)
age_results = summary_df[summary_df['sensitive_feature'] == 'AGE'].copy()
age_results = age_results.sort_values('overall_f1', ascending=False)
best_model  = age_results.iloc[0]['model']
print(f"Highest F1 on AGE split → {best_model}")
print(f"Tell Member B to run ThresholdOptimizer on: {best_model}")