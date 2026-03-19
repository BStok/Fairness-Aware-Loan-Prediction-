import pandas as pd
import numpy as np
import pickle
import os
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────
splits_dir  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\splits"
sens_path   = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\cleaning02\sensitive_features.csv"
models_dir  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\models\baseline"
output_dir  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\results\fairness"
os.makedirs(output_dir, exist_ok=True)

# ── Load data ────────────────────────────────────────────
X_train   = pd.read_csv(os.path.join(splits_dir, "X_train.csv"))
X_test    = pd.read_csv(os.path.join(splits_dir, "X_test.csv"))
y_train   = pd.read_csv(os.path.join(splits_dir, "y_train.csv")).squeeze()
y_test    = pd.read_csv(os.path.join(splits_dir, "y_test.csv")).squeeze()
ids_train = pd.read_csv(os.path.join(splits_dir, "X_train_ids.csv"))
ids_test  = pd.read_csv(os.path.join(splits_dir, "X_test_ids.csv"))

# ── Load and join sensitive features ────────────────────
sens = pd.read_csv(sens_path)

train_merged = ids_train.merge(sens, on='PROSPECTID', how='left')
test_merged  = ids_test.merge(sens,  on='PROSPECTID', how='left')

# ── AGE bins ─────────────────────────────────────────────
def bin_age(age):
    if age <= 30:   return "Young (18-30)"
    elif age <= 50: return "Mid (31-50)"
    else:           return "Senior (51+)"

age_train = train_merged['AGE'].apply(bin_age)
age_test  = test_merged['AGE'].apply(bin_age)

# ── Load baseline LR model ───────────────────────────────
with open(os.path.join(models_dir, "logistic_regression.pkl"), "rb") as f:
    baseline_model = pickle.load(f)

y_pred_baseline = baseline_model.predict(X_test)

# ── Fit ThresholdOptimizer ───────────────────────────────
to = ThresholdOptimizer(
    estimator       = baseline_model,
    constraints     = "equalized_odds",
    objective       = "balanced_accuracy_score",
    predict_method  = "predict_proba",
)
to.fit(X_train, y_train, sensitive_features=age_train)
y_pred_fair = to.predict(X_test, sensitive_features=age_test)

# ── MetricFrame on both ──────────────────────────────────
metrics = {
    "accuracy"  : accuracy_score,
    "f1"        : f1_score,
    "precision" : precision_score,
    "recall"    : recall_score,
}

mf_baseline = MetricFrame(
    metrics            = metrics,
    y_true             = y_test,
    y_pred             = y_pred_baseline,
    sensitive_features = age_test,
)
mf_fair = MetricFrame(
    metrics            = metrics,
    y_true             = y_test,
    y_pred             = y_pred_fair,
    sensitive_features = age_test,
)

# ── Print comparison ─────────────────────────────────────
print("=" * 55)
print("BASELINE — by AGE")
print("=" * 55)
print(mf_baseline.by_group)

print("\n" + "=" * 55)
print("AFTER THRESHOLDOPTIMIZER — by AGE")
print("=" * 55)
print(mf_fair.by_group)

# ── Summary table ────────────────────────────────────────
results = pd.DataFrame({
    "version"                        : ["Baseline LR", "Fair LR (ThresholdOptimizer)"],
    "overall_accuracy"               : [accuracy_score(y_test, y_pred_baseline),
                                        accuracy_score(y_test, y_pred_fair)],
    "overall_f1"                     : [f1_score(y_test, y_pred_baseline),
                                        f1_score(y_test, y_pred_fair)],
    "demographic_parity_difference"  : [demographic_parity_difference(y_test, y_pred_baseline, sensitive_features=age_test),
                                        demographic_parity_difference(y_test, y_pred_fair,     sensitive_features=age_test)],
    "demographic_parity_ratio"       : [demographic_parity_ratio(y_test, y_pred_baseline, sensitive_features=age_test),
                                        demographic_parity_ratio(y_test, y_pred_fair,     sensitive_features=age_test)],
    "equalized_odds_difference"      : [equalized_odds_difference(y_test, y_pred_baseline, sensitive_features=age_test),
                                        equalized_odds_difference(y_test, y_pred_fair,     sensitive_features=age_test)],
})

print("\n" + "=" * 55)
print("COMPARISON SUMMARY")
print("=" * 55)
print(results.to_string(index=False))

# ── Plots ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ThresholdOptimizer — Baseline vs Fair LR by AGE group", fontsize=13, fontweight="bold")

# F1 by group
f1_df = pd.DataFrame({
    "Baseline" : mf_baseline.by_group["f1"],
    "Fair"     : mf_fair.by_group["f1"],
})
f1_df.plot(kind="bar", ax=axes[0], color=["#DC2626", "#16A34A"], rot=15)
axes[0].set_title("F1 Score by AGE Group")
axes[0].set_ylabel("F1 Score")
axes[0].set_ylim(0, 1)
axes[0].grid(axis="y", alpha=0.3)
axes[0].legend()

# Fairness metrics comparison
fair_metrics = ["demographic_parity_difference", "equalized_odds_difference"]
baseline_vals = [results.iloc[0][m] for m in fair_metrics]
fair_vals     = [results.iloc[1][m] for m in fair_metrics]
x = np.arange(len(fair_metrics))
axes[1].bar(x - 0.2, baseline_vals, 0.4, label="Baseline", color="#DC2626")
axes[1].bar(x + 0.2, fair_vals,     0.4, label="Fair",     color="#16A34A")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Dem. Parity Diff", "Equalized Odds Diff"], fontsize=9)
axes[1].set_title("Fairness Metrics: Baseline vs Fair")
axes[1].set_ylabel("Difference (lower = fairer)")
axes[1].grid(axis="y", alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "thresholdoptimizer_comparison.png"), dpi=150, bbox_inches="tight")
print(f"\nPlot saved → results/fairness/thresholdoptimizer_comparison.png")
plt.close()

# ── Save results ─────────────────────────────────────────
results.to_csv(os.path.join(output_dir, "thresholdoptimizer_summary.csv"), index=False)
mf_fair.by_group.to_csv(os.path.join(output_dir, "thresholdoptimizer_bygroup.csv"))

# ── Save fair model ──────────────────────────────────────
fair_model_path = os.path.join(models_dir, "logistic_regression_fair.pkl")
with open(fair_model_path, "wb") as f:
    pickle.dump(to, f)
print(f"Fair model saved → models/baseline/logistic_regression_fair.pkl")