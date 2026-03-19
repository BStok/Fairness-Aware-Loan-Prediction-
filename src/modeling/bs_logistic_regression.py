import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

RANDOM_STATE = 42
THRESHOLD    = 0.5

input_dir   = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\splits"
results_dir = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\results\baseline\logistic_regression"
models_dir  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\models"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load splits
X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(input_dir, "y_test.csv")).squeeze()

# Pipeline — LR needs scaling
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )),
])

pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

y_train_pred = pipeline.predict(X_train)
print(f"Train accuracy: {(y_train_pred == y_train).mean():.4f}")
print(f"Test accuracy:  {(y_pred == y_test).mean():.4f}")

print("=" * 55)
print("  TEST SET RESULTS")
print("=" * 55)
print(f"  ROC-AUC Score       :  {roc_auc_score(y_test, y_prob):.4f}")
print(f"  Avg Precision Score :  {average_precision_score(y_test, y_prob):.4f}")
print(f"  Decision Threshold  :  {THRESHOLD}\n")
print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

# Plots
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Loan Approval — Logistic Regression Diagnostics", fontsize=14, fontweight="bold")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0, 0].plot(fpr, tpr, lw=2, color="#2563EB",
                label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0, 0].set(title="ROC Curve", xlabel="False Positive Rate",
               ylabel="True Positive Rate")
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

precision, recall, _ = precision_recall_curve(y_test, y_prob)
axes[0, 1].plot(recall, precision, lw=2, color="#16A34A",
                label=f"AP = {average_precision_score(y_test, y_prob):.3f}")
axes[0, 1].set(title="Precision-Recall Curve",
               xlabel="Recall", ylabel="Precision")
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["Rejected", "Approved"],
).plot(ax=axes[1, 0], colorbar=False, cmap="Blues")
axes[1, 0].set_title("Confusion Matrix")

# Coefficients instead of feature importance for LR
coef = pipeline.named_steps["model"].coef_[0]
feat_df = (pd.DataFrame({"feature": X_train.columns, "coef": coef})
           .assign(abs_coef=lambda d: d["coef"].abs())
           .nlargest(20, "abs_coef"))
colors = ["#2563EB" if c > 0 else "#DC2626" for c in feat_df["coef"]]
axes[1, 1].barh(feat_df["feature"], feat_df["coef"], color=colors)
axes[1, 1].axvline(0, color="black", linewidth=0.8)
axes[1, 1].set(title="Top-20 Feature Coefficients\n(Blue = Positive  |  Red = Negative)",
               xlabel="Coefficient")
axes[1, 1].invert_yaxis(); axes[1, 1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "logistic_regression_diagnostics.png"), dpi=150, bbox_inches="tight")
print("\nDiagnostic plot saved → results/baseline/logistic_regression_diagnostics.png")
plt.close()

# Save predictions
pred_df = X_test.copy()
pred_df["actual"]               = y_test.values
pred_df["approval_probability"] = y_prob.round(4)
pred_df["predicted_decision"]   = np.where(y_pred == 1, "Approved", "Rejected")
pred_df.to_csv(os.path.join(results_dir, "logistic_regression_predictions.csv"), index=False)
print("Predictions saved → results/baseline/logistic_regression_predictions.csv")

# Save model
with open(os.path.join(models_dir, "logistic_regression.pkl"), "wb") as f:
    pickle.dump(pipeline, f)
print("Model saved → models/logistic_regression.pkl")