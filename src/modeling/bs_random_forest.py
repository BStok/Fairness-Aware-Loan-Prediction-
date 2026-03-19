import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

RANDOM_STATE = 42
THRESHOLD    = 0.5

input_dir   = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\splits"
results_dir = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\results\baseline\random_forest"
models_dir  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\models"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load splits
X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(input_dir, "y_test.csv")).squeeze()

# Pipeline — no scaler needed for RF
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model",   RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,          # uses all CPU cores, speeds up training
    )),
])

pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# Results
print("=" * 55)
print("  TEST SET RESULTS")
print("=" * 55)
print(f"  ROC-AUC Score       :  {roc_auc_score(y_test, y_prob):.4f}")
print(f"  Avg Precision Score :  {average_precision_score(y_test, y_prob):.4f}")
print(f"  Decision Threshold  :  {THRESHOLD}\n")
print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

# Plots
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Loan Approval — Random Forest Diagnostics", fontsize=14, fontweight="bold")

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

# Feature importance
feat_df = (pd.DataFrame({
    "feature"    : X_train.columns,
    "importance" : pipeline.named_steps["model"].feature_importances_
}).nlargest(20, "importance"))
axes[1, 1].barh(feat_df["feature"], feat_df["importance"], color="#2563EB")
axes[1, 1].set(title="Top-20 Feature Importances",
               xlabel="Importance")
axes[1, 1].invert_yaxis(); axes[1, 1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "random_forest_diagnostics.png"), dpi=150, bbox_inches="tight")
print("\nDiagnostic plot saved → results/baseline/random_forest_diagnostics.png")
plt.show()

# Save predictions
pred_df = X_test.copy()
pred_df["actual"]               = y_test.values
pred_df["approval_probability"] = y_prob.round(4)
pred_df["predicted_decision"]   = np.where(y_pred == 1, "Approved", "Rejected")
pred_df.to_csv(os.path.join(results_dir, "random_forest_predictions.csv"), index=False)
print("Predictions saved → results/baseline/random_forest_predictions.csv")

# Save model
with open(os.path.join(models_dir, "random_forest.pkl"), "wb") as f:
    pickle.dump(pipeline, f)
print("Model saved → models/random_forest.pkl")