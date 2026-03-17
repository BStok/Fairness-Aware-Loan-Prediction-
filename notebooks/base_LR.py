import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH   = "External_Cibil_Dataset.xlsx"      # adjust
TARGET_COL  = "Approved_Flag"
ID_COL      = "PROSPECTID"

# non-numeric
CATEGORICAL_COLS = [
    "MARITALSTATUS",
    "EDUCATION",
    "GENDER",
    "last_prod_enq2",
    "first_prod_enq2",
]

THRESHOLD   = 0.5    # Decision threshold
TEST_SIZE   = 0.20
RANDOM_STATE = 42

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()

print(f"Data loaded  :  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
print(f"Target distribution:\n{df[TARGET_COL].value_counts(normalize=True).round(3)}\n")

df.drop(columns=[ID_COL], errors="ignore", inplace=True)

le = LabelEncoder()
for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"Features     :  {X.shape[1]}")
print(f"Missing values:\n{X.isnull().sum()[X.isnull().sum() > 0]}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Train size   :  {X_train.shape[0]:,}")
print(f"Test size    :  {X_test.shape[0]:,}\n")


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
print("Model training complete.\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"5-Fold CV ROC-AUC  :  {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}\n")

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

print("=" * 55)
print("  TEST SET RESULTS")
print("=" * 55)
print(f"  ROC-AUC Score       :  {roc_auc_score(y_test, y_prob):.4f}")
print(f"  Avg Precision Score :  {average_precision_score(y_test, y_prob):.4f}")
print(f"  Decision Threshold  :  {THRESHOLD}\n")
print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

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

coef = pipeline.named_steps["model"].coef_[0]
feat_df = (pd.DataFrame({"feature": X.columns, "coef": coef})
           .assign(abs_coef=lambda d: d["coef"].abs())
           .nlargest(20, "abs_coef"))
colors = ["#2563EB" if c > 0 else "#DC2626" for c in feat_df["coef"]]
axes[1, 1].barh(feat_df["feature"], feat_df["coef"], color=colors)
axes[1, 1].axvline(0, color="black", linewidth=0.8)
axes[1, 1].set(title="Top-20 Feature Coefficients\n(Blue = Positive  |  Red = Negative)",
               xlabel="Coefficient")
axes[1, 1].invert_yaxis(); axes[1, 1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("loan_approval_diagnostics.png", dpi=150, bbox_inches="tight")
print("\nDiagnostic plot saved  →  loan_approval_diagnostics.png")
plt.show()

pred_df = X_test.copy()
pred_df["actual"]               = y_test.values
pred_df["approval_probability"] = y_prob.round(4)
pred_df["predicted_decision"]   = np.where(y_pred == 1, "Approved", "Rejected")
pred_df.to_csv("loan_predictions.csv", index=False)
print("Predictions saved      →  loan_predictions.csv")