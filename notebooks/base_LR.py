import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

X_TRAIN_PATH = r"C:\Users\hp\OneDrive\Desktop\ML Project1\Fairness-Aware-Loan-Prediction-\notebooks\X_train.csv"
X_VAL_PATH   = r"C:\Users\hp\OneDrive\Desktop\ML Project1\Fairness-Aware-Loan-Prediction-\notebooks\X_val.csv"
X_TEST_PATH  = r"C:\Users\hp\OneDrive\Desktop\ML Project1\Fairness-Aware-Loan-Prediction-\notebooks\X_test.csv"
Y_TRAIN_PATH = r"C:\Users\hp\OneDrive\Desktop\ML Project1\Fairness-Aware-Loan-Prediction-\notebooks\Y_train.csv"
Y_VAL_PATH   = r"C:\Users\hp\OneDrive\Desktop\ML Project1\Fairness-Aware-Loan-Prediction-\notebooks\Y_val.csv"
Y_TEST_PATH  = r"C:\Users\hp\OneDrive\Desktop\ML Project1\Fairness-Aware-Loan-Prediction-\notebooks\Y_test.csv"

TARGET_COL = "Approved_Flag"
ID_COL     = "PROSPECTID"

# non-numeric
CATEGORICAL_COLS = [
    "MARITALSTATUS",
    "EDUCATION",
    "GENDER",
    "last_prod_enq2",
    "first_prod_enq2",
]

THRESHOLD    = 0.5    # Decision threshold
RANDOM_STATE = 42

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

X_train = load_csv(X_TRAIN_PATH)
X_val   = load_csv(X_VAL_PATH)
X_test  = load_csv(X_TEST_PATH)
y_train = load_csv(Y_TRAIN_PATH)
y_val   = load_csv(Y_VAL_PATH)
y_test  = load_csv(Y_TEST_PATH)


def extract_target(df):
    if TARGET_COL in df.columns:
        return df[TARGET_COL]
    return df.iloc[:, 0]

y_train = extract_target(y_train)
y_val   = extract_target(y_val)
y_test  = extract_target(y_test)

print(f"Train  :  {X_train.shape[0]:,} rows  |  Val  :  {X_val.shape[0]:,} rows  |  Test  :  {X_test.shape[0]:,} rows")
print(f"Features  :  {X_train.shape[1]}")
print(f"Train target distribution:\n{y_train.value_counts(normalize=True).round(3)}\n")

for df in [X_train, X_val, X_test]:
    df.drop(columns=[ID_COL], errors="ignore", inplace=True)


le_encoders = {}
for col in CATEGORICAL_COLS:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        
        for split in [X_val, X_test]:
            split[col] = split[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )
        le_encoders[col] = le

print(f"Missing values in X_train:\n{X_train.isnull().sum()[X_train.isnull().sum() > 0]}")
print()

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

y_val_prob = pipeline.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_prob >= THRESHOLD).astype(int)

print("=" * 55)
print("  VALIDATION SET RESULTS")
print("=" * 55)
print(f"  ROC-AUC Score       :  {roc_auc_score(y_val, y_val_prob):.4f}")
print(f"  Avg Precision Score :  {average_precision_score(y_val, y_val_prob):.4f}")
print(f"  Decision Threshold  :  {THRESHOLD}\n")
print(classification_report(y_val, y_val_pred, target_names=["Rejected (0)", "Approved (1)"]))


y_test_prob = pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= THRESHOLD).astype(int)

print("=" * 55)
print("  TEST SET RESULTS")
print("=" * 55)
print(f"  ROC-AUC Score       :  {roc_auc_score(y_test, y_test_prob):.4f}")
print(f"  Avg Precision Score :  {average_precision_score(y_test, y_test_prob):.4f}")
print(f"  Decision Threshold  :  {THRESHOLD}\n")
print(classification_report(y_test, y_test_pred, target_names=["Rejected (0)", "Approved (1)"]))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Loan Approval — Logistic Regression Diagnostics (Test Set)", fontsize=14, fontweight="bold")


fpr, tpr, _ = roc_curve(y_test, y_test_prob)
axes[0, 0].plot(fpr, tpr, lw=2, color="#2563EB", label=f"AUC = {roc_auc_score(y_test, y_test_prob):.3f}")
axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0, 0].set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)


precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
axes[0, 1].plot(recall, precision, lw=2, color="#16A34A", label=f"AP = {average_precision_score(y_test, y_test_prob):.3f}")
axes[0, 1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)


ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_test_pred),
    display_labels=["Rejected", "Approved"],
).plot(ax=axes[1, 0], colorbar=False, cmap="Blues")
axes[1, 0].set_title("Confusion Matrix (Test Set)")


coef = pipeline.named_steps["model"].coef_[0]
feat_df = (pd.DataFrame({"feature": X_train.columns, "coef": coef}).assign(abs_coef=lambda d: d["coef"].abs()).nlargest(20, "abs_coef"))
colors = ["#2563EB" if c > 0 else "#DC2626" for c in feat_df["coef"]]
axes[1, 1].barh(feat_df["feature"], feat_df["coef"], color=colors)
axes[1, 1].axvline(0, color="black", linewidth=0.8)
axes[1, 1].set(title="Top-20 Feature Coefficients\n(Blue = Positive  |  Red = Negative)", xlabel="Coefficient")
axes[1, 1].invert_yaxis(); axes[1, 1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("loan_approval_diagnostics.png", dpi=150, bbox_inches="tight")
print("\nDiagnostic plot saved  →  loan_approval_diagnostics.png")
plt.show()

pred_df = X_test.copy()
pred_df["actual"]               = y_test.values
pred_df["approval_probability"] = y_test_prob.round(4)
pred_df["predicted_decision"]   = np.where(y_test_pred == 1, "Approved", "Rejected")
pred_df.to_csv("loan_predictions.csv", index=False)
print("Predictions saved      →  loan_predictions.csv")