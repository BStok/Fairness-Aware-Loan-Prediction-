import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

input_path  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\binaryLabel\label_conversion.csv"
output_dir  = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\cleaning02"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

print("=" * 50)
print("BEFORE ENCODING")
print("=" * 50)
print(f"Shape: {df.shape}")

# ── Save sensitive feature columns BEFORE encoding ──────
# These are needed raw for MetricFrame fairness audit later
sensitive_cols = ['GENDER', 'MARITALSTATUS', 'AGE']
df_sensitive   = df[['PROSPECTID'] + sensitive_cols].copy()
df_sensitive.to_csv(os.path.join(output_dir, "sensitive_features.csv"), index=False)
print(f"\nSensitive features saved separately → cleaning/sensitive_features.csv")
print(f"Columns saved: {sensitive_cols}")

# ── OneHot encode categorical columns ───────────────────
categorical_cols = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2']

encoder = OneHotEncoder(
    drop='first',               # avoids dummy variable trap
    sparse_output=False,        # returns dense array
    handle_unknown='ignore'     # safety for unseen categories at test time
)

encoded_array   = encoder.fit_transform(df[categorical_cols])
encoded_columns = encoder.get_feature_names_out(categorical_cols)
df_encoded      = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)

# Drop original categorical cols and join encoded ones
df = df.drop(columns=categorical_cols)
df = pd.concat([df, df_encoded], axis=1)

print("\n" + "=" * 50)
print("AFTER ENCODING")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nNew columns added:")
for col in encoded_columns:
    print(f"  + {col}")

# ── Save ────────────────────────────────────────────────
output_path = os.path.join(output_dir, "External_Cibil_Dataset.csv")
df.to_csv(output_path, index=False)
print(f"\nSaved encoded dataset → cleaning/External_Cibil_Dataset.csv")