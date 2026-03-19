import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_path = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\cleaning02\External_Cibil_Dataset.csv"
output_dir = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\splits"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

# keep PROSPECTID separate before dropping
prospect_ids = df['PROSPECTID']

X = df.drop(columns=['loan_status_binary', 'PROSPECTID'])
y = df['loan_status_binary']

X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
    X, y, prospect_ids, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
    X_temp, y_temp, ids_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# save splits
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_val.to_csv(os.path.join(output_dir,   "X_val.csv"),   index=False)
X_test.to_csv(os.path.join(output_dir,  "X_test.csv"),  index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_val.to_csv(os.path.join(output_dir,   "y_val.csv"),   index=False)
y_test.to_csv(os.path.join(output_dir,  "y_test.csv"),  index=False)

# save PROSPECTID indexes separately
ids_train.to_csv(os.path.join(output_dir, "X_train_ids.csv"), index=False)
ids_val.to_csv(os.path.join(output_dir,   "X_val_ids.csv"),   index=False)
ids_test.to_csv(os.path.join(output_dir,  "X_test_ids.csv"),  index=False)

print(f"Train : {X_train.shape[0]:,} rows")
print(f"Val   : {X_val.shape[0]:,} rows")
print(f"Test  : {X_test.shape[0]:,} rows")
print(f"Features: {X_train.shape[1]}")
print(f"\nSaved to: {output_dir}")