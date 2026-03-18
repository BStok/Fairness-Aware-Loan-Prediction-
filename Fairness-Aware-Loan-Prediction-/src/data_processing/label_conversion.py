import pandas as pd

df = pd.read_csv(r'C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\feature_reduction\External_Cibil_Dataset.csv')

priority_mapping = {'P1': 0, 'P2': 0, 'P3': 1, 'P4': 1}

# Check distribution BEFORE dropping
print(df['Approved_Flag'].value_counts(normalize=True) * 100)

# Map target
df['loan_status_binary'] = df['Approved_Flag'].map(priority_mapping)

# Drop original
df.drop(columns=['Approved_Flag'], inplace=True)

# Save properly
df.to_csv(r'C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\binaryLabel\label_conversion.csv', index=False)

print("Saved to data/processed/label_conversion.csv")