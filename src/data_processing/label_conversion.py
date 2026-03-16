import pandas as pd

# Read from partner's output
df = pd.read_csv('C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\raw\External_Cibil_Dataset.csv')  # adjust filepath

# Convert target
priority_mapping = {'P1': 0, 'P2': 0, 'P3': 1, 'P4': 1}
df['loan_status_binary'] = df['priority_column'].map(priority_mapping)
df.drop(columns=['priority_column'], inplace=True)

# Save to processed
df.to_csv('data/processed/loan_data_processed.csv', index=False)
print("Saved to data/processed/loan_data_processed.csv")
print(df['loan_status_binary'].value_counts(normalize=True) * 100)