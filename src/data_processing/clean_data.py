import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # goes to src/
file_path = os.path.join(BASE_DIR, "data", "External_Cibil_Dataset.xlsx", "External_Cibil_Dataset.xlsx")
df = pd.read_excel(file_path)
df.head()

df.isnull().sum()

df.info()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(numeric_cols)

X = df.drop(["PROSPECTID", "Approved_Flag"], axis=1)
y = df["Approved_Flag"]

numeric_cols = X.select_dtypes(include=['int64','float64']).columns

scaler = StandardScaler()

X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print(X.head())
print(df.head())