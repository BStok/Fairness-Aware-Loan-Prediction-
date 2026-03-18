import pandas as pd
import numpy as np
import os

input_path = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\raw\External_Cibil_Dataset.csv"
output_dir = r"C:\Users\projects\Fairness-Aware-Loan-Prediction-\data\processed\feature_reduction"
output_path = os.path.join(output_dir, "External_Cibil_Dataset.csv")

df = pd.read_csv(input_path)

cols_to_drop = [
    "time_since_recent_payment",
    "time_since_first_deliquency",
    "time_since_recent_deliquency",
    "num_times_delinquent",
    "max_delinquency_level",
    "max_recent_level_of_deliq",
    "num_deliq_6mts",
    "num_deliq_12mts",
    "num_deliq_6_12mts",
    "max_deliq_6mts",
    "max_deliq_12mts",
    "num_times_30p_dpd",
    "num_times_60p_dpd",
    "num_std",
    "num_std_6mts",
    "num_std_12mts",
    "num_sub",
    "num_sub_6mts",
    "num_sub_12mts",
    "num_dbt",
    "num_dbt_6mts",
    "num_dbt_12mts",
    "num_lss",
    "num_lss_6mts",
    "num_lss_12mts",
    "recent_level_of_deliq",
    "tot_enq",
    "CC_enq",
    "CC_enq_L6m",
    "CC_enq_L12m",
    "PL_enq",
    "PL_enq_L6m",
    "PL_enq_L12m",
    "time_since_recent_enq",
    "enq_L12m",
    "enq_L6m",
    "enq_L3m",
    "pct_opened_TLs_L6m_of_L12m",
    "pct_currentBal_all_TL",
    "CC_utilization",
    "CC_Flag",
    "PL_utilization",
    "pct_PL_enq_L6m_of_L12m",
    "pct_CC_enq_L6m_of_L12m",
    "pct_CC_enq_L6m_of_ever",
    "max_unsec_exposure_inPct",
    "first_prod_enq2"
]

df_reduced = df.drop(columns=cols_to_drop)

os.makedirs(output_dir, exist_ok=True)
df_reduced.to_csv(output_path, index=False)

print(f"Saved reduced dataset to: {output_path}")
print(f"Original shape: {df.shape} → Reduced shape: {df_reduced.shape}")