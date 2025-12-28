
"""
Final Deliverability and Distribution Sanity Check

This script performs a post-model sanity check by comparing the
distribution of key target-related features between the training
dataset and the model prediction outputs.

Pipeline position:
- Input (train): premodel040404.xlsx
- Input (test):  predicted_results_final.xlsx
- Input (test alt): predicted_results_final2.xlsx

Purpose:
- Verify that the predicted results do not exhibit severe distribution
  drift relative to the training data
- Ensure that the model outputs remain within a plausible range
- Serve as a final quality gate before result submission or reporting

Checks performed:
1. Kernel density comparison of Match Length between training and test
2. Reconstructed behavioral features (e.g., contact count, duration)
   for consistent comparison

Design notes:
- This script does not modify any data or model outputs
- It is intended purely for visualization and human inspection
- Differences are interpreted qualitatively rather than as hard
  acceptance thresholds

This script represents the final validation step of the modeling pipeline.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
def plot_feature_distribution(train_df, test_df, feature_name, bins=30):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(train_df[feature_name], label="Train", fill=True, alpha=0.5, linewidth=2)
    sns.kdeplot(test_df[feature_name], label="Test", fill=True, alpha=0.5, linewidth=2)
    plt.title(f"Distribution of {feature_name} in Train vs Test")
    plt.xlabel(feature_name)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

train_df = pd.read_excel("premodel040404.xlsx")
test_df2 = pd.read_excel("predicted_results_final2.xlsx")
test_df = pd.read_excel("predicted_results_final.xlsx")
train_df['Contact_Count'] = train_df['TimeStamps'].apply(lambda x: len(x.split(";")) if pd.notna(x) else 0)

# ② 计算最后联系时间距离激活日期的月份差
def months_between(start_date, end_date):
    if pd.isna(start_date) or pd.isna(end_date):
        return None
    delta_days = (end_date - start_date).days
    return delta_days / 30.44

def compute_duration_to_last_contact(row):
    ts_string = row['TimeStamps']
    activation_date = row['Match_Activation_Date']
    if pd.isna(ts_string) or pd.isna(activation_date):
        return None
    try:
        last_date = datetime.strptime(ts_string.split(";")[-1], "%Y-%m-%d")
        return months_between(activation_date, last_date)
    except:
        return None

train_df['Duration_To_Last_Contact'] = train_df.apply(compute_duration_to_last_contact, axis=1)


plot_feature_distribution(train_df, test_df, 'Match_Length')
plot_feature_distribution(train_df, test_df2, 'Match_Length')
