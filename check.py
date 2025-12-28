"""
Exploratory experiment on train-test distribution alignment.
No artificial adjustment was used in the final model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from functions import *
'''df_with_preds = pd.read_excel("predicted_results.xlsx")
df_with_preds["Abs Error"] = df_with_preds["Prediction Error"].abs()
worst_samples = df_with_preds.sort_values("Abs Error", ascending=False).head(10)
print(worst_samples[["Match_Length", "Predicted Match Length", "Prediction Error"]])'''

dftr = pd.read_excel("Training.xlsx")
dfte = pd.read_excel('Test_Truncated.xlsx')

# 假设训练和测试集是 X_train, X_test，其中 Big_ID_cleaned 是特征列
# 如果是在原始 df 中，可以改成 df.loc[train_index] / df.loc[test_index]

'''

train_big_ids = set(dftr["Big ID"])
test_big_ids = set(dfte["Big ID"])

# 交集
overlap_ids = train_big_ids & test_big_ids

print(f"比赛训练集 Big ID 数量：{len(train_big_ids)}")
print(f"比赛测试集 Big ID 数量：{len(test_big_ids)}")
print(f"测试集中出现在训练集中的 Big ID 数量：{len(overlap_ids)}")
print(f"占测试集 Big ID 的比例：{len(overlap_ids) / len(test_big_ids):.2%}")




df = pd.read_excel("premodel040405.xlsx")
X = df['Big_ID']
y = df["Match_Length"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_big_ids = set(X_train)
test_big_ids = set(X_test)

# 交集
overlap_ids = train_big_ids & test_big_ids

print(f"测试训练集 Big ID 数量：{len(train_big_ids)}")
print(f"测试测试集 Big ID 数量：{len(test_big_ids)}")
print(f"测试集中出现在训练集中的 Big ID 数量：{len(overlap_ids)}")
print(f"占测试集 Big ID 的比例：{len(overlap_ids) / len(test_big_ids):.2%}")'''

completion_map = (
    dftr.dropna(subset=['Completion Date'])
    .sort_values(['Match ID 18Char', 'Completion Date'])
    .groupby('Match ID 18Char')['Completion Date']
    .apply(lambda dates: ';'.join(dates.dt.strftime('%Y-%m-%d')))
)
completion_map2 = (
    dfte.dropna(subset=['Completion Date'])
    .sort_values(['Match ID 18Char', 'Completion Date'])
    .groupby('Match ID 18Char')['Completion Date']
    .apply(lambda dates: ';'.join(dates.dt.strftime('%Y-%m-%d')))
)
dftr['TimeStamps'] = dftr['Match ID 18Char'].map(completion_map)
dftr.loc[dftr.duplicated('Match ID 18Char'), 'TimeStamps'] = None

dftr[['Cadence_Days_List', 'Cadence_Mean', 'Cadence_Std']] = dftr['TimeStamps'].apply(
    lambda x: pd.Series(parse_time_gaps(x))
)

dfte['TimeStamps'] = dfte['Match ID 18Char'].map(completion_map2)
dfte.loc[dfte.duplicated('Match ID 18Char'), 'TimeStamps'] = None

dfte[['Cadence_Days_List', 'Cadence_Mean', 'Cadence_Std']] = dfte['TimeStamps'].apply(
    lambda x: pd.Series(parse_time_gaps(x))
)


notes_combined = (
    dftr.groupby('Match ID 18Char')['Match Support Contact Notes']
    .apply(lambda notes: ' '.join(str(n) for n in notes.dropna()))
)
dftr['Match_All_Notes'] = dftr['Match ID 18Char'].map(notes_combined)
dftr.loc[dftr.duplicated('Match ID 18Char'), 'Match_All_Notes'] = None

first_occurrence_idx = dftr.drop_duplicates(subset='Match ID 18Char', keep='first').index
dftr = dftr.loc[first_occurrence_idx].copy()
dftr.reset_index(drop=True, inplace=True)

notes_combined = (
    dfte.groupby('Match ID 18Char')['Match Support Contact Notes']
    .apply(lambda notes: ' '.join(str(n) for n in notes.dropna()))
)
dfte['Match_All_Notes'] = dfte['Match ID 18Char'].map(notes_combined)
dfte.loc[dfte.duplicated('Match ID 18Char'), 'Match_All_Notes'] = None

first_occurrence_idx = dfte.drop_duplicates(subset='Match ID 18Char', keep='first').index
dfte = dfte.loc[first_occurrence_idx].copy()
dfte.reset_index(drop=True, inplace=True)

print(dftr['TimeStamps'].head())
print(dfte["TimeStamps"].head())
dftr['Contact_Count'] = dftr['TimeStamps'].apply(lambda x: len(str(x).split(';')) if pd.notna(x) else 0)
dfte['Contact_Count'] = dfte['TimeStamps'].apply(lambda x: len(str(x).split(';')) if pd.notna(x) else 0)


print(dftr['Contact_Count'].head())
mean_train = dftr['Contact_Count'].mean()
mean_test = dfte['Contact_Count'].mean()
delta = int(round(mean_train - mean_test))
print(f"Train Avg: {mean_train:.2f}, Test Avg: {mean_test:.2f}, Delta: {delta}")

def adjust_contact_count(x):
    if x < 8:
        return max(0, x-1)
    else:
        shrink = int(x * 0.74)
        noise = np.random.randint(-1, 2)  # -1, 0, or 1
        return max(0, shrink + noise)

dftr['Adjusted_Contact_Count'] = dftr['Contact_Count'].apply(adjust_contact_count)

# 先复制个新列，避免破坏原数据
#dftr['Adjusted_Contact_Count'] = dftr['Contact_Count'] - 3

# 小于 0 的值替换成 0（因为不能有负数）
dftr['Adjusted_Contact_Count'] = dftr['Adjusted_Contact_Count'].clip(lower=0)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.kdeplot(dftr['Adjusted_Contact_Count'], label='Train', color='#66B3FF', fill=True)
sns.kdeplot(dfte['Contact_Count'], label='Test', color='#FF9999', fill=True)
plt.title('Contact Count Distribution: adjusted Train vs. Test')
plt.xlabel('Number of Contacts')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

mean_train2 = dftr['Adjusted_Contact_Count'].mean()
diff = round(mean_train2 - mean_test)
print(f"Adj Train Avg: {mean_train2:.2f}, Test Avg: {mean_test:.2f}, Diff: {diff}")
