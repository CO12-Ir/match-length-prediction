"""
Training Data Truncation for Distribution Alignment

This script adjusts the training dataset to align its contact-frequency
distribution with that of the competition test set.

Background:
- In the competition setting, the test data is intentionally truncated,
  containing only a limited number of contact records per Match ID.
- Using the full, untruncated training data would introduce a systematic
  distribution mismatch between train and test.

Pipeline position:
- Input:  Training.xlsx (full training data)
- Output: Training040504.xlsx (distribution-aligned training data)

Purpose:
- Manually truncate the training data to mirror the structural constraints
  of the test set
- Ensure consistency in contact history length between training and testing
- Reduce train–test distribution shift caused by dataset construction rules

Method:
- Compute the original number of contacts per Match ID
- Apply a heuristic truncation rule to approximate the test-set cutoff
- Preserve temporal ordering and retain only the earliest contacts
- Apply truncation exclusively to the training data

Design notes:
- This step is competition-specific and not part of a generic ML pipeline
- The truncation strategy was empirically chosen to match observed test
  data characteristics
- Test data remains unchanged throughout the process

This script represents a deliberate distribution alignment step required
by the competition data design.
"""



import pandas as pd
import numpy as np


df = pd.read_excel("Training.xlsx")

def adjust_contact_count(x):
    if x < 8:
        return max(0, x-1)
    else:
        shrink = int(x * 0.73)
        noise = np.random.randint(-1, 2)
        return max(0, shrink + noise)

# 计算每个Match ID的应保留条数
to_keep = (
    df.groupby("Match ID 18Char")
      .size()
      .reset_index(name="orig_count")
)
to_keep["keep_count"] = to_keep["orig_count"].apply(adjust_contact_count)

# 把原数据和keep数量合并
df = df.merge(to_keep, on="Match ID 18Char", how="left")

# 对每个 Match ID 内部排序（由早到晚 or 晚到早都可以）
df = df.sort_values(["Match ID 18Char", "Completion Date"], ascending=True)

# 给每组加一个“序号”
df["contact_rank"] = df.groupby("Match ID 18Char").cumcount()

# 根据每组应保留数量进行筛选
df_filtered = df[df["contact_rank"] < df["keep_count"]].copy()

df_filtered.to_excel("Training040504.xlsx",index = False)
