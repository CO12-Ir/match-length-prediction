"""
Feature Engineering and Dataset Construction Script

This script constructs the final modeling datasets from preprocessed
training and test data.

Pipeline position:
- Input (train): Training040501.xlsx
- Input (test):  Test_Truncated.xlsx
- Output (train): premodel040504.xlsx + features040504.txt
- Output (test):  test040504.xlsx

Responsibilities:
1. Time-based feature construction (contact cadence, duration metrics)
2. Text preprocessing and aggregation of contact notes
3. Sentiment analysis and TF-IDF + SVD text features
4. Pairwise matching features between Big and Little participants
5. Categorical encoding and numeric feature normalization
6. Final feature selection and dataset freezing for modeling

Design notes:
- All feature logic is centralized in functions.py and imported here
  via `from functions import *` to ensure consistency across scripts.
- This script performs no model training; it only prepares datasets
  for downstream modeling and evaluation.
- The same feature pipeline is applied to both training and test data
  to guarantee feature alignment.

This script represents the core feature engineering stage of the project.
"""


from functions import *


df = pd.read_excel('Training040501.xlsx')
df2 = df
occupation_mapping = 0
county_mapping = 0
id_mapping = 0
employer_mapping = 0
program_mapping = 0
race_map = 0

# Cadence Calculation
completion_map = (
    df.dropna(subset=['Completion Date'])
    .sort_values(['Match ID 18Char', 'Completion Date'])
    .groupby('Match ID 18Char')['Completion Date']
    .apply(lambda dates: ';'.join(dates.dt.strftime('%Y-%m-%d')))
)
df['TimeStamps'] = df['Match ID 18Char'].map(completion_map)
df.loc[df.duplicated('Match ID 18Char'), 'TimeStamps'] = None

def parse_time_gaps(ts_string):
    if pd.isna(ts_string):
        return [], None, None
    try:
        dates = [datetime.strptime(t.strip(), "%Y-%m-%d") for t in ts_string.split(";")]
        if len(dates) < 2:
            return [], None, None
        deltas = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        mean_gap = sum(deltas) / len(deltas)
        std_gap = pd.Series(deltas).std()
        return deltas, mean_gap, std_gap
    except:
        return [], None, None

df[['Cadence_Days_List', 'Cadence_Mean', 'Cadence_Std']] = df['TimeStamps'].apply(
    lambda x: pd.Series(parse_time_gaps(x))
)

# ① 计算 Contact Count
df['Contact_Count'] = df['TimeStamps'].apply(lambda x: len(x.split(";")) if pd.notna(x) else 0)

# ② 计算最后联系时间距离激活日期的月份差
def months_between(start_date, end_date):
    if pd.isna(start_date) or pd.isna(end_date):
        return None
    delta_days = (end_date - start_date).days
    return delta_days / 30.44

def compute_duration_to_last_contact(row):
    ts_string = row['TimeStamps']
    activation_date = row['Match Activation Date']
    if pd.isna(ts_string) or pd.isna(activation_date):
        return None
    try:
        last_date = datetime.strptime(ts_string.split(";")[-1], "%Y-%m-%d")
        return months_between(activation_date, last_date)
    except:
        return None

df['Duration_To_Last_Contact'] = df.apply(compute_duration_to_last_contact, axis=1)


#Data Cleaning
preprocess_notes_inplace(df,df2)

#Rows Combination
notes_combined = (
    df.groupby('Match ID 18Char')['Match Support Contact Notes']
    .apply(lambda notes: ' '.join(str(n) for n in notes.dropna()))
)
df['Match_All_Notes'] = df['Match ID 18Char'].map(notes_combined)
df.loc[df.duplicated('Match ID 18Char'), 'Match_All_Notes'] = None

first_occurrence_idx = df.drop_duplicates(subset='Match ID 18Char', keep='first').index
df = df.loc[first_occurrence_idx].copy()
df.reset_index(drop=True, inplace=True)


sentiment_scores(df)

#TF-IDF
df = tfidf(df,"Rationale for Match")
df = tfidf(df,"Match_All_Notes")

#Little & Big
df,race_map = little_cl(df,race_map)
df,occupation_mapping,county_mapping,id_mapping = big_cl1(df,occupation_mapping,county_mapping,id_mapping)
df,employer_mapping,program_mapping = set_all(df,employer_mapping,program_mapping)

#Rationale Analysis & Compare
encode_occupation_and_more(df) #对Big的婚姻进行one-hot编码，对配对理由里体育|兴趣|距离|地点|语言|娱乐进行编码 （一系列婚姻状态，匹配情况：体育|兴趣|距离|地点|语言|娱乐）
compare_model(df)

#Save
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
df.to_excel("premodel040504.xlsx",index=False)
features = [
    'Cadence_Mean', 'Cadence_Std',
    'Little_Contact_Language_s_Spoken_cleaned', 'Little_Gender_cleaned', 'Little_Participant_Race_Ethnicity_cleaned',
    'Little_Age_cleaned', 'Big_Occupation_cleaned', 'Big_County_cleaned', 'Big_Age_cleaned', 'Big_ID_cleaned',
    'Big_Military_cleaned', 'Big_Level_of_Education_cleaned', 'Big_Gender_cleaned', 'Big_Employer_cleaned',
    'Program_cleaned', 'Approved_Activation_Gap',
    'Rationale_kw_sports', 'Rationale_kw_hobbies', 'Rationale_kw_places', 'Rationale_kw_entertainment',
    'Rationale_kw_language', 'Rationale_kw_distance', 'Gender_Match', 'Race_Match', 'Same_Census_Prefix5',
    'Age_Diff', 'Profile_Info_Filled', 'Rationale_kw_interests', 'Rationale_kw_outdoors', 'Rationale_kw_talkative',
    'Rationale_kw_friendly', 'Rationale_kw_active', 'Rationale_kw_movies', 'Rationale_kw_curious',
    'Rationale_kw_respectful', 'Rationale_kw_games', 'Rationale_kw_arts', 'Rationale_kw_creative',
    'Rationale_kw_energetic', 'Rationale_kw_mature', 'Rationale_kw_parks', 'Rationale_kw_fun',
    'Rationale_kw_outgoing', 'Rationale_kw_love','Notes_kw_mai','Notes_kw_callahan','Notes_kw_schreiber','Notes_kw_katie','Notes_kw_rob','Notes_kw_mary','Notes_kw_mario',
    'Activation_Year', 'Activation_Month',
    'Rationale_for_Match_SVD0', 'Rationale_for_Match_SVD1', 'Rationale_for_Match_SVD2', 'Rationale_for_Match_SVD3',
    'Rationale_for_Match_SVD4', 'Rationale_for_Match_SVD5', 'Rationale_for_Match_SVD6', 'Rationale_for_Match_SVD7',
    'Rationale_for_Match_SVD8', 'Rationale_for_Match_SVD9', 'Rationale_for_Match_SVD10', 'Rationale_for_Match_SVD11',
    'Rationale_for_Match_SVD12', 'Rationale_for_Match_SVD13', 'Rationale_for_Match_SVD14', 'Rationale_for_Match_SVD15',
    'Rationale_for_Match_SVD16', 'Rationale_for_Match_SVD17', 'Rationale_for_Match_SVD18', 'Rationale_for_Match_SVD19',
    'Match_All_Notes_SVD0', 'Match_All_Notes_SVD1', 'Match_All_Notes_SVD2', 'Match_All_Notes_SVD3',
    'Match_All_Notes_SVD4', 'Match_All_Notes_SVD5', 'Match_All_Notes_SVD6', 'Match_All_Notes_SVD7',
    'Match_All_Notes_SVD8', 'Match_All_Notes_SVD9', 'Match_All_Notes_SVD10', 'Match_All_Notes_SVD11',
    'Match_All_Notes_SVD12', 'Match_All_Notes_SVD13', 'Match_All_Notes_SVD14', 'Match_All_Notes_SVD15',
    'Match_All_Notes_SVD16', 'Match_All_Notes_SVD17', 'Match_All_Notes_SVD18', 'Match_All_Notes_SVD19',
    'Sentiment_Score','Sentiment_Score_Group','Duration_To_Last_Contact','Contact_Count'
]
with open("features040504.txt", "w", encoding="utf-8") as f:
    f.write(str(features))

df = pd.read_excel('Test_Truncated.xlsx')

completion_map = (
    df.dropna(subset=['Completion Date'])
    .sort_values(['Match ID 18Char', 'Completion Date'])
    .groupby('Match ID 18Char')['Completion Date']
    .apply(lambda dates: ';'.join(dates.dt.strftime('%Y-%m-%d')))
)
df['TimeStamps'] = df['Match ID 18Char'].map(completion_map)
df.loc[df.duplicated('Match ID 18Char'), 'TimeStamps'] = None

def parse_time_gaps(ts_string):
    if pd.isna(ts_string):
        return [], None, None
    try:
        dates = [datetime.strptime(t.strip(), "%Y-%m-%d") for t in ts_string.split(";")]
        if len(dates) < 2:
            return [], None, None
        deltas = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        mean_gap = sum(deltas) / len(deltas)
        std_gap = pd.Series(deltas).std()
        return deltas, mean_gap, std_gap
    except:
        return [], None, None

df[['Cadence_Days_List', 'Cadence_Mean', 'Cadence_Std']] = df['TimeStamps'].apply(
    lambda x: pd.Series(parse_time_gaps(x))
)

# ① 计算 Contact Count
df['Contact_Count'] = df['TimeStamps'].apply(lambda x: len(x.split(";")) if pd.notna(x) else 0)

# ② 计算最后联系时间距离激活日期的月份差
def months_between(start_date, end_date):
    if pd.isna(start_date) or pd.isna(end_date):
        return None
    delta_days = (end_date - start_date).days
    return delta_days / 30.44

def compute_duration_to_last_contact(row):
    ts_string = row['TimeStamps']
    activation_date = row['Match Activation Date']
    if pd.isna(ts_string) or pd.isna(activation_date):
        return None
    try:
        last_date = datetime.strptime(ts_string.split(";")[-1], "%Y-%m-%d")
        return months_between(activation_date, last_date)
    except:
        return None

df['Duration_To_Last_Contact'] = df.apply(compute_duration_to_last_contact, axis=1)


#Data Cleaning
preprocess_notes_inplace(df,df2)

#Rows Combination
notes_combined = (
    df.groupby('Match ID 18Char')['Match Support Contact Notes']
    .apply(lambda notes: ' '.join(str(n) for n in notes.dropna()))
)
df['Match_All_Notes'] = df['Match ID 18Char'].map(notes_combined)
df.loc[df.duplicated('Match ID 18Char'), 'Match_All_Notes'] = None

first_occurrence_idx = df.drop_duplicates(subset='Match ID 18Char', keep='first').index
df = df.loc[first_occurrence_idx].copy()
df.reset_index(drop=True, inplace=True)


sentiment_scores(df)

#TF-IDF
df = tfidf(df,"Rationale for Match")
df = tfidf(df,"Match_All_Notes")

#Little & Big
df,race_map = little_cl(df,race_map)
df,occupation_mapping,county_mapping,id_mapping = big_cl1(df,occupation_mapping,county_mapping,id_mapping)
df,employer_mapping,program_mapping = set_all(df,employer_mapping,program_mapping)

#Rationale Analysis & Compare
encode_occupation_and_more(df) #对Big的婚姻进行one-hot编码，对配对理由里体育|兴趣|距离|地点|语言|娱乐进行编码 （一系列婚姻状态，匹配情况：体育|兴趣|距离|地点|语言|娱乐）
compare_model(df)

#Save
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
df.to_excel("test040504.xlsx",index=False)