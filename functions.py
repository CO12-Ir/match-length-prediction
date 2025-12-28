"""
Feature Engineering Utilities for Match Outcome / Duration Prediction

This module contains:
- Text preprocessing and vectorization
- Temporal / cadence feature construction
- Demographic and profile matching features
- Exploratory visualization utilities
- Sentiment and keyword-based features

Note:
Some functions were contributed by team members.
This file is a curated and organized version for pipeline integration
and reproducibility, not the original competition workspace.
"""


import pandas as pd
import numpy as np
import datetime as dt
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
from datetime import datetime
from scipy.signal import savgol_filter

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#download the excel file at the same place of this python file




# Generic helpers for encoding categorical, multilabel,
# and text-based features in a configurable way.

def vectorize_features(df, config):
    """
    Helper function to handle various types of feature vectorization.
    
    Parameters:
    - df: pandas DataFrame
    - config: a list of dictionaries, each with:
        - 'column': column name
        - 'type': one of ['label', 'onehot', 'multilabel', 'tfidf']
        - 'fillna': optional value to fill missing data
        - 'map': optional dict for label encoding
        - 'sep': for multilabel, default ';'
        - 'n_features': for tfidf, max number of features
        - 'svd_components': if applying SVD to tfidf output
    """
    all_transformed = []

    for item in config:
        col = item['column']
        col_type = item['type']
        if 'fillna' in item:
            df[col] = df[col].fillna(item['fillna'])

        if col_type == 'label':
            if 'map' in item:
                df[col] = df[col].map(item['map']).fillna(item.get('unknown_code', 0)).astype(int)
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        elif col_type == 'onehot':
            onehot = pd.get_dummies(df[col], prefix=col)
            onehot = onehot.astype(int)
            all_transformed.append(onehot)

        elif col_type == 'multilabel':
            sep = item.get('sep', ';')
            multilabel_data = df[col].fillna('').apply(lambda x: [s.strip() for s in x.split(sep)] if x else [])
            mlb = MultiLabelBinarizer()
            transformed = pd.DataFrame(mlb.fit_transform(multilabel_data), columns=[f"{col}_{c}" for c in mlb.classes_])
            all_transformed.append(transformed)

        elif col_type == 'tfidf':
            tfidf = TfidfVectorizer(max_features=item.get('n_features', 100))
            text_series = df[col].fillna('').str.lower()
            text_series = text_series.apply(lambda x: re.sub(r'[^a-z\s]', '', x))
            text_series = text_series.str.replace('bfirstname', '', regex=False).str.replace('lfirstname', '', regex=False)
            tfidf_matrix = tfidf.fit_transform(text_series)
            if 'svd_components' in item:
                svd = TruncatedSVD(n_components=item['svd_components'])
                tfidf_matrix = svd.fit_transform(tfidf_matrix)
                tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"{col}_SVD{i}" for i in range(item['svd_components'])])
            else:
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
            all_transformed.append(tfidf_df)

    final_df = pd.concat([df.reset_index(drop=True)] + [t.reset_index(drop=True) for t in all_transformed], axis=1)
    return final_df

def tfidf(df,col_name):
    tfidf = TfidfVectorizer(max_features=200)
    text_series = df[col_name].fillna('').str.lower()
    text_series = text_series.apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    text_series = text_series.str.replace('bfirstname', '', regex=False).str.replace('lfirstname', '', regex=False)
    tfidf_matrix = tfidf.fit_transform(text_series)

    svd = TruncatedSVD(n_components=20)
    tfidf_matrix = svd.fit_transform(tfidf_matrix)
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"{col_name}_SVD{i}" for i in range(20)])
    df = pd.concat([df,tfidf_df], axis=1)

    return df


# Cleaning and normalization of free-text contact notes.
# Includes stopword removal and group-based filtering.

def preprocess_notes_inplace(df,dftr, top_n_common=300):
    stop_words = set(stopwords.words("english"))
    # Step 1: Clean individual notes
    def clean_note(note):
        note = str(note).strip().lower()

        if len(note.split()) < 3:
            return ""

        if note.startswith("question:"):
            cleaned = re.sub(r"(question:|answer:|activities:|msc:|child/family:|volunteer:|see ms contact log)", "", note)
            cleaned = re.sub(r"[^a-z]", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        cleaned = re.sub(r"[^a-z]", " ", note)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = word_tokenize(cleaned)
        filtered = [word for word in tokens if word not in stop_words]
        return " ".join(filtered)

    df["Cleaned Note Text"] = df["Match Support Contact Notes"].apply(clean_note)

    # Step 2: Ensure Match Length is numeric and binned
    dftr["Match Length"] = pd.to_numeric(dftr["Match Length"], errors="coerce")
    dftr["Match Length Group"] = pd.qcut(dftr["Match Length"], q=5, labels=[1, 2, 3, 4, 5])

    # Step 3: Get top common words per group
    group_top_words = []
    for group in range(1, 6):
        notes = df[dftr["Match Length Group"] == group]["Cleaned Note Text"]
        words = " ".join(notes).split()
        top_words = [word for word, _ in Counter(words).most_common(top_n_common)]
        group_top_words.append(set(top_words))

    # Common words across all groups
    common_words = set.intersection(*group_top_words)

    # Custom stopwords
    custom_stopwords = {
        "a", "b", "bb", "bc", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "lb", "m", "mc",
        "n", "o", "summerrelated", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", 
        "l b", "bb", "bbbstc", "httpsbbbsgtcasmemcs", "sr", "ampm", "sheherhers",
        "l_first_name", "b_first_name", "she", "her", "him", "they", "he", "describe", "completed", "pc"
    }
    words_to_remove = common_words.union(custom_stopwords)

    # Step 4: Final filter
    def filter_words(text):
        return " ".join([
            word for word in text.split()
            if word not in words_to_remove and not word.startswith("https")
        ])

    df["Match Support Contact Notes"] = df["Cleaned Note Text"].apply(filter_words)
    df.drop(columns=["Cleaned Note Text"], inplace=True)
   

# Text exploration utilities.
# Used for understanding word distributions and keyword selection.
# Not required for the final training pipeline.

def hotword(df,col_title):
    '''
    The column's hotword analysis and show the top 100 words
    '''
    text_series = df[col_title].fillna('').str.lower()

    
    text_series = text_series.apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    words_to_remove = ['volunteer', 'childfamily', 'lfirstname', 'bfirstname', 'bs','yn','pg','ls','big','lb','bb','yes','no','child','little',"na","ao","pc","ne","match"]
    pattern = '|'.join(words_to_remove)

    text_series = text_series.str.replace(pattern, '', regex=True)
    
    vectorizer = CountVectorizer(stop_words='english', max_features=50)  # 加 stop_words 去除 "the", "is", 等
    X_counts = vectorizer.fit_transform(text_series)
    word_freq = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'count': X_counts.toarray().sum(axis=0)
    }).sort_values(by='count', ascending=False)


    plt.figure(figsize=(10,6))
    sns.barplot(x='count', y='word', data=word_freq.head(100))
    plt.title('Top 100 Words')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.show()


    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(word_freq['word'], word_freq['count'])))

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()

def plt_show(df,col_name,top_n):
    '''
    Display the top_n occurrences of the response
    '''
    top_values = df[col_name].value_counts().nlargest(top_n).index  
    filtered_df = df[df[col_name].isin(top_values)] 
    t = "Distribution of "+col_name +" Top" + str(top_n)
    plt.figure(figsize=(10,6))
    sns.countplot(x=col_name, data=filtered_df, palette='mako',order=top_values)
    plt.title(t)
    plt.xlabel(col_name)
    plt.ylabel('Number')
    plt.xticks(rotation=45, fontsize=8,ticks=range(0,filtered_df[col_name].nunique()), labels=filtered_df[col_name].unique())
    plt.show()

def plt_show_all(df,col_name):
    '''
    Display any responses with occurrences times
    '''
    plt_show(col_name,df[col_name].nunique())

def plt_show2(df,col_name):
    '''
    Automatically choose from countplot or histplot to display the counts
    '''
    plt.figure(figsize=(8,6))
    unique_vals = df[col_name].nunique()
    dtype = df[col_name].dtype

    if dtype == 'object' or dtype == 'category' or unique_vals < 20:
        sns.countplot(x=col_name, data=df, color='#01FB87', edgecolor='black')
    else:
        sns.histplot(df[col_name].dropna(), kde=True, bins=60, color='#01FB87', edgecolor='black')

    plt.title("Distribution of " + col_name)
    plt.xlabel(col_name)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plt_vsML(df,col_name):
    '''
    Draw a box plot to show how the match length changes through your selected column change
    '''
    mean_values = df.groupby(col_name)['Match Length'].mean()
    # 绘制箱型图
    ax = sns.boxplot(x=col_name, y='Match Length', boxprops=dict(facecolor='#102F22', color='#002F2F'),   # 深绿色
    medianprops=dict(color='#01FB87'),
    data=df)

    for i, category in enumerate(mean_values.index):
        mean_val = mean_values[category]
        ax.scatter(i, mean_val, color='limegreen', marker='^', s=50, zorder=3, label='Mean' if i == 0 else "")
    
    plt.title('Match Length vs.'+col_name)
    plt.xlabel(col_name)
    plt.ylabel('Match Length (Months)')
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.show()

def plt_vsML2(df,col_name):
    '''
    Draw a box plot to show how the match length changes through your selected column change
    '''


    # 计算每一类的平均 Match Length
    mean_values = df.groupby(col_name)['Match Length'].mean()
    median_values = df.groupby(col_name)['Match Length'].median()

    # 绘制箱型图
    ax = sns.boxplot(x=col_name, y='Match Length', boxprops=dict(facecolor='#102F22', color='#002F2F'),   # 深绿色
    medianprops=dict(color='#01FB87'),
    data=df)
    # 在图上方标注均值
    for i, category in enumerate(mean_values.index):
        mean_val = mean_values[category]
        ax.scatter(i, mean_val, color='limegreen', marker='^', s=50, zorder=3, label='Mean' if i == 0 else "")
    
    for i, category in enumerate(mean_values.index):
        mean_val = mean_values[category]
        ax.text(i + 0.2, mean_val, f'{mean_val:.1f}', ha='left', color='#01FB87', fontsize=12, fontweight='bold')

    for i, category in enumerate(median_values.index):
        median_val = median_values[category]
        ax.text(i + 0.2, median_val, f'{median_val:.1f}', ha='left', color='#01FB87', fontsize=12, fontweight='bold')
    plt.title('Match Length vs.'+col_name)
    plt.xlabel(col_name)
    plt.ylabel('Match Length (Months)')
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.show()

def plt_corr(df,col_name):
    '''
    Do a correlation analysis on the columns that have numbers so far
    '''
    corr = df.corr(numeric_only=True)
    target = col_name
    corr_sorted = corr[[target]].drop(target).sort_values(by=target, ascending=False)
    plt.figure(figsize=(9, 12))
    sns.heatmap(corr_sorted, annot=True, cmap='coolwarm', center=0, fmt=".2f",cbar_kws={
            "shrink": 1.0,   # 控制颜色条的高度
            "pad": 0.01,     # 控制颜色条和主图之间的距离（越小越靠右）
            "aspect": 20     # 控制颜色条的“宽瘦程度”
        })
    plt.title(f"Correlation with {target}")
    plt.subplots_adjust(left=0.3, right=0.98, top=0.95, bottom=0.05)
    print(corr_sorted)
    plt.savefig(col_name+" correlations.png", dpi=300) 
    plt.show() 
    return corr_sorted

def plt_corr2(df, col_name):
    '''
    Only calculate correlation between the target column and selected features.
    '''
    selected_cols = ['Gender_Match', 'Race_Match', 'Same_Census_Prefix5', 'Age_Diff', 'Rationale_kw_language']
    
    # 计算数值型相关性
    corr = df[[col_name] + selected_cols].corr(numeric_only=True)

    # 排除自己和空值，排序
    corr_sorted = corr[[col_name]].drop(index=col_name).sort_values(by=col_name, ascending=False)
    
    # 画图
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_sorted, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                cbar_kws={"shrink": 1.0, "pad": 0.01, "aspect": 20})
    
    plt.title(f"Correlation with {col_name}")
    plt.tight_layout()
    plt.savefig(col_name + "_selected_correlations.png", dpi=300)
    plt.show()

    print(corr_sorted)
    return corr_sorted

def plt_overyear(df,col_name):
    '''
    display the mean of selected column over year
    must run timechange(df) already
    '''
    df.groupby('Activation_Year')[col_name].mean().plot(kind='line', marker='o')
    plt.title(col_name+" Over Years")
    plt.xlabel("Year")
    plt.ylabel(col_name+" mean")
    plt.tight_layout()
    plt.show()

def plt_overcol(df, col_name, over_col, method='mean'):
    agg_func = {'mean': 'mean', 'median': 'median'}[method]
    df.groupby(over_col)[col_name].agg(agg_func).plot(kind='line', marker='o',color='#002F2F')
    plt.title(f"{col_name} over {over_col} ({method})")
    plt.xlabel(over_col)
    plt.ylabel(f"{col_name} ({method})")
    plt.tight_layout()
    plt.show()

def plt_overcol2(df, col_name, over_col, method='mean', smooth=True):
    agg_func = {'mean': 'mean', 'median': 'median'}[method]
    
    grouped = df.groupby(over_col)[col_name].agg(agg_func)
    x = grouped.index
    y = grouped.values

    plt.figure(figsize=(8, 6))
    
    plt.plot(x, y, 'o', color='gray', alpha=0.4, label='Original')

    if smooth and len(y) >= 5:
        # 进行平滑处理（窗口为奇数且小于数据点数）
        window = min(len(y) // 2 * 2 + 1, 21)  
        y_smooth = savgol_filter(y, window_length=window, polyorder=2)
        plt.plot(x, y_smooth, color='#01FB87', label='Smoothed')  
    else:
        plt.plot(x, y, color='#01FB87', label=method.capitalize())

    plt.title(f"{col_name} over {over_col} ({method})")
    plt.xlabel(over_col)
    plt.ylabel(f"{col_name} ({method})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plt_col_overyear(df,col_name):
    df['Activation_Year'] = pd.to_datetime(df['Match Activation Date'], errors='coerce').dt.year
    df[col_name] = df[col_name].fillna("Unknown")
    closure_counts = df.groupby(['Activation_Year', col_name]).size().unstack(fill_value=0)

    closure_percent = closure_counts.div(closure_counts.sum(axis=1), axis=0)

    closure_percent.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='tab20b')
    plt.title(col_name+"Distribution by Year (Percentage)")
    plt.xlabel("Year")
    plt.ylabel("Percentage")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    plt.tight_layout()
    plt.show()

def plt_col_overcol(df,col_name,over_col):
    df[over_col] = df[over_col].fillna("Unknown")
    df[col_name] = df[col_name].fillna("Unknown")
    closure_counts = df.groupby([over_col, col_name]).size().unstack(fill_value=0)

    closure_percent = closure_counts.div(closure_counts.sum(axis=1), axis=0)

    closure_percent.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='tab20b')
    plt.title(col_name+"Distribution by "+over_col)
    plt.xlabel(over_col)
    plt.ylabel("Percentage")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.show()

def activated_timechange(df):
    df['Activation_Date'] = pd.to_datetime(df['Match Activation Date'], errors='coerce')
    df['Activation_Year'] = df['Activation_Date'].dt.year
    df['Activation_Quarter'] = df['Activation_Date'].dt.to_period('Q')
    df['Activation_Month'] = df['Activation_Date'].dt.month

    
# Definition of success labels based on match duration,
# closure reasons, and activity status.

def is_success(df):
    '''
    Determine whether the match is successful or not, success 1 failure 0:
    1. the column 'Stage' is shown as Active;
    2. The column titled 'Match Length' has a numeric size in the top 15%;
    3. 'Closure Details' column contains “age” or “graduated” or “successful” (choose one of the three can)
    '''
    df['Closure Details'] = df['Closure Details'].fillna('').str.lower()
    df['Stage'] = df['Stage'].fillna('').str.lower()
    df['Closure Reason'] = df['Closure Reason'].fillna('').str.lower()

    length_threshold = df['Match Length'].quantile(0.85)
    failure_closure_reasons = [
        "Volunteer: Feels incompatible with child/family",
        "Volunteer: Infraction of match rules/agency policies",
        "Child/Family: Feels incompatible with volunteer",
        "Child/Family: Infraction of match rules/agency policies",
        "Volunteer: Unrealistic expectations",
        "Child/Family: Unrealistic expectations",
        "Agency: Concern with Volunteer re: child safety"
    ]

    df['Is_Successful'] = (
        (df['Stage'] == 'active') |
        (df['Match Length'] >= length_threshold) |
        (df['Closure Details'].str.contains('age|graduated|successful')) |
        (df['Closure Reason'].str.contains('graduated|successful'))
    ).astype(int)
    df.loc[df['Closure Reason'].isin(failure_closure_reasons), 'Is_Successful'] = 0

def map_closure_reason_category(reason):
    if not isinstance(reason, str):
        return 'Active'
    r = reason.lower()
    if any(x in r for x in [
        "changed school", "moved", "workplace", "service area", "family structure"
    ]):
        return "Geographical Change"
    elif "time constraint" in r or "time" in r:
        return "Time"
    elif "lost contact" in r:
        return "Lost Contact"
    elif "health" in r or "covid" in r or "deceased" in r:
        return "Health Factor"
    elif "successful" in r or "graduated" in r:
        return "Success"
    elif "incompatible" in r or "expectations" in r:
        return "Mismatch"
    elif "infraction" in r or "safety" in r:
        return "Rules Violation"
    elif "agency" in r or "partnership" in r:
        return "Agency Related"
    elif "lost interest" in r or "severity of challenges" in r:
        return "Lost Interest"
    else:
        return "Active"

def closure_category(df):
    df["Closure Category"] = df["Closure Reason"].apply(map_closure_reason_category)

# Hand-crafted keyword features extracted from rationale
# and contact notes using domain-specific hotword analysis.

def hotwordfilter1(df):
    '''
    I went ahead and did a hotword analysis of the reasons for the matches, 
    selecting the top 100 common hotwords and choosing 25 keywords among them 
    that were relevant to the specific content and could affect the model, 
    to quantify OneHot for this column.
    '''

    rationale_keywords = [
    'interests', 'outdoors',
    'talkative', 'friendly', 'active', 'movies', 'curious',
    'respectful', 'games', 'arts', 'creative', 
    'energetic', 'mature',  'parks', 'fun',
    'outgoing', 'love'
]
    for word in rationale_keywords:
        df[f'Rationale_kw_{word}'] = df['Rationale for Match'].str.lower().str.contains(fr'\b{word}\b', na=False).astype(int)

    notes_keywords = ['mai', 'callahan', 'schreiber', 'katie', 'rob','mary', 'mario']
    for word in notes_keywords:
        df[f'Notes_kw_{word}'] = df['Match_All_Notes'].str.lower().str.contains(fr'\b{word}\b', na=False).astype(int)

def hotwordfilter2(df):
    '''
    I went ahead and did a hotword analysis of the reasons for the closure, 
    selecting the top 100 common hotwords and choosing 25 keywords among them 
    that were relevant to the specific content and could affect the model, 
    to quantify OneHot for this column.
    '''

    closure_keywords = [
    'lost', 'moved', 'contact', 'closure', 'incompatible', 'constraint', 'impact',
    'agency', 'schoolsite', 'service', 'child', 'family', 'deceased',
    'covid', 'time', 'challenges', 'constraints',
    'graduated', 'successful', 'partnership', 'policies', 'health', 'safety',"age"
]
    for word in closure_keywords:
        df[f'Closure_kw_{word}'] = df['Closure Details'].str.lower().str.contains(fr'\b{word}\b', na=False).astype(int)

def one_hot_closure_reason(df):
    """
   I got it wrong!! It is the Closure Reason not Closure Details :(
    """
    df['Closure Reason'] = df['Closure Reason'].fillna("Unknown")
    closure_dummies = pd.get_dummies(df['Closure Reason'], prefix='Closure_kw', dtype=int)
    df = pd.concat([df, closure_dummies], axis=1)
    return df


# Pairwise matching features between Big and Little participants.
# Includes gender, race, age, location, and language compatibility.

def compare_gender(df):
    #This function checks if both partners belong to the same gender
    df['Gender_Match'] = (df['Big Gender'] == df['Little Gender']).astype(int)

def clean_race_string(s):
    if pd.isna(s): return []
    return [x.strip().lower() for x in re.split(r'[;,]', s) if x.strip()]

def compare_race(df):
    #This function checks if both partners belong to the same race
    df['Race_Match'] = [
    int(len(set(clean_race_string(big)) & set(clean_race_string(little))) > 0)
    for big, little in zip(df['Big Race/Ethnicity'], df['Little Participant: Race/Ethnicity'])
]

def compare_county(df):
    #This function checks to see if both participants are located in a county (and thus closer together)
    df['Same_Census_Prefix5'] = (
    df['Big Home Census Block Group'].astype(str).str[:5] ==
    df['Little Mailing Address Census Block Group'].astype(str).str[:5]
).astype(int)
    
def compare_age(df):
    #This function adds a column with the age difference between the older and younger participants
    df['Little Birthdate'] = pd.to_datetime(df['Little Birthdate'])
    df['Big Birthdate'] = pd.to_datetime(df['Big Birthdate'])
    df['Age_Diff'] = (df['Little Birthdate'] - df['Big Birthdate']).dt.days / 365.25

def compare_age_n(df):
    #This function is for Novice division age analysis
    df['Little Birthdate'] = pd.to_datetime(df['Little Birthdate'])
    df['Big Birthdate'] = pd.to_datetime(df['Big Birthdate'])
    df['Age_Diff'] = (df['Little Birthdate'] - df['Big Birthdate']).dt.days // 365.25

def compare_age_n2(df):
    #This function is for Novice division age analysis
    df['Little Birthdate'] = pd.to_datetime(df['Little Birthdate'])
    df['Big Birthdate'] = pd.to_datetime(df['Big Birthdate'])
    df['Age_Diff2'] = ((df['Little Birthdate'] - df['Big Birthdate']).dt.days // 3652.5) * 10

def clean_occupation(df):
    '''
    clean occupation column for analysis
    '''
    # 处理空值 + 提取第一个单词（不影响 NaN）
    df['Occupation_FirstWord'] = df['Big Occupation'].apply(
    lambda x: x.split()[0].replace(':', '') if isinstance(x, str) else 'Unknown'
    )

def get_language_match(big_lang, little_lang):
    if pd.isna(little_lang) or little_lang.strip() == '':
        return 1  
    if pd.isna(big_lang) or 'no preference' in big_lang.lower():
        return 1  
    return int(little_lang.strip().lower() in big_lang.lower())

def compare_language(df):
    # This function checks that the instructor's language skills meet the student's needs
    df['Language_Match'] = df.apply(
        lambda row: get_language_match(row['Big Languages'], row["Little Contact: Language(s) Spoken"]),
        axis=1
    )

def filled_check(df):
    '''
    This function intends to handle a bunch of columns in the middle.

    There are too many blank values for items in the middle section, 
    and this function counts how many are filled in each case.

    My conjectural hypothesis is that perhaps participants who carefully fill out more information 
    take the match more seriously and thus tend to extend the matching time.
    '''

    cols_to_check = [
    'Big Enrollment: Record Type', 'Big Assessment Uploaded', 'Big Acceptance Date', 'Big Car Access',
    'Big Days Acceptance to Match', 'Big Days Interview to Acceptance', 'Big Days Interview to Match',
    'Big Open to Cross-Gender Match', 'Big Re-Enroll', 'Big Contact: Preferred Communication Type',
    'Big Contact: Former Big/Little', 'Big Contact: Interest Finder - Sports',
    'Big Contact: Interest Finder - Places To Go', 'Big Contact: Interest Finder - Hobbies',
    'Big Contact: Interest Finder - Entertainment', 'Big Contact: Created Date',
    'Big Enrollment: Created Date', 'Big Contact: Volunteer Availability', 'Big Contact: Marital Status',
    'Little RTBM Date in MF', 'Little RTBM in Matchforce', 'Little Moved to RTBM in MF',
    'Little Application Received', 'Little Contact: Language(s) Spoken', 'Little Interview Date',
    'Little Acceptance Date', 'Little Contact: Interest Finder - Sports', 'Little Contact: Interest Finder - Outdoors',
    'Little Contact: Interest Finder - Arts', 'Little Contact: Interest Finder - Places To Go',
    'Little Contact: Interest Finder - Hobbies', 'Little Contact: Interest Finder - Entertainment',
    'Little Contact: Interest Finder - Other Interests', 'Little Other Interests',
    'Little Contact: Interest Finder - Career', 'Little Contact: Interest Finder - Personality',
    'Little Contact: Interest Finder - Three Wishes'
]
    df['Profile_Info_Filled'] = df[cols_to_check].notna().sum(axis=1)

def extract_interest_features(df):
    '''
    This function was originally intended to handle a bunch of columns in the middle.
    
    Included is an analysis and comparison of whether and how similar the instructor's and student's interests are, 
    as well as an analysis of the student's interests and aspirations.

    However, I'm not going to use it anymore because there are too many blanks to handle it well
    '''

    little_interest_cols = [col for col in df.columns if col.startswith("Little Contact: Interest Finder -") and any(key in col.lower() for key in ['sports', 'arts', 'places', 'hobbies', 'entertainment', 'outdoors'])]
    big_interest_cols = [col.replace("Little", "Big") for col in little_interest_cols if col.replace("Little", "Big") in df.columns]

    if little_interest_cols:
        df['Little_All_Interests'] = df[little_interest_cols].agg(
    lambda row: ';'.join([x for x in row if pd.notna(x) and x.strip()]), axis=1
    )
        df['Little_All_Interests'] = df[little_interest_cols].fillna('').agg(';'.join, axis=1)
    else:
        df['Little_All_Interests'] = ''

    if big_interest_cols:
        df['Big_All_Interests'] = df[big_interest_cols].agg(
    lambda row: ';'.join([x for x in row if pd.notna(x) and x.strip()]), axis=1
    )
    else:
        df['Big_All_Interests'] = ''

    # Jaccard + Overlap
    def jaccard_similarity(a, b):
        set_a = set([x.strip().lower() for x in a.split(';') if x.strip()])
        set_b = set([x.strip().lower() for x in b.split(';') if x.strip()])
        if not set_a and not set_b:
            return -1.0
        if not set_a or not set_b:
            return -1.0
        return len(set_a & set_b) / len(set_a | set_b)

    df['Interest_Jaccard'] = df.apply(
        lambda row: jaccard_similarity(row['Big_All_Interests'], row['Little_All_Interests']),
        axis=1
    )
    df['Interest_Overlap_Count'] = df.apply(
        lambda row: -1 if not row['Big_All_Interests'] or not row['Little_All_Interests']
        else len(set(row['Big_All_Interests'].lower().split(';')) & set(row['Little_All_Interests'].lower().split(';'))),
        axis=1
    )

    # Personality
    if 'Little Contact: Interest Finder - Personality' in df.columns:
        df['Personality_Trait_Count'] = df['Little Contact: Interest Finder - Personality'].apply(
            lambda x: -1 if pd.isna(x) else len([trait for trait in x.split(';') if trait.strip()])
        )

    # Three Wishes
    if 'Little Contact: Interest Finder - Three Wishes' in df.columns:
        df['Wish_Count'] = df['Little Contact: Interest Finder - Three Wishes'].apply(
            lambda x: -1 if pd.isna(x) else max(1, x.count('.') + x.count('\n')) if x.strip() else 0
        )

    # Other Interests
    if 'Little Other Interests' in df.columns:
        df['OtherInterest_WordCount'] = df['Little Other Interests'].apply(
            lambda x: -1 if pd.isna(x) else len(x.split())
        )

    # Career
    if 'Little Contact: Interest Finder - Career' in df.columns:
        df['Career_CharCount'] = df['Little Contact: Interest Finder - Career'].apply(
            lambda x: -1 if pd.isna(x) else len(x)
        )

    print(df[['Big_All_Interests', 'Little_All_Interests', 'Interest_Overlap_Count']].sample(10))

    df.drop(columns=['Little_All_Interests', 'Big_All_Interests'], inplace=True)

    return df

def compare(df):
    """
    Apply the full feature engineering pipeline used for training.
    Modifies the dataframe in-place.
    """

    compare_gender(df)
    compare_race(df)
    compare_county(df)
    compare_age(df)
    compare_language(df)
    filled_check(df)
    hotwordfilter1(df)
    one_hot_closure_reason(df)
    activated_timechange(df)
    is_success(df)

def compare_model(df):
    """
    numerical data of comparing results
    """

    compare_gender(df)
    compare_race(df)
    compare_county(df)
    compare_age(df)
    filled_check(df)
    hotwordfilter1(df)
    activated_timechange(df)


# Encoding categorical attributes such as occupation,
# education, employer, program, and demographics.

def big_cl_new(df):
    # === DEPRECATED ===
    # Not used in final model due to sparsity / missing data
    df['Big Age_cleaned'] = pd.to_numeric(df['Big Age'], errors='coerce')
    df['Big Age_cleaned'] = df['Big Age_cleaned'].fillna(-1)

    df['Big: Military_cleaned'] = df['Big: Military'].map({"Yes": 1, "No": 2}).fillna(0).astype(int)
    
def big_cl1(df,occupation_mapping,county_mapping,id_mapping):
    unique_occupation = df['Big Occupation'].unique()
    if occupation_mapping == 0:
        occupation_mapping = {val: idx for idx, val in enumerate(unique_occupation)}
    df['Big Occupation_cleaned'] = df['Big Occupation'].map(occupation_mapping).fillna(-1).astype(int)


    if county_mapping == 0:
        county_mapping = {county: idx for idx, county in enumerate(df['Big County'].unique())}
    df['Big County_cleaned'] = df['Big County'].map(county_mapping)
    df['Big County_cleaned'] = df['Big County_cleaned'].fillna(-1).astype(int)


    df['Big Age_cleaned'] = pd.to_numeric(df['Big Age'], errors='coerce')
    df['Big Age_cleaned'] = df['Big Age_cleaned'].fillna(-1)


    unique_id = df['Big ID'].unique()
    if id_mapping == 0:
        id_mapping = {val: idx for idx, val in enumerate(unique_id)}
    df['Big ID_cleaned'] = df['Big ID'].map(id_mapping).fillna(-1).astype(int)


    df['Big: Military_cleaned'] = df['Big: Military'].map({"Yes": 1, "No": 2}).fillna(0).astype(int)
    return df,occupation_mapping,county_mapping,id_mapping

def little_cl(df,race_map):
    
    #map laguages to unique ints
    df['Little Contact: Language(s) Spoken_cleaned'] = df['Little Contact: Language(s) Spoken'].map({'Hmong':1, 'American Sign Language':2, 'Spanish':3})
    df['Little Contact: Language(s) Spoken_cleaned'] = df['Little Contact: Language(s) Spoken_cleaned'].fillna(0).astype(int)

    #map genders to unique ints
    df['Little Gender_cleaned'] = df['Little Gender'].map({'Female':1, 'Genderqueer/Nonbinary':2, 'Trans Male':3, 'Male':4})
    df['Little Gender_cleaned'] = df['Little Gender_cleaned'].fillna(0).astype(int)

    #map race/ethnicity to unique ints using factorize//
    if race_map == 0 :
        race_map = {county: idx for idx, county in enumerate(df['Little Participant: Race/Ethnicity'].unique())}

    df['Little Participant: Race/Ethnicity_cleaned'] = df['Little Participant: Race/Ethnicity'].map(race_map).fillna(-1).astype(int)

    #convert birthdates into datetime
    df['Little Birthdate_cleaned'] =pd.to_datetime(df['Little Birthdate'])
    #subtract birthdate from 2025-02-01 and return as age
    datas_now = dt.datetime(2025, 2, 1)
    df['Little Age_cleaned'] = (datas_now - df['Little Birthdate']).dt.days
    df['Little Age_cleaned'] = df['Little Age_cleaned']//365
    return df,race_map

def get_elements(df,col):
    '''
    A function to get all the elements in a column
    col: str, the column name
    '''
    unique_values = df[col].unique()
    print("Elements: ", unique_values)
    print("# of elements: ", len(unique_values))
    print("type of elements: ", unique_values.dtype)
    print("=====================================")

def set_big_approved_date(df):
    '''
    Paul Liao
    Column Big Approved Date (I)
    Nothing to do with it
    dtype: datetime64[ns]
    '''
    return df

def set_big_level_of_education(df):
    '''
    Paul Liao
    Column Big Level of Education (J) to int
    '''
    df['Big Level of Education_cleaned'] = df['Big Level of Education'].map({'Some High School': 1, 
                                                                     'High School Graduate': 2, 
                                                                     'Some College': 3,
                                                                     'Associate Degree': 4,
                                                                     'Bachelors Degree': 5,
                                                                     'Masters Degree': 6,
                                                                     'Juris Doctorate (JD)': 7,
                                                                     'Doctor of Medicine (MD)': 8,
                                                                     'PHD': 9})
    df['Big Level of Education_cleaned'] = df['Big Level of Education_cleaned'].fillna(0).astype(int)
    return df

def set_big_languages(df):
    '''
    Paul Liao
    Column Big Languages (K) to int
    '''
    print(df['Big Languages'])
    print("=====================================")
    get_elements(df,'Big Languages')
    df['Big Languages_cleaned'] = df['Big Languages'].fillna('').apply(lambda x: [item.strip() for item in x.split(';')])
    df['Big Languages_cleaned'] = df['Big Languages_cleaned'].apply(tuple)
    mlb = MultiLabelBinarizer()
    lang_encoded = mlb.fit_transform(df['Big Languages_cleaned'])
    lang_df = pd.DataFrame(lang_encoded, columns=[f'Lang_{l}' for l in mlb.classes_])
    df = pd.concat([df.drop(columns=['Big Languages_cleaned']), lang_df], axis=1)

    return df

def set_big_gender(df):
    '''
    Paul Liao
    Column Big Gender (L) to int
    '''
    df['Big Gender_cleaned'] = df['Big Gender'].map({'Female': 1, 
                                             'Male': 2, 
                                             'Genderqueer/Nonbinary': 3,
                                             'Trans Male': 4,
                                             'Trans Female': 5,
                                             'Prefer not to say': 6})
    df['Big Gender_cleaned'] = df['Big Gender_cleaned'].fillna(0).astype(int)
    return df

def set_big_birthdate(df):
    '''
    Paul Liao
    Column Big Birthdate (M) to int
    Nothing to do with it
    dtype: datetime64[ns]
    '''
    return df

def set_big_employer(df,map):
    '''
    Paul Liao
    Column Big Employer (N) to int
    '''
    print(df['Big Employer'])
    print("=====================================")
    get_elements(df,'Big Employer')
    if map == 0:
        map = {name: idx for idx, name in enumerate(df['Big Employer'].dropna().unique())}
    df['Big Employer_cleaned'] = df['Big Employer'].map(map).fillna(-1).astype(int)
    return df, map

def set_approved_activation_gap_day(df):
    '''
    Paul Liao
    The gap days between Big Approved Date (I) and Match Activation Date (T)
    Denoted by Approved-Activation Gap
    type: timedelta64[ns] (days)
    '''
    df['Approved-Activation Gap'] = (df['Match Activation Date'] - df['Big Approved Date']).dt.days.fillna(-1).astype(int)
    #NaN to -1
    return df

def set_program(df,map):

    get_elements(df,'Program')
    if map == 0:
        map = {name: idx for idx, name in enumerate(df['Program'].dropna().unique())}
    df['Program_cleaned'] = df['Program'].map(map).fillna(-1).astype(int)
    return df, map

def set_all(df,employer_mapping,program_mapping):
    set_big_level_of_education(df)
    set_big_languages(df)
    get_elements(df,'Big Languages')
    set_big_gender(df)
    df, employer_mapping = set_big_employer(df,employer_mapping)
    df, program_mapping = set_program(df,program_mapping)
    set_approved_activation_gap_day(df)
    return df,employer_mapping,program_mapping

def encode_occupation_and_more(df):
    """
    对 Big Occupation 和 Big Marital Status 进行 one-hot 编码。
    输入：原始 DataFrame
    输出：包含新 one-hot 特征的 DataFrame（保留原始列）
    """

    
    #df['Big Occupation'] = df['Big Occupation'].fillna("Unknown")
    #occ_dummies = pd.get_dummies(df['Big Occupation'], prefix='Occ', dtype=int)

    # Big Marital Status one-hot
    df['Big Contact: Marital Status'] = df['Big Contact: Marital Status'].fillna("Unknown")
    mar_dummies = pd.get_dummies(df['Big Contact: Marital Status'], prefix='Marital', dtype=int)

    # 提取 Rationale 列并转小写
    rationale_col = 'Rationale for Match'
    df[rationale_col] = df[rationale_col].fillna("").str.lower()

    
    keywords = {
        "sports": ["sport", "sports"],
        "hobbies": ["hobby", "hobbies"],
        "places": ["place", "places"],
        "entertainment": ["entertainment"],
        "language": ["language"],
        "distance": ["distance"]
    }

    # 创建 one-hot 列
    for key, word_list in keywords.items():
        df[f'Rationale_kw_{key}'] = df[rationale_col].apply(
            lambda x: int(any(word in x for word in word_list))
        )

    
    df = pd.concat([df, mar_dummies], axis=1)

    return df


def categorize_occupation(word):
    if word in ['Business', 'Finance', 'Consultant', 'Insurance', 'Real', 'Self-Employed']:
        return 'Business/Finance'
    elif word in ['Medical', 'Personal']:
        return 'Medical/Health'
    elif word in ['Tech', 'Engineer', 'Scientist']:
        return 'Technology'
    elif word in ['Education', 'Child/Day', 'Librarian']:
        return 'Education/Childcare'
    elif word in ['Law', 'Govt', 'Military', 'Firefighter', 'Clergy']:
        return 'Law/Government'
    elif word in ['Arts,', 'Journalist/Media']:
        return 'Creative/Media'
    elif word in ['Retail', 'Service', 'Customer', 'Barber/Hairstylist']:
        return 'Retail/Service'
    elif word in ['Construction', 'Architect', 'Landscaper/Groundskeeper', 'Facilities/Maintenance', 'Laborer', 'Craftsman', 'Factory']:
        return 'Skilled Labor'
    elif word in ['Agriculture', 'Forestry']:
        return 'Agriculture/Nature'
    elif word == 'Student':
        return 'Student'
    elif word in ['Unemployed', 'Homemaker', 'Disabled', 'Retired']:
        return 'Unemployed'
    else:
        return 'Unknown'

def occupation_category(df):
    df['Occupation_Category'] = df['Occupation_FirstWord'].apply(categorize_occupation)


# Time-based behavioral features.
# Includes contact cadence, mean/variance of contact gaps,
# and activation time decomposition.

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

def cadence_cal(df):
    '''
    create TimeStamps, cadence mean, cadence std column
    '''

    # 将 Completion Date 全部按 Match ID 聚合为分号分隔的字符串，按时间排序
    completion_map = (
        df.dropna(subset=['Completion Date'])
        .sort_values(['Match ID 18Char', 'Completion Date'])
        .groupby('Match ID 18Char')['Completion Date']
        .apply(lambda dates: ';'.join(dates.dt.strftime('%Y-%m-%d')))
    )

    # 仅在首次出现该 Match ID 的行中填入时间串
    df['TimeStamps'] = df['Match ID 18Char'].map(completion_map)
    df.loc[df.duplicated('Match ID 18Char'), 'TimeStamps'] = None

    df[['Cadence_Days_List', 'Cadence_Mean', 'Cadence_Std']] = df['TimeStamps'].apply(
        lambda x: pd.Series(parse_time_gaps(x))
    )

def combine(df):
    notes_combined = (
        df.groupby('Match ID 18Char')['Match Support Contact Notes']
        .apply(lambda notes: ' ||| '.join(str(n) for n in notes.dropna()))
    )
    df['Match_All_Notes'] = df['Match ID 18Char'].map(notes_combined)
    df.loc[df.duplicated('Match ID 18Char'), 'Match_All_Notes'] = None

    # 合并后的每个 Match ID 只保留一行
    first_occurrence_idx = df.drop_duplicates(subset='Match ID 18Char', keep='first').index
    df = df.loc[first_occurrence_idx].copy()
    df.reset_index(drop=True, inplace=True)
    
def combine_notes(df):
    # 合并所有 notes
    notes_combined = (
        df.groupby('Match ID 18Char')['Match Support Contact Notes']
        .apply(lambda notes: ' ||| '.join(str(n) for n in notes.dropna()))
        .reset_index(name='Match_All_Notes')
    )

    # 获取每个 Match ID 的第一行
    first_rows = df.sort_values('Match ID 18Char').drop_duplicates('Match ID 18Char', keep='first')

    # 合并回主表（只保留每个配对的首行）
    merged = pd.merge(first_rows, notes_combined, on='Match ID 18Char', how='left')
    merged.reset_index(drop=True, inplace=True)
    return merged



# Sentiment features derived from aggregated contact notes.
# Uses VADER sentiment analyzer.

def sentiment_scores(df, mode="both"):
    """
    Adds sentiment analysis results to the dataframe.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        mode (str): One of ["score", "group", "both"].
                    "score" adds only the sentiment score column.
                    "group" adds only the sentiment group column.
                    "both" adds both columns.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    if mode not in ["score", "group", "both"]:
        raise ValueError("mode must be one of ['score', 'group', 'both']")
    
    if mode in ["score", "both"]:
        df["Sentiment_Score"] = df["Match_All_Notes"].apply(
            lambda text: analyzer.polarity_scores(text)["compound"]
        )
    
    if mode in ["group", "both"]:
        if "Sentiment_Score" not in df.columns:
            df["Sentiment_Score"] = df["Match_All_Notes"].apply(
                lambda text: analyzer.polarity_scores(text)["compound"]
            )
        df["Sentiment_Score_Group"] = df["Sentiment_Score"].apply(lambda x: 1 if x > 0.75 else 0)
    
    # Show score stats if score was added
    if mode in ["score", "both"]:
        print(df["Sentiment_Score"].describe())

