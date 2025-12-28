"""
Final model training and prediction script.

Pipeline stage:
- Input: premodel040504.xlsx + features list
- Output: predicted_results_final2.xlsx

Note:
This script also contains early-stage model comparison and Optuna-based
hyperparameter tuning code, which was used to select the final LightGBM
configuration. These sections are retained for reference but are not
required for final inference.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import numpy as np
import optuna
import optuna.visualization as vis  # 备用
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor
)
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# 正确文件
df = pd.read_excel("premodel040504.xlsx")
with open("ce.txt", "r", encoding="utf-8") as f:
    features = eval(f.read())
features = [f for f in features if f in df.columns]

# 只保留数值型
features = df[features].select_dtypes(include=["number", "bool"]).columns.tolist()

# 标准化
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df_scaled[features])




# 分离 X, y，去掉 NaN
X = df_scaled[features]
y = df_scaled["Match_Length"]
df_all = pd.concat([X, y], axis=1).dropna()
X = df_all[X.columns]
y = df_all[y.name]

# 模型 + CV
model = LGBMRegressor(
    n_estimators=472, max_depth=10,
    learning_rate=0.0341, num_leaves=39,
    min_child_samples=27, random_state=42
)
modelnew = LGBMRegressor(
    n_estimators= 425, max_depth= 10, learning_rate= 0.05319236480308879, num_leaves= 48, min_child_samples= 43, random_state=42
)
model2 = CatBoostRegressor(verbose=0)

models3 = {
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "ExtraTrees": ExtraTreesRegressor()
}


def eval_model_cv(name, model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmses = []
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        y_train_cv= np.log1p(y_train_cv) 
        model.fit(X_train_cv, y_train_cv)
        preds =  np.expm1(model.predict(X_val_cv))
        rmse = np.sqrt(mean_squared_error(y_val_cv, preds))
        print(f"Fold {i+1} RMSE: {rmse:.2f}")
        rmses.append(rmse)
    print(f">>> {name} CV RMSE: {np.mean(rmses):.2f}")
    return np.mean(rmses)

'''for name, model in models3.items():
    eval_model_cv(name, model, X, y)
'''

#eval_model_cv("gbm", model, X, y)



def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 15, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': 42
    }

    model = LGBMRegressor(**params)
    # 负 RMSE（因为 optuna 默认是最大化目标）
    rmse = eval_model_cv("LGBM (Optuna)", model,X, y)
    
    return rmse 
'''
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'verbose': 0
    }

    model = CatBoostRegressor(**params)
    rmse = eval_model_cv("CatBoost (Optuna)", model, X,y, cv=5)
    return rmse'''

study = optuna.create_study(direction='minimize', study_name='my_study', storage='sqlite:///example.db', load_if_exists=True)
study.optimize(objective, n_trials=50)

print(" Best Params:", study.best_params)
print(" Best RMSE:", study.best_value)
best_params = study.best_params
final_model = LGBMRegressor(**best_params)

vis.plot_param_importances(study).show()
vis.plot_optimization_history(study).show()
fig = vis.plot_param_importances(study)
fig.write_html("param_importance.html")

fig2 = vis.plot_optimization_history(study)
fig2.write_html("optimization_history.html")

modelnew = final_model

X_train = X
y_train = y
dfte = pd.read_excel("test040504.xlsx")

dfte = dfte.copy()
dfte[features] = scaler.fit_transform(dfte[features])
X_test = dfte[features]


y_train = np.log1p(y_train) 
modelnew.fit(X_train,y_train)
preds = modelnew.predict(X_test)  

# 对 y 做过 log1p，需要反变换一下：
dfte["Pred"] = np.expm1(preds)



# 保存
dfte.to_excel("predicted_results_final2.xlsx", index=False)

model = modelnew  # 或 xgb_model, cat_model 等

# 特征名来自训练的 X
feature_names = X.columns  
importances = model.feature_importances_

# 排序整理
feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# 打印前20
print(feat_imp_df.head(20))

#  画图（前30个）
plt.figure(figsize=(10, 8))
plt.barh(feat_imp_df["Feature"][:30][::-1], feat_imp_df["Importance"][:30][::-1], color="#66B3FF")
plt.xlabel("Importance Score")
plt.title("Top 30 Feature Importances")
plt.tight_layout()
plt.show()

#  画图（后30个）
plt.figure(figsize=(10, 8))
plt.barh(
    feat_imp_df["Feature"].tail(30),  # 取最后30个特征
    feat_imp_df["Importance"].tail(30),
    color="#FF9999"
)
plt.xlabel("Importance Score")
plt.title("Lowest 30 Feature Importances")
plt.tight_layout()
plt.show()




