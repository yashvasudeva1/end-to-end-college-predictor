"""
v3: Force replacement with best generalised model.
The user explicitly wants better generalisation over raw accuracy.
The Deep-Reg_raw config has: 
  - 4.5x better generalisation (2.6x vs 11.5x ratio)
  - Only 10% higher test MAE (997 vs 903)
  - Consistent across 2024 AND 2025 test sets
"""

import os, shutil, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

df = pd.read_csv("data/final/jossa_features.csv")
if "type" in df.columns:
    df = df.drop(columns=["type"])

feature_cols = [c for c in df.columns if c not in ["open_rank", "close_rank"]]

params = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.03,
    "subsample": 0.75, "colsample_bytree": 0.7, "min_child_weight": 15,
    "reg_alpha": 1.5, "reg_lambda": 8.0, "gamma": 0.5,
    "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1,
}

# ── Train on all data for production ──
X_all = df[feature_cols]
y_close = df["close_rank"].values
y_open = df["open_rank"].values

close_model = xgb.XGBRegressor(**params)
close_model.fit(X_all, y_close, verbose=False)
open_model = xgb.XGBRegressor(**params)
open_model.fit(X_all, y_open, verbose=False)

# Sanity check
c_pred = close_model.predict(X_all).clip(min=1)
o_pred = open_model.predict(X_all).clip(min=1)
print(f"Production Close: MAE={mean_absolute_error(y_close, c_pred):,.0f}, R²={r2_score(y_close, c_pred):.4f}")
print(f"Production Open:  MAE={mean_absolute_error(y_open, o_pred):,.0f}, R²={r2_score(y_open, o_pred):.4f}")

# ── Also compute validation metrics for the server ──
# Train 2023, test 2024
train_23 = df[df["year"]==2023]
test_24 = df[df["year"]==2024]
cm1 = xgb.XGBRegressor(**params)
cm1.fit(train_23[feature_cols], train_23["close_rank"].values, verbose=False)
om1 = xgb.XGBRegressor(**params)
om1.fit(train_23[feature_cols], train_23["open_rank"].values, verbose=False)

t23_c = cm1.predict(train_23[feature_cols]).clip(min=1)
t24_c = cm1.predict(test_24[feature_cols]).clip(min=1)
t23_o = om1.predict(train_23[feature_cols]).clip(min=1)
t24_o = om1.predict(test_24[feature_cols]).clip(min=1)

train_close_mae = mean_absolute_error(train_23["close_rank"], t23_c)
test_close_mae = mean_absolute_error(test_24["close_rank"], t24_c)
train_open_mae = mean_absolute_error(train_23["open_rank"], t23_o)
test_open_mae = mean_absolute_error(test_24["open_rank"], t24_o)

print(f"\nValidation (train=2023, test=2024):")  
print(f"  Close: train={train_close_mae:,.0f}, test={test_close_mae:,.0f}, ratio={test_close_mae/train_close_mae:.2f}x")
print(f"  Open:  train={train_open_mae:,.0f}, test={test_open_mae:,.0f}, ratio={test_open_mae/train_open_mae:.2f}x")

# Train 2023+2024, test 2025
train_2324 = df[df["year"].isin([2023,2024])]
test_25 = df[df["year"]==2025]
cm2 = xgb.XGBRegressor(**params)
cm2.fit(train_2324[feature_cols], train_2324["close_rank"].values, verbose=False)
om2 = xgb.XGBRegressor(**params)
om2.fit(train_2324[feature_cols], train_2324["open_rank"].values, verbose=False)

t2324_c = cm2.predict(train_2324[feature_cols]).clip(min=1)
t25_c = cm2.predict(test_25[feature_cols]).clip(min=1)
t2324_o = om2.predict(train_2324[feature_cols]).clip(min=1)
t25_o = om2.predict(test_25[feature_cols]).clip(min=1)

v2_train_close_mae = mean_absolute_error(train_2324["close_rank"], t2324_c)
v2_test_close_mae = mean_absolute_error(test_25["close_rank"], t25_c)
v2_train_open_mae = mean_absolute_error(train_2324["open_rank"], t2324_o)
v2_test_open_mae = mean_absolute_error(test_25["open_rank"], t25_o)

print(f"\nValidation (train=2023-24, test=2025):")
print(f"  Close: train={v2_train_close_mae:,.0f}, test={v2_test_close_mae:,.0f}, ratio={v2_test_close_mae/v2_train_close_mae:.2f}x")
print(f"  Open:  train={v2_train_open_mae:,.0f}, test={v2_test_open_mae:,.0f}, ratio={v2_test_open_mae/v2_train_open_mae:.2f}x")

# ── Backup & Save ──
os.makedirs("model/backup", exist_ok=True)
for f in ["closing_rank_model.pkl", "opening_rank_model.pkl"]:
    src = os.path.join("model", f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join("model", "backup", f))

joblib.dump(close_model, "model/closing_rank_model.pkl")
joblib.dump(open_model, "model/opening_rank_model.pkl")

meta = {
    "config": "Deep-Reg",
    "params": {k: v for k, v in params.items() if k != "n_jobs"},
    "log_transform": False,
    "train_years": [2023, 2024, 2025],
    "feature_cols": feature_cols,
    "dropped_cols": ["type"],
}
with open("model/training_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)

print("\nModels saved! Old models backed up to model/backup/")
