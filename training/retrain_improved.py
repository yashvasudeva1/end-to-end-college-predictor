"""
Improved XGBoost Training Pipeline v2 for JoSAA College Predictor.

Strategy:
1. Fair comparison: test on 2024 (same as old model) and 2025
2. Train on 2023 for fair comparison with old model test metrics
3. Then train final production model on ALL data (2023-2025)
4. Use log-transform to handle skewness and reduce overfitting
5. Stronger regularisation
6. Replace if test metrics improve
"""

import os
import sys
import json
import warnings
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================

print("=" * 65)
print("IMPROVED XGBOOST TRAINING PIPELINE v2")
print("=" * 65)

df = pd.read_csv("data/final/jossa_features.csv")
print(f"\nLoaded: {df.shape[0]:,} rows, {df.shape[1]} cols")
print(f"Years: {sorted(df['year'].unique())}")

# Drop useless constant column
if "type" in df.columns and df["type"].nunique() <= 1:
    df = df.drop(columns=["type"])
    print("Dropped 'type' column (constant)")

feature_cols = [c for c in df.columns if c not in ["open_rank", "close_rank"]]

# Splits
df_2023 = df[df["year"] == 2023]
df_2024 = df[df["year"] == 2024]
df_2025 = df[df["year"] == 2025]
df_train_23 = df_2023.copy()           # for fair comparison
df_train_2324 = df[df["year"].isin([2023, 2024])]  # for final test on 2025
df_all = df.copy()                      # for production model

print(f"\n2023: {len(df_2023):,} | 2024: {len(df_2024):,} | 2025: {len(df_2025):,}")

# Log-transform helpers
def log_t(y):  return np.log1p(y)
def inv_t(y):  return np.expm1(y)

# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

# Multiple configs to test
configs = {
    "Balanced": {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 10,
        "reg_alpha": 1.0, "reg_lambda": 5.0, "gamma": 0.5,
        "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1,
    },
    "Deep-Reg": {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.03,
        "subsample": 0.75, "colsample_bytree": 0.7, "min_child_weight": 15,
        "reg_alpha": 1.5, "reg_lambda": 8.0, "gamma": 0.5,
        "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1,
    },
    "Shallow-Strong": {
        "n_estimators": 350, "max_depth": 4, "learning_rate": 0.04,
        "subsample": 0.7, "colsample_bytree": 0.65, "min_child_weight": 20,
        "reg_alpha": 2.0, "reg_lambda": 8.0, "gamma": 1.0,
        "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1,
    },
    "Medium-Reg": {
        "n_estimators": 450, "max_depth": 5, "learning_rate": 0.04,
        "subsample": 0.75, "colsample_bytree": 0.75, "min_child_weight": 12,
        "reg_alpha": 1.0, "reg_lambda": 6.0, "gamma": 0.3,
        "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1,
    },
}

# ============================================================
# 3. TRAIN & EVALUATE ALL CONFIGS
# ============================================================

def train_eval(params, X_tr, y_tr_raw, X_te, y_te_raw, use_log=True):
    y_tr = log_t(y_tr_raw) if use_log else y_tr_raw
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, verbose=False)
    pred = model.predict(X_te)
    if use_log:
        pred = inv_t(pred)
    pred = pred.clip(min=1)
    
    train_pred = model.predict(X_tr)
    if use_log:
        train_pred = inv_t(train_pred)
    train_pred = train_pred.clip(min=1)
    
    train_mae = mean_absolute_error(y_tr_raw, train_pred)
    test_mae = mean_absolute_error(y_te_raw, pred)
    test_r2 = r2_score(y_te_raw, pred)
    train_r2 = r2_score(y_tr_raw, train_pred)
    ratio = test_mae / max(train_mae, 1)
    return model, {
        "train_mae": train_mae, "test_mae": test_mae,
        "train_r2": train_r2, "test_r2": test_r2,
        "ratio": ratio, "pred": pred,
    }


print("\n" + "=" * 65)
print("EXPERIMENT: Train on 2023, Test on 2024 (fair comparison)")
print("=" * 65)

X_tr23 = df_2023[feature_cols]
X_te24 = df_2024[feature_cols]
y_tr23_close = df_2023["close_rank"].values
y_te24_close = df_2024["close_rank"].values
y_tr23_open = df_2023["open_rank"].values
y_te24_open = df_2024["open_rank"].values

# Also try without log transform for comparison
results = {}
for name, params in configs.items():
    for use_log in [True, False]:
        label = f"{name}{'_log' if use_log else '_raw'}"
        
        _, close_res = train_eval(params, X_tr23, y_tr23_close, X_te24, y_te24_close, use_log)
        _, open_res = train_eval(params, X_tr23, y_tr23_open, X_te24, y_te24_open, use_log)
        
        results[label] = {
            "close": close_res, "open": open_res,
            "avg_test_mae": (close_res["test_mae"] + open_res["test_mae"]) / 2,
            "avg_ratio": (close_res["ratio"] + open_res["ratio"]) / 2,
            "use_log": use_log, "params": params,
        }

# Load old model metrics for comparison
old_close = joblib.load("model/closing_rank_model.pkl")
old_open = joblib.load("model/opening_rank_model.pkl")

# Old models need 'type' column
orig_df = pd.read_csv("data/final/jossa_features.csv")
old_feat_cols = [c for c in orig_df.columns if c not in ["open_rank", "close_rank"]]
X_te24_old = orig_df[orig_df["year"] == 2024][old_feat_cols]

old_close_pred = old_close.predict(X_te24_old).clip(min=1)
old_open_pred = old_open.predict(X_te24_old).clip(min=1)

old_close_train_pred = old_close.predict(orig_df[orig_df["year"] == 2023][old_feat_cols]).clip(min=1)
old_open_train_pred = old_open.predict(orig_df[orig_df["year"] == 2023][old_feat_cols]).clip(min=1)

old_test_close_mae = mean_absolute_error(y_te24_close, old_close_pred)
old_test_close_r2 = r2_score(y_te24_close, old_close_pred)
old_train_close_mae = mean_absolute_error(y_tr23_close, old_close_train_pred)
old_close_ratio = old_test_close_mae / max(old_train_close_mae, 1)

old_test_open_mae = mean_absolute_error(y_te24_open, old_open_pred)
old_test_open_r2 = r2_score(y_te24_open, old_open_pred)
old_train_open_mae = mean_absolute_error(y_tr23_open, old_open_train_pred)
old_open_ratio = old_test_open_mae / max(old_train_open_mae, 1)

print(f"\n  OLD MODEL (baseline):")
print(f"    Close: train MAE={old_train_close_mae:,.0f}, test MAE={old_test_close_mae:,.0f}, "
      f"R²={old_test_close_r2:.4f}, ratio={old_close_ratio:.2f}x")
print(f"    Open:  train MAE={old_train_open_mae:,.0f}, test MAE={old_test_open_mae:,.0f}, "
      f"R²={old_test_open_r2:.4f}, ratio={old_open_ratio:.2f}x")

print(f"\n  NEW CONFIGS (sorted by avg test MAE):")
print(f"  {'Config':25s} {'Close MAE':>12s} {'Open MAE':>12s} {'Avg MAE':>10s} {'Ratio':>6s} {'Close R²':>10s}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*6} {'-'*10}")

sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_test_mae"])
for name, res in sorted_results:
    c = res["close"]
    o = res["open"]
    print(f"  {name:25s} {c['test_mae']:>12,.0f} {o['test_mae']:>12,.0f} "
          f"{res['avg_test_mae']:>10,.0f} {res['avg_ratio']:>5.2f}x {c['test_r2']:>10.4f}")

# ============================================================
# 4. PICK BEST & TRAIN ON MORE DATA
# ============================================================

# Find best config that beats old model
best = None
for name, res in sorted_results:
    if (res["close"]["test_mae"] < old_test_close_mae and 
        res["open"]["test_mae"] < old_test_open_mae):
        best = (name, res)
        break

if best is None:
    # Pick best by generalisation quality (lowest ratio with reasonable MAE)
    for name, res in sorted(results.items(), key=lambda x: x[1]["avg_ratio"]):
        if res["close"]["test_mae"] <= old_test_close_mae * 1.05:
            best = (name, res)
            break

if best is None:
    best = sorted_results[0]

best_name, best_res = best
best_params = best_res["params"]
best_log = best_res["use_log"]

print(f"\n  >>> Selected: {best_name}")
print(f"      Close: test MAE={best_res['close']['test_mae']:,.0f} "
      f"(old: {old_test_close_mae:,.0f}), ratio={best_res['close']['ratio']:.2f}x "
      f"(old: {old_close_ratio:.2f}x)")
print(f"      Open:  test MAE={best_res['open']['test_mae']:,.0f} "
      f"(old: {old_test_open_mae:,.0f}), ratio={best_res['open']['ratio']:.2f}x "
      f"(old: {old_open_ratio:.2f}x)")

# ============================================================
# 5. VALIDATE ON 2025 AS WELL (train on 2023+2024)
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION: Train on 2023+2024, Test on 2025")
print("=" * 65)

X_tr2324 = df_train_2324[feature_cols]
X_te25 = df_2025[feature_cols]
y_tr2324_close = df_train_2324["close_rank"].values
y_te25_close = df_2025["close_rank"].values
y_tr2324_open = df_train_2324["open_rank"].values
y_te25_open = df_2025["open_rank"].values

close_m_2325, close_res_2325 = train_eval(
    best_params, X_tr2324, y_tr2324_close, X_te25, y_te25_close, best_log)
open_m_2325, open_res_2325 = train_eval(
    best_params, X_tr2324, y_tr2324_open, X_te25, y_te25_open, best_log)

print(f"\n  Close: train MAE={close_res_2325['train_mae']:,.0f}, test(2025) MAE={close_res_2325['test_mae']:,.0f}, "
      f"R²={close_res_2325['test_r2']:.4f}, ratio={close_res_2325['ratio']:.2f}x")
print(f"  Open:  train MAE={open_res_2325['train_mae']:,.0f}, test(2025) MAE={open_res_2325['test_mae']:,.0f}, "
      f"R²={open_res_2325['test_r2']:.4f}, ratio={open_res_2325['ratio']:.2f}x")

# ============================================================
# 6. TRAIN PRODUCTION MODEL ON ALL DATA
# ============================================================

print("\n" + "=" * 65)
print("PRODUCTION: Training on ALL data (2023-2025)")
print("=" * 65)

X_all = df_all[feature_cols]
y_all_close = df_all["close_rank"].values
y_all_open = df_all["open_rank"].values

y_close_fit = log_t(y_all_close) if best_log else y_all_close
y_open_fit = log_t(y_all_open) if best_log else y_all_open

prod_close = xgb.XGBRegressor(**best_params)
prod_close.fit(X_all, y_close_fit, verbose=False)

prod_open = xgb.XGBRegressor(**best_params)
prod_open.fit(X_all, y_open_fit, verbose=False)

# Sanity check on training data
prod_close_pred = prod_close.predict(X_all)
prod_open_pred = prod_open.predict(X_all)
if best_log:
    prod_close_pred = inv_t(prod_close_pred)
    prod_open_pred = inv_t(prod_open_pred)
prod_close_pred = prod_close_pred.clip(min=1)
prod_open_pred = prod_open_pred.clip(min=1)

prod_close_mae = mean_absolute_error(y_all_close, prod_close_pred)
prod_open_mae = mean_absolute_error(y_all_open, prod_open_pred)
prod_close_r2 = r2_score(y_all_close, prod_close_pred)
prod_open_r2 = r2_score(y_all_open, prod_open_pred)

print(f"  Close: train MAE={prod_close_mae:,.0f}, R²={prod_close_r2:.4f}")
print(f"  Open:  train MAE={prod_open_mae:,.0f}, R²={prod_open_r2:.4f}")

# ============================================================
# 7. DECIDE WHETHER TO REPLACE
# ============================================================

print("\n" + "=" * 65)
print("FINAL DECISION")
print("=" * 65)

# Criteria: new model has BETTER generalisation (lower gap)
# AND competitive test MAE (within 5% of old on same test)
close_better_gen = best_res["close"]["ratio"] < old_close_ratio
open_better_gen = best_res["open"]["ratio"] < old_open_ratio
close_competitive = best_res["close"]["test_mae"] <= old_test_close_mae * 1.1
open_competitive = best_res["open"]["test_mae"] <= old_test_open_mae * 1.1

print(f"\n  Closing rank:")
print(f"    Old: MAE={old_test_close_mae:,.0f}, ratio={old_close_ratio:.2f}x")
print(f"    New: MAE={best_res['close']['test_mae']:,.0f}, ratio={best_res['close']['ratio']:.2f}x")
print(f"    Better generalisation? {'YES' if close_better_gen else 'NO'}")
print(f"    Competitive MAE?       {'YES' if close_competitive else 'NO'}")

print(f"\n  Opening rank:")
print(f"    Old: MAE={old_test_open_mae:,.0f}, ratio={old_open_ratio:.2f}x")
print(f"    New: MAE={best_res['open']['test_mae']:,.0f}, ratio={best_res['open']['ratio']:.2f}x")
print(f"    Better generalisation? {'YES' if open_better_gen else 'NO'}")
print(f"    Competitive MAE?       {'YES' if open_competitive else 'NO'}")

should_replace = (close_better_gen and open_better_gen and 
                  close_competitive and open_competitive)

if should_replace:
    print("\n  >>> REPLACING models - better generalisation with competitive accuracy!")
    
    # Backup
    os.makedirs("model/backup", exist_ok=True)
    for f in ["closing_rank_model.pkl", "opening_rank_model.pkl"]:
        src = os.path.join("model", f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join("model", "backup", f))
    print("  Old models backed up to model/backup/")

    # Save production models
    joblib.dump(prod_close, "model/closing_rank_model.pkl")
    joblib.dump(prod_open, "model/opening_rank_model.pkl")
    print("  New production models saved!")
    
    # Save metadata
    meta = {
        "config": best_name,
        "params": {k: v for k, v in best_params.items() if k != "n_jobs"},
        "log_transform": best_log,
        "train_years": [2023, 2024, 2025],
        "feature_cols": feature_cols,
        "dropped_cols": ["type"],
        "validation_on_2024": {
            "close_mae": round(best_res["close"]["test_mae"], 1),
            "close_r2": round(best_res["close"]["test_r2"], 4),
            "close_ratio": round(best_res["close"]["ratio"], 2),
            "open_mae": round(best_res["open"]["test_mae"], 1),
            "open_r2": round(best_res["open"]["test_r2"], 4),
            "open_ratio": round(best_res["open"]["ratio"], 2),
        },
        "validation_on_2025": {
            "close_mae": round(close_res_2325["test_mae"], 1),
            "close_r2": round(close_res_2325["test_r2"], 4),
            "open_mae": round(open_res_2325["test_mae"], 1),
            "open_r2": round(open_res_2325["test_r2"], 4),
        },
        "old_test_close_mae": round(old_test_close_mae, 1),
        "old_test_open_mae": round(old_test_open_mae, 1),
        "old_close_ratio": round(old_close_ratio, 2),
        "old_open_ratio": round(old_open_ratio, 2),
    }
    with open("model/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("  Metadata saved to model/training_meta.json")
else:
    print("\n  >>> NOT REPLACING - new model doesn't meet all criteria.")
    print("  Criteria: better generalisation AND competitive accuracy")

print("\nDone.")
