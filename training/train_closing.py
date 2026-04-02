import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/final/jossa_features.csv")

train_df = df[df["year"] <= 2023]
val_df = df[df["year"] == 2024]

feature_cols = [c for c in df.columns if c not in [
    "open_rank",
    "close_rank"
]]

X_train = train_df[feature_cols]
y_train = train_df["close_rank"]

X_val = val_df[feature_cols]
y_val = val_df["close_rank"]

model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_val)
print("Closing Rank MAE:", mean_absolute_error(y_val, pred))
print("Closing Rank R2:", r2_score(y_val, pred))

joblib.dump(model, "model/closing_rank_model.pkl")
