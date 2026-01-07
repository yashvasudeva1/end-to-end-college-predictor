import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------------- PATHS ----------------
PROCESSED_DIR = "data/processed"
FINAL_DIR = "data/final"
MODEL_DIR = "models"   # 🔧 FIXED: was "model"

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD ALL YEAR-WISE MASTERS ----------------
files = glob(os.path.join(PROCESSED_DIR, "jossa_*_master.csv"))

if not files:
    raise FileNotFoundError("No master CSV files found in data/processed")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# ---------------- STANDARDIZE COLUMN NAMES ----------------
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

# 🔧 REMOVE DUPLICATE COLUMNS (CRITICAL FIX)
df = df.loc[:, ~df.columns.duplicated()]

# ---------------- RENAME IMPORTANT COLUMNS ----------------
df = df.rename(columns={
    "academic_program_name": "branch",
    "opening_rank": "open_rank",
    "closing_rank": "close_rank"
})

# ---------------- SELECT REQUIRED COLUMNS ----------------
required_cols = [
    "year", "round", "type", "institute", "branch",
    "quota", "seat_type", "gender",
    "open_rank", "close_rank"
]

missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required_cols]

# ---------------- CLEAN VALUES ----------------
df = df.replace(["-", "NA", "None", ""], np.nan)
df = df.dropna()

df["open_rank"] = pd.to_numeric(df["open_rank"], errors="coerce")
df["close_rank"] = pd.to_numeric(df["close_rank"], errors="coerce")

df = df.dropna()
df = df[df["close_rank"] > 0]

df["year"] = df["year"].astype(int)
df["round"] = df["round"].astype(int)

# ---------------- BASIC NUMERIC FEATURES ----------------
df["rank_gap"] = df["close_rank"] - df["open_rank"]
df["rank_mid"] = (df["open_rank"] + df["close_rank"]) / 2

# ---------------- ENCODE CATEGORICAL COLUMNS ----------------
categorical_cols = [
    "type",
    "institute",
    "branch",
    "quota",
    "seat_type",
    "gender"
]

encoders = {}

for col in categorical_cols:
    # Safety check: must be 1D
    if isinstance(df[col], pd.DataFrame):
        raise ValueError(f"Column '{col}' is not 1D. Duplicate column issue.")

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ---------------- SAVE OUTPUTS ----------------
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
df.to_csv(os.path.join(FINAL_DIR, "final_jossa_dataset.csv"), index=False)

print("✅ Saved successfully:")
print(" - data/final/final_jossa_dataset.csv")
print(" - models/encoders.pkl")
