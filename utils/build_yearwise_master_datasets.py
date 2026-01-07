import pandas as pd
import os
import re
from glob import glob

RAW_BASE = "data/raw"
PROCESSED_BASE = "data/processed"

os.makedirs(PROCESSED_BASE, exist_ok=True)

# -------------------------------------------------
# Institute type classifier (CASE-SAFE & ROBUST)
# -------------------------------------------------
def get_institute_type(name):
    name = str(name).lower()

    if "indian institute of technology" in name or name.startswith("iit"):
        return "IIT"
    if "national institute of technology" in name or name.startswith("nit"):
        return "NIT"
    if "indian institute of information technology" in name or name.startswith("iiit"):
        return "IIIT"
    return "GFTI"

# -------------------------------------------------
# Iterate through year folders
# -------------------------------------------------
years = os.listdir(RAW_BASE)

for year in years:
    year_path = os.path.join(RAW_BASE, year)

    if not os.path.isdir(year_path):
        continue

    files = glob(os.path.join(year_path, "*.csv"))
    if not files:
        continue

    dfs = []

    for file in files:
        df = pd.read_csv(file)

        # Extract round number from filename
        round_match = re.search(r"round(\d+)", file)
        round_no = int(round_match.group(1)) if round_match else None

        df["round"] = round_no
        df["year"] = int(year)

        # -------------------------------------------------
        # FORCE recompute institute type (DO NOT TRUST OLD)
        # -------------------------------------------------
        if "Institute" in df.columns:
            df["institute"] = df["Institute"]
        elif "institute" not in df.columns:
            raise ValueError(f"'Institute' column missing in {file}")

        df["type"] = df["institute"].astype(str).apply(get_institute_type)

        dfs.append(df)

    master_df = pd.concat(dfs, ignore_index=True)

    output_path = os.path.join(PROCESSED_BASE, f"jossa_{year}_master.csv")
    master_df.to_csv(output_path, index=False)

    print(f"✅ Saved: {output_path}")
