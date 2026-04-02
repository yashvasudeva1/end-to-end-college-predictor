import pandas as pd

df = pd.read_csv("data/final_jossa_dataset.csv")

group_cols = [
    "institute",
    "branch",
    "quota",
    "seat_type",
    "gender",
    "round"
]

df = df.sort_values(["year"])

df["open_rank_t-1"] = df.groupby(group_cols)["open_rank"].shift(1)
df["close_rank_t-1"] = df.groupby(group_cols)["close_rank"].shift(1)

df["open_rank_t-2"] = df.groupby(group_cols)["open_rank"].shift(2)
df["close_rank_t-2"] = df.groupby(group_cols)["close_rank"].shift(2)

df["open_rank_mean_last_2"] = (
    df.groupby(group_cols)["open_rank"]
      .rolling(2)
      .mean()
      .reset_index(level=group_cols, drop=True)
)

df["close_rank_mean_last_2"] = (
    df.groupby(group_cols)["close_rank"]
      .rolling(2)
      .mean()
      .reset_index(level=group_cols, drop=True)
)

df["open_rank_std_last_2"] = (
    df.groupby(group_cols)["open_rank"]
      .rolling(2)
      .std()
      .reset_index(level=group_cols, drop=True)
)

df["close_rank_std_last_2"] = (
    df.groupby(group_cols)["close_rank"]
      .rolling(2)
      .std()
      .reset_index(level=group_cols, drop=True)
)

df = df.dropna()

df.to_csv("data/final_jossa_dataset_with_features.csv", index=False)
