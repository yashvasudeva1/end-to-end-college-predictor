import pandas as pd
import joblib
import numpy as np

OPEN_MODEL_PATH = "model/opening_rank_model.pkl"
CLOSE_MODEL_PATH = "model/closing_rank_model.pkl"

DATA_PATH = "data/final/jossa_features.csv"

PREDICT_YEAR = 2026

open_model = joblib.load(OPEN_MODEL_PATH)
close_model = joblib.load(CLOSE_MODEL_PATH)

df = pd.read_csv(DATA_PATH)

latest_df = (
    df.sort_values("year")
      .groupby([
          "institute",
          "branch",
          "quota",
          "seat_type",
          "gender",
          "round"
      ])
      .tail(1)
      .copy()
)

latest_df["year"] = PREDICT_YEAR

feature_cols = [c for c in latest_df.columns if c not in [
    "open_rank",
    "close_rank"
]]

latest_df["pred_open_rank"] = open_model.predict(latest_df[feature_cols])
latest_df["pred_close_rank"] = close_model.predict(latest_df[feature_cols])

latest_df["pred_open_rank"] = latest_df["pred_open_rank"].clip(lower=1)
latest_df["pred_close_rank"] = latest_df["pred_close_rank"].clip(lower=1)

latest_df.to_csv("data/final/jossa_2026_predictions.csv", index=False)

print("Saved predictions to data/final/jossa_2026_predictions.csv")

def rank_to_chance(user_rank, open_rank, close_rank):
    if user_rank <= open_rank:
        return "Safe", 0.9
    if user_rank <= close_rank:
        return "Moderate", 0.6
    if user_rank <= close_rank * 1.1:
        return "Risky", 0.3
    return "Very Risky", 0.1


def predict_for_user(user_rank):
    df = pd.read_csv("data/final/jossa_2026_predictions.csv")

    chances = df.apply(
        lambda row: rank_to_chance(
            user_rank,
            row["pred_open_rank"],
            row["pred_close_rank"]
        ),
        axis=1
    )

    df["chance"] = chances.apply(lambda x: x[0])
    df["confidence"] = chances.apply(lambda x: x[1])

    return df.sort_values("confidence", ascending=False)


if __name__ == "__main__":
    user_rank = int(input("Enter your rank: "))
    result = predict_for_user(user_rank)

    print(
        result[[
            "institute",
            "branch",
            "chance",
            "confidence",
            "pred_open_rank",
            "pred_close_rank"
        ]].head(20)
    )
