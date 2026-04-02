"""
Flask backend for JoSAA College Predictor.
Serves the static frontend and provides REST API endpoints.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ──────────────── LOAD MODELS & DATA ────────────────

open_model = joblib.load("model/opening_rank_model.pkl")
close_model = joblib.load("model/closing_rank_model.pkl")
encoders = joblib.load("model/encoders.pkl")

raw_df = pd.read_csv("data/final/final_jossa_dataset.csv")
feature_df = pd.read_csv("data/final/jossa_features.csv")

# Drop 'type' column (constant, not used by the new models)
if "type" in feature_df.columns:
    feature_df = feature_df.drop(columns=["type"])


def decode(df, encs):
    df = df.copy()
    for col in ["institute", "branch", "quota", "seat_type", "gender"]:
        if col in encs and col in df.columns:
            try:
                df[col] = encs[col].inverse_transform(df[col].astype(int))
            except Exception:
                pass
    return df


decoded_raw = decode(raw_df, encoders)
decoded_feat = decode(feature_df, encoders)


# ──────────────── STATIC ROUTES ────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ──────────────── API: OVERVIEW STATS ────────────────

@app.route("/api/stats")
def stats():
    return jsonify({
        "institutes": int(decoded_feat["institute"].nunique()),
        "branches": int(decoded_feat["branch"].nunique()),
        "years": int(decoded_raw["year"].nunique()),
        "records": len(decoded_feat),
    })


# ──────────────── API: CLOSING RANK TREND OVERVIEW ────────────────

@app.route("/api/overview-trend")
def overview_trend():
    trend = (
        decoded_raw.groupby("year")["close_rank"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    return jsonify(trend.to_dict(orient="records"))


# ──────────────── API: DATA ANALYSIS ────────────────

@app.route("/api/analysis/institute-stability")
def institute_stability():
    vol = (
        decoded_raw.groupby(["institute", "year"])["close_rank"]
        .median()
        .reset_index()
        .groupby("institute")["close_rank"]
        .std()
        .dropna()
        .sort_values()
        .head(10)
        .reset_index(name="std_dev")
    )
    return jsonify(vol.to_dict(orient="records"))


@app.route("/api/analysis/branch-volatility")
def branch_volatility():
    vol = (
        decoded_raw.groupby(["branch", "year"])["close_rank"]
        .median()
        .reset_index()
        .groupby("branch")["close_rank"]
        .std()
        .dropna()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="std_dev")
    )
    return jsonify(vol.to_dict(orient="records"))


@app.route("/api/analysis/year-trend")
def year_trend():
    trend = (
        decoded_raw.groupby("year")["close_rank"]
        .median()
        .reset_index()
    )
    return jsonify(trend.to_dict(orient="records"))


@app.route("/api/analysis/round-trend")
def round_trend():
    trend = (
        decoded_raw.groupby("round")["close_rank"]
        .median()
        .reset_index()
    )
    return jsonify(trend.to_dict(orient="records"))


@app.route("/api/analysis/competitive-institutes")
def competitive_institutes():
    comp = (
        decoded_raw.groupby("institute")["close_rank"]
        .median()
        .sort_values()
        .head(10)
        .reset_index()
    )
    return jsonify(comp.to_dict(orient="records"))


@app.route("/api/analysis/competitive-branches")
def competitive_branches():
    comp = (
        decoded_raw.groupby("branch")["close_rank"]
        .median()
        .sort_values()
        .head(10)
        .reset_index()
    )
    return jsonify(comp.to_dict(orient="records"))


# ──────────────── API: CLOSING RANK TRENDS (INTERACTIVE) ────────────────

@app.route("/api/institutes")
def list_institutes():
    insts = sorted(decoded_feat["institute"].unique().tolist())
    return jsonify(insts)


@app.route("/api/branches")
def list_branches():
    inst = request.args.get("institute", "")
    branches = sorted(
        decoded_feat[decoded_feat["institute"] == inst]["branch"].unique().tolist()
    )
    return jsonify(branches)


@app.route("/api/closing-trend")
def closing_trend():
    inst = request.args.get("institute", "")
    branch = request.args.get("branch", "")

    subset = decoded_raw[
        (decoded_raw["institute"] == inst) & (decoded_raw["branch"] == branch)
    ]

    if subset.empty:
        return jsonify([])

    agg = (
        subset.groupby(["year", "round"], as_index=False)
        .agg({"close_rank": "max"})
        .sort_values(["year", "round"])
    )

    return jsonify(agg.to_dict(orient="records"))


# ──────────────── API: PREDICT ────────────────

# Pre-compute test MAE for error range display
# New model: train on 2023+2024, validate on 2025
_test_df = feature_df[feature_df["year"] == 2025]
_test_feat_cols = [c for c in feature_df.columns if c not in ["open_rank", "close_rank"]]
if len(_test_df) > 0:
    _test_close_pred = close_model.predict(_test_df[_test_feat_cols]).clip(min=1)
    _test_open_pred = open_model.predict(_test_df[_test_feat_cols]).clip(min=1)
    CLOSE_MAE = float(mean_absolute_error(_test_df["close_rank"], _test_close_pred))
    OPEN_MAE = float(mean_absolute_error(_test_df["open_rank"], _test_open_pred))
else:
    CLOSE_MAE = 0
    OPEN_MAE = 0


def rank_to_chance(user_rank, open_r, close_r):
    if user_rank <= open_r:
        return "Safe", 0.9
    if user_rank <= close_r:
        return "Moderate", 0.6
    if user_rank <= close_r * 1.1:
        return "Risky", 0.3
    return "Very Risky", 0.1


@app.route("/api/predict")
def predict():
    try:
        user_rank = int(request.args.get("rank", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid rank"}), 400

    if user_rank < 1:
        return jsonify({"error": "Rank must be >= 1"}), 400

    latest = (
        feature_df.sort_values("year")
        .groupby(["institute", "branch", "quota", "seat_type", "gender", "round"])
        .tail(1)
        .copy()
    )
    latest["year"] = 2026

    feat_cols = [c for c in latest.columns if c not in ["open_rank", "close_rank"]]

    latest["pred_open"] = open_model.predict(latest[feat_cols]).clip(min=1)
    latest["pred_close"] = close_model.predict(latest[feat_cols]).clip(min=1)

    chances = latest.apply(
        lambda r: rank_to_chance(user_rank, r["pred_open"], r["pred_close"]),
        axis=1,
    )
    latest["chance"] = chances.apply(lambda x: x[0])
    latest["confidence"] = chances.apply(lambda x: x[1])

    latest = decode(latest, encoders)

    # Keep round - don't deduplicate across rounds
    result = latest.sort_values(["confidence", "round"], ascending=[False, True])

    out = result[
        ["institute", "branch", "round", "quota", "seat_type", "gender",
         "pred_open", "pred_close", "chance", "confidence"]
    ].copy()

    out["pred_open"] = out["pred_open"].round(0).astype(int)
    out["pred_close"] = out["pred_close"].round(0).astype(int)
    out["round"] = out["round"].astype(int)

    # Add error range columns (based on test-set MAE)
    out["close_low"] = (out["pred_close"] - CLOSE_MAE).clip(lower=1).round(0).astype(int)
    out["close_high"] = (out["pred_close"] + CLOSE_MAE).round(0).astype(int)
    out["open_low"] = (out["pred_open"] - OPEN_MAE).clip(lower=1).round(0).astype(int)
    out["open_high"] = (out["pred_open"] + OPEN_MAE).round(0).astype(int)

    return jsonify({
        "results": out.to_dict(orient="records"),
        "close_mae": round(CLOSE_MAE),
        "open_mae": round(OPEN_MAE),
    })


# ──────────────── API: MODEL PERFORMANCE ────────────────

@app.route("/api/model-performance")
def model_performance():
    feat_cols = [c for c in feature_df.columns if c not in ["open_rank", "close_rank"]]

    # Train / Test split (matching new model validation)
    train_df = feature_df[feature_df["year"].isin([2023, 2024])]
    test_df = feature_df[feature_df["year"] == 2025]

    def metrics(y_true, y_pred):
        return {
            "mae": round(mean_absolute_error(y_true, y_pred), 1),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
            "r2": round(r2_score(y_true, y_pred), 4),
        }

    # ── Train metrics ──
    X_train = train_df[feat_cols]
    train_open_pred = open_model.predict(X_train)
    train_close_pred = close_model.predict(X_train)

    # ── Test metrics ──
    X_test = test_df[feat_cols]
    test_open_pred = open_model.predict(X_test)
    test_close_pred = close_model.predict(X_test)

    # ── Overall metrics (all data) ──
    X_all = feature_df[feat_cols]
    all_open_pred = open_model.predict(X_all)
    all_close_pred = close_model.predict(X_all)

    # Scatter - test set only (honest view of generalisation)
    rng = np.random.RandomState(42)
    n_scatter = min(5000, len(test_df))
    idx = rng.choice(len(test_df), n_scatter, replace=False)

    scatter = [
        {
            "actual": round(float(test_df["close_rank"].iloc[i]), 1),
            "predicted": round(float(test_close_pred[i]), 1),
        }
        for i in idx
    ]

    # Error histogram - test set
    test_errors = test_close_pred - test_df["close_rank"].values
    counts, edges = np.histogram(test_errors, bins=50)
    histogram = [
        {
            "bin_start": round(float(edges[i]), 1),
            "bin_end": round(float(edges[i + 1]), 1),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    return jsonify({
        "train": {
            "opening": metrics(train_df["open_rank"], train_open_pred),
            "closing": metrics(train_df["close_rank"], train_close_pred),
            "size": len(train_df),
            "years": "2023-2024",
        },
        "test": {
            "opening": metrics(test_df["open_rank"], test_open_pred),
            "closing": metrics(test_df["close_rank"], test_close_pred),
            "size": len(test_df),
            "years": "2025",
        },
        "overall": {
            "opening": metrics(feature_df["open_rank"], all_open_pred),
            "closing": metrics(feature_df["close_rank"], all_close_pred),
        },
        "scatter": scatter,
        "histogram": histogram,
    })


# ──────────────── RUN ────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
