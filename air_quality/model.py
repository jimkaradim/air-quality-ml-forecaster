"""Reproducible chronological evaluation against a persistence baseline."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .features import CATEGORICAL_COLS, NUMERIC_COLS, FEATURE_COLS, make_features, prepare_data, supervised_rows, time_split_indexes

def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=False), NUMERIC_COLS),
        ("wind_direction", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])
    return Pipeline([
        ("preprocessing", preprocess),
        ("regressor", HistGradientBoostingRegressor(max_iter=160, learning_rate=0.06, max_leaf_nodes=25,
                                                     min_samples_leaf=20, random_state=42, early_stopping=False)),
    ])

def metrics(y_true, y_pred):
    return {"MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "R2": round(float(r2_score(y_true, y_pred)), 4)}

def run_experiment(raw: pd.DataFrame, output_dir: str | Path = "artifacts") -> dict:
    clean = prepare_data(raw)
    all_features = make_features(clean)
    splits = time_split_indexes(len(clean))
    prepared = {}
    for name, sl in splits.items():
        X, y = supervised_rows(clean.iloc[sl].reset_index(drop=True), all_features.iloc[sl].reset_index(drop=True))
        if len(X) < 10:
            raise ValueError(f"Not enough usable observations in {name}: {len(X)}")
        prepared[name] = (X, y)
    X_train, y_train = prepared["train"]
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "UCI Beijing PM2.5 dataset 381", "forecast_horizon_hours": 1,
              "method": "Chronological 70/15/15 raw-row split; one-hour boundary embargo; no shuffle",
              "target_units": "micrograms per cubic meter (ug/m3)", "model": "HistGradientBoostingRegressor",
              "feature_columns": FEATURE_COLS, "splits": {}}
    prediction_rows = []
    for name, (X, y) in prepared.items():
        pred = np.clip(pipeline.predict(X), 0, None)
        baseline = X["pm2_5_now"].to_numpy()
        report["splits"][name] = {"observations": int(len(y)), "model": metrics(y, pred), "persistence_baseline": metrics(y, baseline)}
        if name == "test":
            prediction_rows = [{"actual_pm25": float(actual), "predicted_pm25": round(float(p), 3), "baseline_pm25": float(b)}
                               for actual, p, b in zip(y, pred, baseline)]
    joblib.dump({"pipeline": pipeline, "features": FEATURE_COLS, "horizon_hours": 1}, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(output / "test_predictions.csv", index=False)
    return report

def predict_latest_from_csv(raw: pd.DataFrame, model_path: str | Path = "artifacts/model.joblib") -> dict:
    clean = prepare_data(raw)
    if len(clean) < 3:
        raise ValueError("Provide at least three consecutive hourly observations")
    if clean["timestamp"].diff().iloc[-2:].ne(pd.Timedelta(hours=1)).any():
        raise ValueError("Most recent three measurements must be consecutive hours")
    if pd.isna(clean["pm2.5"].iloc[-1]):
        raise ValueError("The latest hour must include the observed PM2.5 value")
    artifact = joblib.load(model_path)
    X = make_features(clean).tail(1)
    if list(X.columns) != artifact["features"]:
        raise ValueError("Model feature schema mismatch")
    prediction = max(0.0, float(artifact["pipeline"].predict(X)[0]))
    return {"observation_timestamp": str(clean["timestamp"].iloc[-1]),
            "forecast_timestamp": str(clean["timestamp"].iloc[-1] + pd.Timedelta(hours=1)),
            "predicted_pm25_ug_m3": round(prediction, 2),
            "current_pm25_ug_m3": float(clean["pm2.5"].iloc[-1])}
