"""Data validation and features available at forecast time t."""
from __future__ import annotations
import pandas as pd
RAW_COLS = ["year", "month", "day", "hour", "pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
NUMERIC_COLS = ["pm2_5_now", "pm2_5_lag1", "pm2_5_lag3", "pm2_5_roll3", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "hour", "month", "dayofweek"]
CATEGORICAL_COLS = ["cbwd"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw UCI schema; preserve missing pollution values."""
    missing = sorted(set(RAW_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing UCI dataset columns: {', '.join(missing)}")
    clean = df.copy()
    for col in [c for c in RAW_COLS if c != "cbwd"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    if clean[["year", "month", "day", "hour"]].isna().any().any():
        raise ValueError("Invalid date/time columns")
    clean["timestamp"] = pd.to_datetime(clean[["year", "month", "day", "hour"]], errors="coerce")
    if clean["timestamp"].isna().any():
        raise ValueError("Invalid timestamps")
    clean = clean.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if clean["timestamp"].duplicated().any():
        raise ValueError("Duplicate hourly timestamps in the dataset")
    clean["cbwd"] = clean["cbwd"].fillna("unknown").astype(str)
    return clean

def make_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Use current and *past* measured values, never future observations."""
    out = pd.DataFrame(index=clean.index)
    out["pm2_5_now"] = clean["pm2.5"]
    previous_one = clean["pm2.5"].shift(1)
    previous_three = clean["pm2.5"].shift(3)
    out["pm2_5_lag1"] = previous_one.where(clean["timestamp"].diff().eq(pd.Timedelta(hours=1)))
    out["pm2_5_lag3"] = previous_three.where(clean["timestamp"].diff(3).eq(pd.Timedelta(hours=3)))
    out["pm2_5_roll3"] = clean["pm2.5"].rolling(3, min_periods=1).mean()
    for col in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
        out[col] = clean[col]
    out["hour"] = clean["timestamp"].dt.hour
    out["month"] = clean["timestamp"].dt.month
    out["dayofweek"] = clean["timestamp"].dt.dayofweek
    out["cbwd"] = clean["cbwd"]
    return out[FEATURE_COLS]

def supervised_rows(clean: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Predict the next hour; omit missing labels, current readings, and time gaps."""
    next_values = clean["pm2.5"].shift(-1)
    next_is_hourly = clean["timestamp"].shift(-1).sub(clean["timestamp"]).eq(pd.Timedelta(hours=1))
    mask = next_is_hourly & next_values.notna() & clean["pm2.5"].notna()
    return features.loc[mask, FEATURE_COLS], next_values.loc[mask]

def time_split_indexes(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Indices on RAW sorted rows; a one-row embargo at split boundaries."""
    if n < 60:
        raise ValueError("At least 60 hourly rows are required for time-based train/validation/test splits")
    a = int(n * train_ratio)
    b = int(n * (train_ratio + val_ratio))
    if not (2 < a < b - 2 < n - 2):
        raise ValueError("Insufficient rows for split")
    return {"train": slice(0, a - 1), "validation": slice(a, b - 1), "test": slice(b, n - 1)}
