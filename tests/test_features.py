import pandas as pd
import pytest
from air_quality.features import prepare_data, make_features, supervised_rows, time_split_indexes

def test_next_hour_target_and_lag_are_correct(hourly_sample):
    clean = prepare_data(hourly_sample)
    f = make_features(clean)
    X, y = supervised_rows(clean, f)
    assert y.iloc[0] == pytest.approx(clean["pm2.5"].iloc[1])
    assert X["pm2_5_now"].iloc[0] == pytest.approx(clean["pm2.5"].iloc[0])
    assert X["pm2_5_lag1"].iloc[1] == pytest.approx(clean["pm2.5"].iloc[0])
    assert X["pm2_5_roll3"].iloc[2] == pytest.approx(clean["pm2.5"].iloc[:3].mean())

def test_missing_future_label_excluded(hourly_sample):
    hourly_sample.loc[5, "pm2.5"] = float("nan")
    clean = prepare_data(hourly_sample)
    X, _ = supervised_rows(clean, make_features(clean))
    assert 4 not in X.index and 5 not in X.index

def test_time_split_has_gap(hourly_sample):
    idx = time_split_indexes(len(hourly_sample))
    assert idx["train"].stop + 1 == idx["validation"].start
    assert idx["validation"].stop + 1 == idx["test"].start

def test_missing_column_rejected(hourly_sample):
    with pytest.raises(ValueError, match="Missing UCI"):
        prepare_data(hourly_sample.drop(columns=["TEMP"]))

def test_duplicate_timestamps_rejected(hourly_sample):
    hourly_sample.loc[1, "hour"] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_data(hourly_sample)
