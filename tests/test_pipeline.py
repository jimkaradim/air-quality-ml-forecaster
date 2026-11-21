import json
import math
import pandas as pd
from air_quality.model import run_experiment, predict_latest_from_csv

def test_end_to_end_with_synthetic_fixture(hourly_sample, tmp_path):
    result = run_experiment(hourly_sample, tmp_path)
    assert (tmp_path / "model.joblib").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "test_predictions.csv").is_file()
    assert result["splits"]["test"]["observations"] > 10
    assert math.isfinite(result["splits"]["test"]["model"]["MAE"])
    assert result["splits"]["test"]["persistence_baseline"]["MAE"] >= 0
    forecast = predict_latest_from_csv(hourly_sample, tmp_path / "model.joblib")
    assert math.isfinite(forecast["predicted_pm25_ug_m3"])
    assert forecast["predicted_pm25_ug_m3"] >= 0
