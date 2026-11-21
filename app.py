"""Optional, local-only Streamlit exploration UI for an already-trained model."""
from pathlib import Path
import pandas as pd
import streamlit as st
from air_quality.features import prepare_data
from air_quality.model import predict_latest_from_csv

st.set_page_config(page_title="Air Quality ML Forecaster", page_icon="🌫️", layout="wide")
st.title("Air Quality · 1-Hour PM2.5 Forecast")
st.caption("Portfolio demo · not official air-quality advice or a deployed monitoring system")
st.markdown("Upload hourly observations in the [UCI Beijing PM2.5 CSV schema](https://archive.ics.uci.edu/dataset/381/beijing%2Bpm2%2B5%2Bdata). The model predicts the next hour from the most recent known conditions.")
model_file = Path("artifacts/model.joblib")
if not model_file.is_file():
    st.warning("No trained model yet. Run `python scripts/download_data.py` then `python train.py`.")
    st.stop()
file = st.file_uploader("Upload air-quality CSV", type="csv")
if file:
    try:
        raw = pd.read_csv(file)
        clean = prepare_data(raw)
        result = predict_latest_from_csv(raw, model_file)
        c1, c2 = st.columns(2)
        c1.metric("Current PM2.5 (µg/m³)", f"{result['current_pm25_ug_m3']:.1f}")
        c2.metric("Next hour forecast (µg/m³)", f"{result['predicted_pm25_ug_m3']:.1f}")
        st.caption(f"Forecast for {result['forecast_timestamp']} (dataset's local time)")
        recent = clean.tail(72).set_index("timestamp")[["pm2.5"]].rename(columns={"pm2.5": "Observed PM2.5"})
        st.line_chart(recent)
        st.json(result)
    except Exception as exc:
        st.error(f"Could not forecast: {exc}")
