"""Use historical hourly CSV observations to predict PM2.5 one hour ahead."""
import argparse
import json
from pathlib import Path
import pandas as pd
from air_quality.model import predict_latest_from_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict PM2.5 at the next hour")
    parser.add_argument("--data", default="data/PRSA_data_2010.1.1-2014.12.31.csv")
    parser.add_argument("--model", default="artifacts/model.joblib")
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error("Model not found. Run python train.py first.")
    print(json.dumps(predict_latest_from_csv(pd.read_csv(args.data), args.model), indent=2))
