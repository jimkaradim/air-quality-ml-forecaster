"""Train the PM2.5 model on the downloaded original UCI CSV."""
import argparse
import json
from pathlib import Path
import pandas as pd
from air_quality.model import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-aware 1h PM2.5 forecasting experiment")
    parser.add_argument("--data", default="data/PRSA_data_2010.1.1-2014.12.31.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    if not Path(args.data).is_file():
        parser.error(f"No CSV at {args.data}. Run: python scripts/download_data.py")
    report = run_experiment(pd.read_csv(args.data), args.output)
    print(json.dumps(report["splits"], indent=2))
    print(f"\nModel and evaluation saved in {args.output}/")
