"""Download original UCI Beijing PM2.5 data and validate the source schema."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import pandas as pd

DATASET_FILENAME = "PRSA_data_2010.1.1-2014.12.31.csv"
SOURCES = (
    "https://archive.ics.uci.edu/static/public/381/beijing%2Bpm2%2B5%2Bdata.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/"
    "PRSA_data_2010.1.1-2014.12.31.csv",
)
EXPECTED_COLUMNS = {
    "year", "month", "day", "hour", "pm2.5",
    "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir",
}


def download() -> Path:
    dest = Path("data") / DATASET_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Dataset already exists: {dest}")
        return dest

    errors = []
    for source in SOURCES:
        try:
            with urlopen(Request(source, headers={"User-Agent": "Mozilla/5.0 air-quality-forecaster/0.1"}), timeout=50) as resp:
                payload = resp.read()

            if source.lower().endswith(".zip"):
                with ZipFile(BytesIO(payload)) as archive:
                    csv_names = [name for name in archive.namelist() if name.endswith(DATASET_FILENAME)]
                    if not csv_names:
                        raise ValueError("ZIP archive does not contain expected dataset")
                    payload = archive.read(csv_names[0])

            # Check the schema BEFORE writing any data; do not accept incompatible mirrors.
            head = pd.read_csv(BytesIO(payload), nrows=2)
            missing = EXPECTED_COLUMNS.difference(head.columns)
            if missing:
                raise ValueError(f"Source schema mismatch: {sorted(missing)}")
            if len(head) < 2:
                raise ValueError("Source contains insufficient rows")
            dest.write_bytes(payload)
            print(f"Saved official UCI dataset to {dest} ({len(payload):,} bytes)")
            return dest
        except Exception as err:
            errors.append(f"{source}: {err}")

    raise RuntimeError("Could not download the original UCI dataset. Visit "
                       "https://archive.ics.uci.edu/dataset/381/beijing%2Bpm2%2B5%2Bdata "
                       "and save the CSV manually to data/. Errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    download()
