"""Fetch publicly licensed source CSV (not a copy embedded in the repository)."""
from __future__ import annotations
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
from io import BytesIO

ORIGINAL_ZIP = "https://archive.ics.uci.edu/static/public/381/beijing%2Bpm2%2B5%2Bdata.zip"
MIRROR_CSV = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
FILENAME = "PRSA_data_2010.1.1-2014.12.31.csv"

def download() -> Path:
    dest = Path("data") / FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already exists: {dest}")
        return dest
    try:
        with urlopen(Request(ORIGINAL_ZIP, headers={"User-Agent": "air-quality-ml-forecaster/0.1"}), timeout=40) as r:
            content = r.read()
        with ZipFile(BytesIO(content)) as z:
            members = [name for name in z.namelist() if name.endswith(".csv")]
            if not members:
                raise RuntimeError("The UCI ZIP did not contain a CSV")
            content = z.read(members[0])
        print("Downloaded official UCI dataset.")
    except Exception as err:
        print(f"UCI download unavailable ({err}); trying public mirror.")
        with urlopen(Request(MIRROR_CSV, headers={"User-Agent": "air-quality-ml-forecaster/0.1"}), timeout=40) as r:
            content = r.read()
        print("Downloaded publicly hosted CSV mirror; original source is UCI dataset 381.")
    if not content.startswith(b"No,year,month,day,hour,pm2.5"):
        raise RuntimeError("Dataset content did not match the expected column header")
    dest.write_bytes(content)
    print(f"Saved {dest} ({len(content):,} bytes)")
    return dest

if __name__ == "__main__":
    download()
