import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def hourly_sample():
    """Synthetic fixture for code tests ONLY; no real-data performance claims."""
    rng = np.random.default_rng(123)
    n = 300
    dt = pd.date_range("2023-01-01", periods=n, freq="h")
    base = 65 + 10 * np.sin(np.arange(n) / 16) + rng.normal(0, 3, n)
    return pd.DataFrame({"year": dt.year, "month": dt.month, "day": dt.day, "hour": dt.hour,
                         "pm2.5": base, "DEWP": rng.normal(4, 3, n), "TEMP": rng.normal(17, 4, n),
                         "PRES": rng.normal(1010, 6, n), "cbwd": rng.choice(["NW", "NE", "SE", "cv"], n),
                         "Iws": rng.uniform(0, 3, n), "Is": np.zeros(n), "Ir": np.zeros(n)})
