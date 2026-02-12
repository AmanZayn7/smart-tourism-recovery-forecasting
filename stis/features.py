import numpy as np
import pandas as pd

def exclude_covid(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_datetime(df["date"]).dt.year
    # Use 2017–2019 + 2023–2024 (your evaluation split)
    return df[(y.between(2017, 2019)) | (y == 2023) | (y == 2024)].copy()

def add_log_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["log_arrivals"] = np.log(out["visitor_arrivals"].clip(lower=1.0))

    # Lags
    for L in (1, 3, 6, 12):
        out[f"log_lag{L}"] = out["log_arrivals"].shift(L)

    # Rolling means — **NO extra shift** (match notebooks)
    out["log_roll3"] = out["log_arrivals"].rolling(3).mean()
    out["log_roll6"] = out["log_arrivals"].rolling(6).mean()

    # Month dummies (drop-first)
    out["month"] = out["date"].dt.month
    dums = pd.get_dummies(out["month"], prefix="m", drop_first=True)
    out = pd.concat([out, dums], axis=1)
    return out

def train_test_masks(df: pd.DataFrame):
    y = df["date"].dt.year
    is_train = (y.between(2017, 2019)) | (y == 2023)
    is_test  = (y == 2024)
    return is_train, is_test

def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith(("log_lag", "log_roll", "m_"))]
