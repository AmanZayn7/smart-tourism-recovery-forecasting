import pandas as pd
from .config import CITIES

REQUIRED = ["date", "visitor_arrivals"]

def load_city_raw(city: str) -> pd.DataFrame:
    path = CITIES[city]["raw"]
    df = pd.read_csv(path)

    # Standardize HK duplicate year/month columns if present
    for cand in ("year_x", "year_y"):
        if cand in df.columns: df = df.rename(columns={cand: "year"})
    for cand in ("month_x", "month_y"):
        if cand in df.columns: df = df.rename(columns={cand: "month"})

    # Date handling
    if "date" not in df.columns:
        raise ValueError(f"{city}: missing required 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").drop_duplicates("date")
    df = df.set_index("date").asfreq("MS").reset_index()

    # Target
    if "visitor_arrivals" not in df.columns:
        raise ValueError(f"{city}: missing required 'visitor_arrivals' column.")
    df["visitor_arrivals"] = pd.to_numeric(df["visitor_arrivals"], errors="coerce")
    if df["visitor_arrivals"].isna().any():
        raise ValueError(f"{city}: NA found in visitor_arrivals after coercion.")

    # Optional exogenous → forward/back fill
    for exo in ("hotel_occupancy", "google_trends"):
        if exo in df.columns:
            df[exo] = pd.to_numeric(df[exo], errors="coerce").ffill().bfill()

    keep = [c for c in ["date","year","month","visitor_arrivals","hotel_occupancy","google_trends"] if c in df.columns]
    return df[keep]

