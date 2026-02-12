import pandas as pd

def predict_snaive_2024(df: pd.DataFrame) -> pd.Series:
    # needs 2023 same-month for lag-12
    df = df.set_index("date").asfreq("MS")
    y = df["visitor_arrivals"]
    yhat = y.shift(12)
    return yhat.loc["2024-01-01":"2024-12-01"]

def forecast_snaive_12m(df: pd.DataFrame) -> pd.DataFrame:
    # Outlook: copy last year's seasonality
    df = df.set_index("date").asfreq("MS")
    last_year = df.loc["2024-01-01":"2024-12-01","visitor_arrivals"]
    future_idx = pd.date_range("2025-01-01", periods=12, freq="MS")
    pred = last_year.values  # repeat 2024 into 2025
    return pd.DataFrame({"date": future_idx, "forecast": pred})
