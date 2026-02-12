# stis/forecast.py
import numpy as np
import pandas as pd
from .config import OUTLOOK_PATH
from .features import exclude_covid, add_log_feats, feature_columns
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from .config import RANDOM_STATE

def _fit_ridge_valid(dfx, feat_cols):
    """Fit Ridge on rows <= 2024 with no NaNs in features/target."""
    mask_train_year = (dfx["date"].dt.year <= 2024)
    need_cols = feat_cols + ["log_arrivals"]
    valid_mask = dfx[need_cols].notna().all(axis=1)
    is_train = mask_train_year & valid_mask

    X = dfx.loc[is_train, feat_cols]
    y = dfx.loc[is_train, "log_arrivals"]

    alphas = np.logspace(-2, 2, 20)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=alphas, cv=None))
    ])
    pipe.fit(X, y)
    return pipe

def _fit_rf_valid(dfx, feat_cols):
    """Fit RF on rows <= 2024 with no NaNs in features/target."""
    mask_train_year = (dfx["date"].dt.year <= 2024)
    need_cols = feat_cols + ["log_arrivals"]
    valid_mask = dfx[need_cols].notna().all(axis=1)
    is_train = mask_train_year & valid_mask

    X = dfx.loc[is_train, feat_cols]
    y = dfx.loc[is_train, "log_arrivals"]

    rf = RandomForestRegressor(
        n_estimators=600, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X, y)
    return rf

def _recursive_forecast_log_model(dfx, feat_cols, model, start="2025-01-01", steps=12):
    """
    Generic 12-step recursive forecast for models that predict log(arrivals).
    At each step, use the latest observed/predicted value as the new lag.
    """
    # Work on arrivals only for rolling forward
    work = dfx.loc[dfx["date"].dt.year <= 2024, ["date", "visitor_arrivals"]].copy()
    months = pd.date_range(start, periods=steps, freq="MS")
    results = []

    for dt in months:
        # Recompute features based on all known data up to the last row
        tmp = add_log_feats(work.copy())
        # Build the feature row from the last available month (one-step ahead)
        tmp_feat = tmp.iloc[[-1]][feat_cols]

        # If some features are still NaN (shouldn't happen after 2018), guard gracefully
        if tmp_feat.isna().any(axis=None):
            raise ValueError("Forecast feature row contains NaNs. Not enough history to compute lags/rolls.")

        # Predict log(arrivals) and invert
        yhat_log = float(model.predict(tmp_feat)[0])
        yhat = float(np.exp(yhat_log))
        results.append({"date": dt, "forecast": int(round(max(yhat, 0)))})

        # Append predicted point to continue recursion
        work = pd.concat([work, pd.DataFrame([{"date": dt, "visitor_arrivals": yhat}])], ignore_index=True)

    return pd.DataFrame(results)

def outlook_next12(city, df, model_code):
    """
    Produce the 12-month scenario outlook per city/model.
    - Ridge / RF: recursive forecast in log space.
    - SNaive: repeat last year's seasonality.
    """
    # Prep, features, columns
    dfx = exclude_covid(df)
    dfx = add_log_feats(dfx)
    feat_cols = feature_columns(dfx)

    if model_code == "ridge":
        model = _fit_ridge_valid(dfx, feat_cols)
        out = _recursive_forecast_log_model(dfx, feat_cols, model)

    elif model_code == "rf":
        model = _fit_rf_valid(dfx, feat_cols)
        out = _recursive_forecast_log_model(dfx, feat_cols, model)

    elif model_code == "snaive":
        # Copy 2024 month-by-month into 2025
        dfx_ms = dfx.set_index("date").asfreq("MS")
        last_year = dfx_ms.loc["2024-01-01":"2024-12-01", "visitor_arrivals"]
        if last_year.isna().any():
            raise ValueError("SNaive requires complete 2024 to copy seasonality.")
        future_idx = pd.date_range("2025-01-01", periods=12, freq="MS")
        out = pd.DataFrame({"date": future_idx, "forecast": last_year.values})

    else:
        raise ValueError("Unknown model code")

    out["model"] = model_code
    out["notes"] = "Scenario forecast; COVID years excluded in training"
    return out

def write_outlook(city, df, model_code, path):
    out = outlook_next12(city, df, model_code)
    out.to_csv(path, index=False)
    return out
