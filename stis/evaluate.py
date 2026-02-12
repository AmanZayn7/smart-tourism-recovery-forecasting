import json
import numpy as np
import pandas as pd
from math import sqrt
from .config import EVAL_PATH, METRICS_PATH
from .features import exclude_covid, add_log_feats, train_test_masks, feature_columns
from .models import fit_predict_ridge, fit_predict_rf, predict_snaive_2024

def _metrics(y_true, y_pred):
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(sqrt(np.mean((y_true - y_pred)**2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return r2, mae, rmse, mape

def run_evaluation(city, df, model_code):
    # 1) Prep + features
    dfx = exclude_covid(df)
    dfx = add_log_feats(dfx)
    is_train, is_test = train_test_masks(dfx)
    feat_cols = feature_columns(dfx)

    # 2) SPECIAL CASE: Seasonal-Naive — evaluate without ML feature mask
    if model_code == "snaive":
        dfx_ms = dfx.set_index("date").asfreq("MS")
        y_true = dfx_ms.loc["2024-01-01":"2024-12-01", "visitor_arrivals"].astype(int)
        y_pred = dfx_ms["visitor_arrivals"].shift(12).loc["2024-01-01":"2024-12-01"].fillna(0).round().astype(int)

        eval_df = pd.DataFrame({
            "date": y_true.index,
            "actual": y_true.values,
            "pred": y_pred.values,
        })
        eval_df["abs_err"] = (eval_df["actual"] - eval_df["pred"]).abs()
        eval_df["ape_pct"] = (eval_df["abs_err"]/eval_df["actual"]*100).round(2)
        eval_df["model"] = "snaive"

        r2, mae, rmse, mape = _metrics(eval_df["actual"].values, eval_df["pred"].values)

        eval_df.to_csv(EVAL_PATH(city), index=False)
        metrics_json = {
            "city": city, "model": "snaive",
            "train_period": "2017-01..2019-12 + 2023-01..2023-12",
            "test_period": "2024-01..2024-12",
            "rows_train": int((dfx["date"].dt.year <= 2024).sum()),
            "rows_test": 12,
            "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
            "model_card": {
                "target": "arrivals",
                "features": "lag-12 (same month last year)",
                "hyperparams": "none",
                "notes": "COVID years excluded; requires 2023 coverage",
            },
        }
        with open(METRICS_PATH(city), "w") as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)
        return eval_df, metrics_json

    # 3) ML models — mask rows with NaNs in lags/rolls/log_arrivals
    need_cols = feat_cols + ["log_arrivals"]
    valid_mask = dfx[need_cols].notna().all(axis=1)
    is_train = is_train & valid_mask
    is_test  = is_test  & valid_mask

    if model_code == "ridge":
        preds, alpha = fit_predict_ridge(dfx, feat_cols, is_train, is_test)
        model_card = {
            "target": "log(arrivals) → exp inverse",
            "features": "lag1,3,6,12 + roll3,6 + month dummies",
            "hyperparams": f"alpha={alpha:.2f}",
            "notes": "COVID years excluded",
        }
    elif model_code == "rf":
        preds, _ = fit_predict_rf(dfx, feat_cols, is_train, is_test)
        model_card = {
            "target": "log(arrivals) → exp inverse",
            "features": "lag1,3,6,12 + roll3,6 + month dummies",
            "hyperparams": "n_estimators=600, min_samples_leaf=2, random_state=42",
            "notes": "COVID years excluded",
        }
    else:
        raise ValueError("Unknown model code")

    # Build aligned 2024 evaluation from the same filtered rows
    y2024_mask = (dfx["date"].dt.year == 2024) & valid_mask
    eval_df = dfx.loc[y2024_mask, ["date", "visitor_arrivals"]].copy()
    eval_df["actual"] = eval_df["visitor_arrivals"].astype(int)
    eval_df["pred"]   = pd.Series(preds, index=eval_df.index).clip(lower=0).round().astype(int)
    eval_df["abs_err"] = (eval_df["actual"] - eval_df["pred"]).abs()
    eval_df["ape_pct"] = (eval_df["abs_err"]/eval_df["actual"]*100).round(2)
    eval_df["model"]   = model_code
    eval_df = eval_df[["date","actual","pred","abs_err","ape_pct","model"]]

    r2, mae, rmse, mape = _metrics(eval_df["actual"].values, eval_df["pred"].values)

    eval_df.to_csv(EVAL_PATH(city), index=False)
    metrics_json = {
        "city": city, "model": model_code,
        "train_period": "2017-01..2019-12 + 2023-01..2023-12",
        "test_period": "2024-01..2024-12",
        "rows_train": int(is_train.sum()),
        "rows_test": int(is_test.sum()),
        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
        "model_card": model_card,
    }
    with open(METRICS_PATH(city), "w") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    return eval_df, metrics_json

