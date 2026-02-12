import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def fit_predict_ridge(df_feats: pd.DataFrame, feat_cols, is_train, is_test):
    y_log = df_feats["log_arrivals"]
    X = df_feats[feat_cols]
    alphas = np.logspace(-2, 2, 20)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=alphas, cv=None))
    ])
    pipe.fit(X[is_train], y_log[is_train])
    yhat_log_test = pipe.predict(X[is_test])
    yhat_test = np.exp(yhat_log_test)
    return yhat_test, pipe.named_steps["model"].alpha_
