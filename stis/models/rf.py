import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ..config import RANDOM_STATE

def fit_predict_rf(df_feats: pd.DataFrame, feat_cols, is_train, is_test):
    y_log = df_feats["log_arrivals"]
    X = df_feats[feat_cols]
    rf = RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=2, n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X[is_train], y_log[is_train])
    yhat_log_test = rf.predict(X[is_test])
    yhat_test = np.exp(yhat_log_test)
    return yhat_test, rf
