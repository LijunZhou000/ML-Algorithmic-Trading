import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='returns')
def percent_return_features(df, return_horizon_min, sampling_minutes, lags=None, rolling_windows=None, ):
    """Create short- and multi-horizon return features.

    - `sampling_minutes` is the minutes per sample (1 for 1-min, 30 for 30-min, etc.)
    - `return_horizon_min` is the horizon in minutes for the horizon return feature.
    """
    if lags is None:
        lags = [1, 5, 10, 20]
    if rolling_windows is None:
        rolling_windows = [5, 10, 20]
    ret = df['close'].pct_change()
    log_ret = np.log(df['close'] / df['close'].shift(1))

    # horizon shift computed from minutes and sampling rate
    horizon_shift = int(return_horizon_min / max(1, int(sampling_minutes)))

    out = {'return': ret, 'log_return': log_ret}
    if horizon_shift >= 1:
        out[f'return_horizon_{return_horizon_min}m'] = df['close'].pct_change(horizon_shift)
        out[f'log_return_horizon_{return_horizon_min}m'] = np.log(df['close'] / df['close'].shift(horizon_shift))
    for lag in lags:
        out[f'return_lag_{lag}'] = ret.shift(lag)
    for w in rolling_windows:
        out[f'return_mean_{w}'] = ret.rolling(w).mean()
        out[f'return_std_{w}'] = ret.rolling(w).std()
        out[f'return_z_{w}'] = (ret - ret.rolling(w).mean()) / ret.rolling(w).std()
    return {k: pd.Series(v, index=df.index) for k,v in out.items()}
