import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='derived')
def rolling_slope(df, window=10, source='close'):
    y = df[source].values.astype(float)
    n = window
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    sum_x2 = (x * x).sum()
    kernel = x[::-1]
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(y, n)
    slopes = windows @ kernel / sum_x2
    result = np.full(len(y), np.nan)
    result[n - 1:] = slopes
    return {f'slope_{window}': pd.Series(result, index=df.index)}

@feature(group='derived')
def tr_direction(df):
    return {'tr_dir': np.sign(df['true_range'].diff())}