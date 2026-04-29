import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='derived', requires=['return'])
def rolling_moments(df, windows=None):
    # avoid mutable default argument; set default list inside function
    if windows is None:
        windows = [10, 20]
    out = {}
    for w in windows:
        out[f'rolling_skew_{w}'] = df['return'].rolling(w).skew()
        out[f'rolling_kurt_{w}'] = df['return'].rolling(w).kurt()
    return out

@feature(group='derived')
def percent_above_below_ma(df, ma_windows=None):
    if ma_windows is None:
        ma_windows = [20, 50]
    out = {}
    for w in ma_windows:
        ma = df['close'].rolling(w).mean()
        out[f'pct_above_ma_{w}'] = (df['close'] - ma) / ma
    return out