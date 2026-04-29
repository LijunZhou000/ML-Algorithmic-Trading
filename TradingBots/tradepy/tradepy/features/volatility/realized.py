import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='volatility')
def historical_volatility(df, window=20):
    returns = np.log(df['close'] / df['close'].shift(1))
    return {'hv': returns.rolling(window).std() * np.sqrt(252)}

@feature(group='volatility')
def realized_volatility(df, window=20):
    # sum of squared intraday returns -> sqrt
    ret = df['log_return'] if 'log_return' in df.columns else np.log(df['close'] / df['close'].shift(1))
    return {f'realized_vol_{window}': np.sqrt((ret**2).rolling(window).sum()) * np.sqrt(252 / window)}
