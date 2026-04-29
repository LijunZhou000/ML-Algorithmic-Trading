import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='volatility')
def true_range(df, window=14):
    # pointwise true range (not ATR)
    tr = ta.TRANGE(df['high'], df['low'], df['close'])
    rng = df['high'] - df['low']
    rng_mean = rng.rolling(window).mean()
    return {'true_range': pd.Series(tr, index=df.index), 'range': rng, 'range_mean': rng_mean}
