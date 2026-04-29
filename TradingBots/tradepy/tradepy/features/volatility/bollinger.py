import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='volatility')
def bollinger_bands(df, n=20, num_std_dev=2):
    middle = df['close'].rolling(n).mean()
    std = df['close'].rolling(n).std()
    return {
        'bb_middle': middle,
        'bb_std': std,
        'bb_upper': middle + num_std_dev * std,
        'bb_lower': middle - num_std_dev * std,
    }
    
@feature(group='volatility', requires=['bb_upper', 'bb_lower', 'bb_middle'])
def bollinger_extra(df):
    # requires bb_upper, bb_lower, bb_middle
    out = {}
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns:
        out['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        out['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    return out