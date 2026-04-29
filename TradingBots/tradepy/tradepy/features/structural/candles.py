import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='structural')
def candle_body(df):
    return {'body': (df['close'] - df['open']).abs()}

@feature(group='structural')
def wick_ratio(df):
    upper = df['high'] - df[['close','open']].max(axis=1)
    lower = df[['close','open']].min(axis=1) - df['low']
    return {'wick_ratio': (upper + lower) / (df['high'] - df['low'])}

@feature(group='structural')
def is_doji(df, threshold=0.1):
    # small body relative to candle range
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return {'is_doji': (body / rng < threshold).astype(float)}