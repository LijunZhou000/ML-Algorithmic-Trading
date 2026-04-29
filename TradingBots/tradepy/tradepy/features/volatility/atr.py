import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='volatility', requires=['true_range'])
def atr(df, period=14):
    # ATR computed as rolling mean of true_range.
    atr = df['true_range'].ewm(span=period, adjust=False).mean()
    return {'atr': atr}

@feature(group='volatility', requires=['atr'])
def atr_normalized(df):
    return {'atr_norm': df['atr'] / df['close']}

@feature(group='volatility', requires=['atr'])
def volatility_zscore(df, window=20):
    mean = df['atr'].rolling(window).mean()
    std = df['atr'].rolling(window).std()
    return {'atr_z': (df['atr'] - mean) / std}