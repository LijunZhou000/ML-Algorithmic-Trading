import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='momentum')
def rsi(df, n=14):
    return {'rsi': pd.Series(ta.RSI(df['close'], timeperiod=n), index=df.index)}

@feature(group='momentum')
def stoch(df, n=14, slowk_period=3, slowd_period=3, slowk_matype=0, slowd_matype=0):
    slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=n, slowk_period=slowk_period, slowd_period=slowd_period, slowk_matype=slowk_matype, slowd_matype=slowd_matype)
    return {'slowk': pd.Series(slowk, index=df.index), 'slowd': pd.Series(slowd, index=df.index)}

@feature(group='momentum')
def cmo(df, window=14):
    diff = df['close'].diff()
    up = diff.clip(lower=0).rolling(window).sum()
    down = -diff.clip(upper=0).rolling(window).sum()
    return {'cmo': 100 * (up - down) / (up + down)}

@feature(group='momentum')
def williams_r(df, window=14):
    highest = df['high'].rolling(window).max()
    lowest = df['low'].rolling(window).min()
    return {'williams_r': (highest - df['close']) / (highest - lowest) * -100}

@feature(group='momentum')
def cci(df, n=20):
    return {'cci': pd.Series(ta.CCI(df['high'], df['low'], df['close'], timeperiod=n), index=df.index)}
