import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='trend')
def macd(df, n_fast=12, n_slow=26):
    ema_fast = df['close'].ewm(span=n_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=n_slow, adjust=False).mean()
    macd_s = ema_fast - ema_slow
    signal = macd_s.ewm(span=9, adjust=False).mean()
    histogram = macd_s - signal
    return {
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'macd': macd_s,
        'signal': signal,
        'histogram': histogram,
    }