import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='trend')
def sma(df, n=20):
    return {f"sma_{n}": df['close'].rolling(n).mean()}

@feature(group='trend')
def ema(df, n=20):
    return {f"ema_{n}": df['close'].ewm(span=n, adjust=False).mean()}

@feature(group='trend')
def dema(df, span=20):
    ema = df['close'].ewm(span=span).mean()
    return {'dema': 2*ema - ema.ewm(span=span).mean()}

@feature(group='trend')
def kama(df, window=10, fast=2, slow=30):
    # Prefer TA-Lib's KAMA for correctness and performance. Fall back to
    # a safe pandas/numpy implementation if TA-Lib call fails.
    try:
        return {'kama': pd.Series(ta.KAMA(df['close'], timeperiod=window), index=df.index)}
    except Exception:
        close = df['close'].astype(float).to_numpy()
        n = len(close)
        if n == 0:
            return {'kama': pd.Series(np.full(0, np.nan), index=df.index)}

        # Efficiency Ratio (ER): abs(close - close.shift(window)) / sum(abs(diff(close)))
        change = np.abs(close - np.roll(close, window)).astype(float)
        change[:window] = np.nan

        vol = np.concatenate([[np.nan], np.abs(np.diff(close))])
        vol = pd.Series(vol).rolling(window).sum().to_numpy()

        # avoid division by zero
        er = np.where(vol == 0, 0.0, change / vol)

        fast_sc = 2.0 / (fast + 1.0)
        slow_sc = 2.0 / (slow + 1.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = np.full(n, np.nan, dtype=float)
        # initialize first value at the end of the warm-up window
        if n > window:
            kama[window] = close[window]
            for i in range(window + 1, n):
                kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])

        return {'kama': pd.Series(kama, index=df.index)}