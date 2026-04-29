import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='momentum')
def roc(df, window=10):
    return {f'roc_{window}': df['close'].pct_change(window)}

@feature(group='momentum')
def multi_roc(df, windows=None):
    if windows is None:
        windows = [1,5,10,20]
    out = {}
    for w in windows:
        out[f'roc_{w}'] = df['close'].pct_change(w)
    return out
