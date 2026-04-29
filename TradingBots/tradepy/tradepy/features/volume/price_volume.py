import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='volume')
def vwap(df):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = typical_price * df['volume']
    group_keys = []
    if 'ticker' in df.columns:
        group_keys.append('ticker')
    if 'dtyyyymmdd' in df.columns:
        group_keys.append('dtyyyymmdd')
    if group_keys:
        vwap_s = tp_vol.groupby([df[k] for k in group_keys], sort=False).cumsum() / df['volume'].groupby([df[k] for k in group_keys], sort=False).cumsum()
    else:
        vwap_s = tp_vol.cumsum() / df['volume'].cumsum()
    return {'vwap': vwap_s}