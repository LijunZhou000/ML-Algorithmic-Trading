import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='volume', requires=['vpt'])
def vpt_norm(df):
    return {'vpt_norm': df['vpt'] / df['close']}

@feature(group='volume')
def volume_by_range(df):
    return {'volume_range': df['volume'] / (df['high'] - df['low']).replace(0, np.nan)}

@feature(group='volume')
def volume_by_price(df):
    return {'volume_price': df['volume'] / df['close']}

@feature(group='volume', requires=['atr'])
def vol_vol_ratio(df):
    return {'vol_vol_ratio': df['volume'] / df['atr']}