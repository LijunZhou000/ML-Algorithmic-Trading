import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='volume')
def relative_volume(df, window=20):
    # normalize names to english snake_case
    avg = df['volume'].rolling(window=window).mean()
    return {'avg_volume': avg, 'relative_volume': df['volume'] / avg}

@feature(group='volume')
def volume_zscore(df, window=20):
    mean = df['volume'].rolling(window).mean()
    std = df['volume'].rolling(window).std()
    return {'volume_z': (df['volume'] - mean) / std}

@feature(group='volume')
def volume_sum(df, window=20):
    return {f'volume_sum_{window}': df['volume'].rolling(window).sum()}

@feature(group='volume')
def volume_delta(df):
    return {'volume_delta': df['volume'].diff()}