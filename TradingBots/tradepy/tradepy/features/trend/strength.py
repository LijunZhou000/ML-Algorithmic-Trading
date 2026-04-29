import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='trend')
def adx(df, n=14):
    return {'adx': pd.Series(ta.ADX(df['high'], df['low'], df['close'], timeperiod=n), index=df.index)}
