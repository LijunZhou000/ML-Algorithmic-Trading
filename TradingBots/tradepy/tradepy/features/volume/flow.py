import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='volume')
def mfi(df, window=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    pos_sum = positive_flow.rolling(window).sum()
    neg_sum = negative_flow.rolling(window).sum()
    mfi_s = 100 - (100 / (1 + (pos_sum / neg_sum)))
    return {'mfi': mfi_s}

@feature(group='volume')
def cmf(df, window=20):
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    money_flow_volume = clv * df['volume']
    cmf_s = money_flow_volume.rolling(window).sum() / df['volume'].rolling(window).sum()
    return {'cmf': cmf_s}

@feature(group='volume')
def elder_force_index(df, n=13):
    efi = ta.EMA((df['close'] - df['close'].shift(1)) * df['volume'], timeperiod=n)
    return {'efi': pd.Series(efi, index=df.index)}
