import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='volume')
def obv(
    df,
    roc_windows=None,
    smooth_windows=None,
    zscore_window=20,
    relative_volume_window=20,
    normalize_by="range",  # "range", "close", "none"
    include_manual=False,
    include_direction=True,
    include_delta=True,
    session_key=None,
):
    obv_s = pd.Series(ta.OBV(df['close'], df['volume']), index=df.index)
    out = {'obv': obv_s}

    if include_manual:
        manual = ((df['close'].diff() > 0).astype(int) * df['volume'] - (df['close'].diff() < 0).astype(int) * df['volume'])
        if session_key is not None or 'trading_date' in df.columns:
            manual = manual.groupby(df['trading_date']).cumsum()
        else:
            manual = manual.cumsum()
        out['obv_manual'] = pd.Series(manual, index=df.index)

    if normalize_by == 'range':
        out['obv_norm'] = obv_s / (df['high'] - df['low']).replace(0, np.nan)
    elif normalize_by == 'close':
        out['obv_norm'] = obv_s / df['close']
    else:
        out['obv_norm'] = obv_s

    if roc_windows is None:
        roc_windows = [5,10]
    if smooth_windows is None:
        smooth_windows = [10]

    for w in roc_windows:
        out[f'obv_roc_{w}'] = obv_s.pct_change(w)
    for w in smooth_windows:
        out[f'obv_ema_{w}'] = obv_s.ewm(span=w, adjust=False).mean()
        out[f'obv_sma_{w}'] = obv_s.rolling(w).mean()

    mean = obv_s.rolling(zscore_window).mean()
    std = obv_s.rolling(zscore_window).std()
    out['obv_z'] = (obv_s - mean) / std

    if include_direction:
        out['obv_dir'] = np.sign(obv_s.diff())

    out['obv_rel'] = obv_s / df['volume'].rolling(relative_volume_window).sum()

    if include_delta:
        out['obv_delta'] = obv_s.diff()

    # wrap as Series
    return {k: pd.Series(v, index=df.index) for k,v in out.items()}

@feature(group='volume')
def vpt(df):
    vpt_s = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
    return {'vpt': vpt_s}

@feature(group='volume')
def ad(
    df,
    zscore_window=20,
    roc_windows=None,
    smooth_windows=None,
    relative_volume_window=20,
    normalize_by="range",   # "range", "close", "none"
    include_clv=True,
    include_delta=True,
    include_direction=True,
    session_key=None,
):
    out = {}
    if include_clv:
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
        out['clv'] = clv
    else:
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)

    ad_s = (clv * df['volume']).cumsum()
    out['ad'] = ad_s

    if normalize_by == 'range':
        out['ad_norm'] = ad_s / (df['high'] - df['low']).replace(0, np.nan)
    elif normalize_by == 'close':
        out['ad_norm'] = ad_s / df['close']
    else:
        out['ad_norm'] = ad_s

    if roc_windows is None:
        roc_windows = [5,10]
    if smooth_windows is None:
        smooth_windows = [10]

    for w in roc_windows:
        out[f'ad_roc_{w}'] = ad_s.pct_change(w)
    for w in smooth_windows:
        out[f'ad_ema_{w}'] = ad_s.ewm(span=w, adjust=False).mean()
        out[f'ad_sma_{w}'] = ad_s.rolling(w).mean()

    mean = ad_s.rolling(zscore_window).mean()
    std = ad_s.rolling(zscore_window).std()
    out['ad_z'] = (ad_s - mean) / std

    if include_delta:
        out['ad_delta'] = ad_s.diff()
    if include_direction:
        out['ad_dir'] = np.sign(ad_s.diff())

    out['ad_rel'] = ad_s / df['volume'].rolling(relative_volume_window).sum()

    return {k: pd.Series(v, index=df.index) for k,v in out.items()}
