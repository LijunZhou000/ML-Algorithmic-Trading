import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='structural')
def add_tick_features(df, tick_size=0.1):

    # Ensure base cols exist
    if 'open' not in df.columns or 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
        return {}

    # compute basic micro-structure if missing
    if 'body' not in df.columns:
        body = (df['close'] - df['open']).abs()
    else:
        body = df['body']

    # absolute ticks
    body_ticks = (df['close'] - df['open']) / float(tick_size)
    range_ticks = (df['high'] - df['low']) / float(tick_size)
    upper_wick_ticks = (df['high'] - df[['open','close']].max(axis=1)) / float(tick_size)
    lower_wick_ticks = (df[['open','close']].min(axis=1) - df['low']) / float(tick_size)

    # proportions (safe denom)
    denom = range_ticks.replace(0, np.nan)
    body_pct = body_ticks.abs() / (denom + 1e-8)
    upper_wick_pct = upper_wick_ticks / (denom + 1e-8)
    lower_wick_pct = lower_wick_ticks / (denom + 1e-8)

    # close location within candle [0..1]
    rng = (df['high'] - df['low']).replace(0, np.nan)
    close_location = (df['close'] - df['low']) / (rng + 1e-8)

    # distances in ticks to intra-day aggregates (fall back to NaN if missing)
    if 'high_cum' in df.columns:
        dist_high_cum_ticks = (df['high_cum'] - df['close']) / float(tick_size)
    else:
        dist_high_cum_ticks = np.nan

    if 'low_cum' in df.columns:
        dist_low_cum_ticks = (df['close'] - df['low_cum']) / float(tick_size)
    else:
        dist_low_cum_ticks = np.nan

    if 'open_day' in df.columns:
        dist_open_day_ticks = (df['close'] - df['open_day']) / float(tick_size)
    else:
        dist_open_day_ticks = np.nan

    # volume dynamics (safe denominators)
    vol_per_tick = df['volume'] / (range_ticks.abs() + 1.0)
    vol_vs_body = df['volume'] / (body_ticks.abs() + 1.0)
    if 'volume_cum' in df.columns:
        vol_weight_intraday = df['volume'] / (df['volume_cum'] + 1.0)
    else:
        vol_weight_intraday = np.nan

    out = {
        'body_ticks': body_ticks,
        'range_ticks': range_ticks,
        'upper_wick_ticks': upper_wick_ticks,
        'lower_wick_ticks': lower_wick_ticks,
        'body_pct': body_pct,
        'upper_wick_pct': upper_wick_pct,
        'lower_wick_pct': lower_wick_pct,
        'close_location': close_location,
        'dist_high_cum_ticks': dist_high_cum_ticks,
        'dist_low_cum_ticks': dist_low_cum_ticks,
        'dist_open_day_ticks': dist_open_day_ticks,
        'vol_per_tick': vol_per_tick,
        'vol_vs_body': vol_vs_body,
        'vol_weight_intraday': vol_weight_intraday,
        'body': body,
    }
    return {k: pd.Series(v, index=df.index) if not isinstance(v, (float, int)) else pd.Series([v]*len(df), index=df.index) for k,v in out.items()}