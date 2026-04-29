import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature

@feature(group='time')
def time_features(df, break_hour=21):
    if 'datetime' in df.columns:
        dt_series = pd.to_datetime(df['datetime'])
        df['hour'] = dt_series.dt.hour
        df['minute'] = dt_series.dt.minute
        df['dayofweek'] = dt_series.dt.dayofweek
        df['day'] = dt_series.dt.day
        df['month'] = dt_series.dt.month
        df['is_month_end'] = dt_series.dt.is_month_end

        # Cyclical transforms: hour (24), minute (60), dayofweek (7), day (31), month (12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['day_sin'] = np.sin(2 * np.pi * (df['day'] - 1) / 31)
        df['day_cos'] = np.cos(2 * np.pi * (df['day'] - 1) / 31)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        df["is_prev_break_hour"] = df['hour'] + 1 == break_hour
        df["is_post_break_hour"] = df['hour'] - 1 == break_hour
        
        df["is_NY_open_hour"] = df['hour'] == 13
        df['week_of_month'] = (df['day'] - 1) // 7 + 1
        df['seconds_since_midnight'] = (df['hour'] * 3600 + df['minute'] * 60)
        df['session_progress'] = df['seconds_since_midnight'] / (24*3600)

        df['is_london'] = df['datetime'].dt.hour.isin(range(8, 17)).astype(int)
        df['is_ny'] = df['datetime'].dt.hour.isin(range(13, 22)).astype(int)
        df['is_overlap'] = ((df['is_london'] == 1) & (df['is_ny'] == 1)).astype(int)
        # collect created columns and return as dict
        keys = ['hour','minute','dayofweek','day','month','is_month_end','hour_sin','hour_cos','minute_sin','minute_cos','dow_sin','dow_cos','day_sin','day_cos','month_sin','month_cos','is_prev_break_hour','is_post_break_hour','is_NY_open_hour','week_of_month','seconds_since_midnight','session_progress','is_london','is_ny','is_overlap']
        return {k: df[k] for k in keys}
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        df['hour'] = idx.hour
        df['minute'] = idx.minute
        df['dayofweek'] = idx.dayofweek
        df['day'] = idx.day
        df['month'] = idx.month
        # is_month_end is an index property accessed via .is_month_end
        try:
            df['is_month_end'] = idx.is_month_end
        except Exception:
            df['is_month_end'] = False

        # Cyclical transforms for index-based datetime
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['day_sin'] = np.sin(2 * np.pi * (df['day'] - 1) / 31)
        df['day_cos'] = np.cos(2 * np.pi * (df['day'] - 1) / 31)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        df["is_prev_break_hour"] = df['hour'] + 1 == break_hour
        df["is_post_break_hour"] = df['hour'] - 1 == break_hour
        
        df["is_NY_open_hour"] = df['hour'] == 13
        df['week_of_month'] = (df['day'] - 1) // 7 + 1
        df['seconds_since_midnight'] = (df['hour'] * 3600 + df['minute'] * 60)
        df['session_progress'] = df['seconds_since_midnight'] / (24*3600)
        
        df['is_london'] = df['datetime'].dt.hour.isin(range(8, 17)).astype(int)
        df['is_ny'] = df['datetime'].dt.hour.isin(range(13, 22)).astype(int)
        df['is_overlap'] = ((df['is_london'] == 1) & (df['is_ny'] == 1)).astype(int)

        keys = ['hour','minute','dayofweek','day','month','is_month_end','hour_sin','hour_cos','minute_sin','minute_cos','dow_sin','dow_cos','day_sin','day_cos','month_sin','month_cos','is_prev_break_hour','is_post_break_hour','is_NY_open_hour','week_of_month','seconds_since_midnight','session_progress','is_london','is_ny','is_overlap']
        return {k: df[k] for k in keys}
    else:
        return df