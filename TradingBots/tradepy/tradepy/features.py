import talib as ta
import numpy as np
import pandas as pd
import json
import inspect

# Media movil simple

# Media movil exponencial


# Convergencia y divergencia de medias moviles


# Índice direccional medio
def adx(df, n=14):
    return {'adx': pd.Series(ta.ADX(df['high'], df['low'], df['close'], timeperiod=n), index=df.index)}

# Índice de fuerza relativa
def rsi(df, n=14):
    return {'rsi': pd.Series(ta.RSI(df['close'], timeperiod=n), index=df.index)}

# Estocástico
def stoch(df, n=14, slowk_period=3, slowd_period=3, slowk_matype=0, slowd_matype=0):
    slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=n, slowk_period=slowk_period, slowd_period=slowd_period, slowk_matype=slowk_matype, slowd_matype=slowd_matype)
    return {'slowk': pd.Series(slowk, index=df.index), 'slowd': pd.Series(slowd, index=df.index)}

# Índice de fuerza de elder
def elder_force_index(df, n=13):
    efi = ta.EMA((df['close'] - df['close'].shift(1)) * df['volume'], timeperiod=n)
    return {'efi': pd.Series(efi, index=df.index)}

# Bandas de bollinger
def bollinger_bands(df, n=20, num_std_dev=2):
    middle = df['close'].rolling(n).mean()
    std = df['close'].rolling(n).std()
    return {
        'bb_middle': middle,
        'bb_std': std,
        'bb_upper': middle + num_std_dev * std,
        'bb_lower': middle - num_std_dev * std,
    }

# Rango verdadero medio
def true_range(df, window=14):
    # pointwise true range (not ATR)
    tr = ta.TRANGE(df['high'], df['low'], df['close'])
    rng = df['high'] - df['low']
    rng_mean = rng.rolling(window).mean()
    return {'true_range': pd.Series(tr, index=df.index), 'range': rng, 'range_mean': rng_mean}

def compute_atr(df, period=14):
    # ATR computed as rolling mean of true_range.
    atr = df['true_range'].ewm(span=period, adjust=False).mean()
    return {'atr': atr}

# Commodity Channel Index
def cci(df, n=20):
    return {'cci': pd.Series(ta.CCI(df['high'], df['low'], df['close'], timeperiod=n), index=df.index)}

# Volumen en balance


# Volumen por precio

# Volumen relativo


# Acumulación/distribución




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






def historical_volatility(df, window=20):
    returns = np.log(df['close'] / df['close'].shift(1))
    return {f'hv': returns.rolling(window).std() * np.sqrt(252)}
def atr_normalized(df):
    return {'atr_norm': df['atr'] / df['close']}
def volatility_zscore(df, window=20):
    mean = df['atr'].rolling(window).mean()
    std = df['atr'].rolling(window).std()
    return {'atr_z': (df['atr'] - mean) / std}


def roc(df, window=10):
    return {f'roc_{window}': df['close'].pct_change(window)}

def multi_roc(df, windows=None):
    if windows is None:
        windows = [1,5,10,20]
    out = {}
    for w in windows:
        out[f'roc_{w}'] = df['close'].pct_change(w)
    return out
def williams_r(df, window=14):
    highest = df['high'].rolling(window).max()
    lowest = df['low'].rolling(window).min()
    return {'williams_r': (highest - df['close']) / (highest - lowest) * -100}
def cmo(df, window=14):
    diff = df['close'].diff()
    up = diff.clip(lower=0).rolling(window).sum()
    down = -diff.clip(upper=0).rolling(window).sum()
    return {'cmo': 100 * (up - down) / (up + down)}


def tr_direction(df):
    return {'tr_dir': np.sign(df['true_range'].diff())}
def candle_body(df):
    return {'body': (df['close'] - df['open']).abs()}
def wick_ratio(df):
    upper = df['high'] - df[['close','open']].max(axis=1)
    lower = df[['close','open']].min(axis=1) - df['low']
    return {'wick_ratio': (upper + lower) / (df['high'] - df['low'])}

def percent_return_features(df, return_horizon_min, sampling_minutes, lags=None, rolling_windows=None, ):
    """Create short- and multi-horizon return features.

    - `sampling_minutes` is the minutes per sample (1 for 1-min, 30 for 30-min, etc.)
    - `return_horizon_min` is the horizon in minutes for the horizon return feature.
    """
    if lags is None:
        lags = [1, 5, 10, 20]
    if rolling_windows is None:
        rolling_windows = [5, 10, 20]
    ret = df['close'].pct_change()
    log_ret = np.log(df['close'] / df['close'].shift(1))

    # horizon shift computed from minutes and sampling rate
    horizon_shift = int(return_horizon_min / max(1, int(sampling_minutes)))

    out = {'return': ret, 'log_return': log_ret}
    if horizon_shift >= 1:
        out[f'return_horizon_{return_horizon_min}m'] = df['close'].pct_change(horizon_shift)
        out[f'log_return_horizon_{return_horizon_min}m'] = np.log(df['close'] / df['close'].shift(horizon_shift))
    for lag in lags:
        out[f'return_lag_{lag}'] = ret.shift(lag)
    for w in rolling_windows:
        out[f'return_mean_{w}'] = ret.rolling(w).mean()
        out[f'return_std_{w}'] = ret.rolling(w).std()
        out[f'return_z_{w}'] = (ret - ret.rolling(w).mean()) / ret.rolling(w).std()
    return {k: pd.Series(v, index=df.index) for k,v in out.items()}

def percent_above_below_ma(df, ma_windows=None):
    if ma_windows is None:
        ma_windows = [20, 50]
    out = {}
    for w in ma_windows:
        ma = df['close'].rolling(w).mean()
        out[f'pct_above_ma_{w}'] = (df['close'] - ma) / ma
    return out

def bollinger_extra(df):
    # requires bb_upper, bb_lower, bb_middle
    out = {}
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns:
        out['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        out['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    return out

def rolling_moments(df, windows=None):
    # avoid mutable default argument; set default list inside function
    if windows is None:
        windows = [10, 20]
    out = {}
    for w in windows:
        out[f'rolling_skew_{w}'] = df['return'].rolling(w).skew()
        out[f'rolling_kurt_{w}'] = df['return'].rolling(w).kurt()
    return out

def rolling_slope(df, window=10, source='close'):
    y = df[source].values.astype(float)
    n = window
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    sum_x2 = (x * x).sum()
    kernel = x[::-1]
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(y, n)
    slopes = windows @ kernel / sum_x2
    result = np.full(len(y), np.nan)
    result[n - 1:] = slopes
    return {f'slope_{window}': pd.Series(result, index=df.index)}

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

def realized_volatility(df, window=20):
    # sum of squared intraday returns -> sqrt
    ret = df['log_return'] if 'log_return' in df.columns else np.log(df['close'] / df['close'].shift(1))
    return {f'realized_vol_{window}': np.sqrt((ret**2).rolling(window).sum()) * np.sqrt(252 / window)}

def is_doji(df, threshold=0.1):
    # small body relative to candle range
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return {'is_doji': (body / rng < threshold).astype(float)}

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

def generate_features(df, config_json=None, dropna_strategy='any'):
    cfg = {}
    if config_json:
        if isinstance(config_json, str):
            cfg = json.loads(config_json)
        elif isinstance(config_json, dict):
            cfg = config_json
        else:
            raise ValueError("config_json debe ser dict o JSON string")

    global_cfg = cfg.get("global", {})
    if 'dropna_strategy' in global_cfg and (dropna_strategy == 'any' or dropna_strategy is None):
        dropna_strategy = global_cfg.get('dropna_strategy')

    def p(name):
        return {**global_cfg, **cfg.get(name, {})}

    # Copia única al inicio
    base = df.copy()
    # Acumulador de todas las columnas nuevas: {col_name: Series}
    buf = {}

    def _snapshot():
        """Materializa buf sobre base y limpia buf. Necesario antes de
        funciones que leen columnas producidas por funciones anteriores."""
        nonlocal base
        if buf:
            base = pd.concat([base, pd.DataFrame(buf, index=base.index)], axis=1)
            buf.clear()

    def _call(func, name):
        params = p(name)
        session_funcs = {'compute_atr', 'obv', 'ad', 'rolling_slope'}
        if name in session_funcs and 'session_key' not in params:
            params = {**params, 'session_key': 'trading_date'}
        try:
            sig = inspect.signature(func)
            has_varkw = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in sig.parameters.values()
            )
            allowed = params if has_varkw else {
                k: v for k, v in params.items() if k in sig.parameters
            }
        except Exception:
            allowed = params

        result = func(base, **allowed) if allowed else func(base)

        # Normalizar distintos tipos de retorno a un mapping col->Series/array
        out_items = {}
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                out_items[col] = result[col]
        elif isinstance(result, dict):
            out_items = result
        elif isinstance(result, pd.Series):
            name = result.name or 'value'
            out_items[name] = result
        else:
            # intento best-effort: si es array 1D con longitud correcta
            try:
                arr = np.asarray(result)
                if arr.ndim == 1 and arr.shape[0] == len(base):
                    out_items['value'] = arr
            except Exception:
                out_items = {}

        # Acumular solo columnas nuevas
        for col, series in out_items.items():
            if col in base.columns or col in buf:
                continue
            if not isinstance(series, pd.Series):
                try:
                    series = pd.Series(series, index=base.index)
                except Exception:
                    # skip incompatible shapes
                    continue
            buf[col] = series.values  # almacenar valores crudos (más ligero)

    # ── Grupo 1: sin dependencias entre sí ──────────────────────────────
    _call(sma, 'sma')
    _call(ema, 'ema')
    _call(dema, 'dema')
    _call(kama, 'kama')
    _call(macd, 'macd')
    _call(rsi, 'rsi')
    _call(stoch, 'stoch')
    _call(cmo, 'cmo')
    _call(williams_r, 'williams_r')
    _call(adx, 'adx')
    _call(true_range, 'true_range')       # → atr, tr_direction
    _call(vpt, 'vpt')                     # → vpt_norm
    _call(obv, 'obv')
    _call(relative_volume, 'relative_volume')
    _call(ad, 'ad')
    _call(mfi, 'mfi')
    _call(cmf, 'cmf')
    _call(vwap, 'vwap')
    _call(volume_zscore, 'volume_zscore')
    _call(volume_delta, 'volume_delta')
    _call(volume_sum, 'volume_sum')
    _call(volume_by_range, 'volume_by_range')
    _call(volume_by_price, 'volume_by_price')
    _call(bollinger_bands, 'bollinger_bands')   # → bollinger_extra
    _call(percent_return_features, 'percent_return_features')  # → 'return', 'log_return'
    _call(time_features, 'time_features')
    _call(candle_body, 'candle_body')           # → 'body' para add_tick_features
    _call(historical_volatility, 'historical_volatility')
    _call(cci, 'cci')
    _call(elder_force_index, 'elder_force_index')
    _call(wick_ratio, 'wick_ratio')
    _call(is_doji, 'is_doji')

    _snapshot()  # ← materializar antes del grupo 2

    # ── Grupo 2: leen columnas del grupo 1 ──────────────────────────────
    _call(compute_atr, 'compute_atr')          # necesita true_range
    _call(vpt_norm, 'vpt_norm')                # necesita vpt
    _call(bollinger_extra, 'bollinger_extra')  # necesita bb_upper/lower/middle
    _call(rolling_moments, 'rolling_moments')  # necesita 'return'
    _call(realized_volatility, 'realized_volatility')  # necesita 'log_return'
    _call(percent_above_below_ma, 'percent_above_below_ma')
    _call(multi_roc, 'multi_roc')
    _call(rolling_slope, 'rolling_slope')
    _call(tr_direction, 'tr_direction')        # necesita true_range

    _snapshot()  # ← materializar antes del grupo 3

    # ── Grupo 3: leen atr (grupo 2) ─────────────────────────────────────
    _call(atr_normalized, 'atr_normalized')
    _call(volatility_zscore, 'volatility_zscore')
    _call(vol_vol_ratio, 'vol_vol_ratio')
    _call(roc, 'roc')
    _call(add_tick_features, 'add_tick_features')  # necesita body, range_ticks

    _snapshot()  # ← concat final

    base.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── dropna ───────────────────────────────────────────────────────────
    try:
        if dropna_strategy in (0, '0', None, False):
            pass
        elif isinstance(dropna_strategy, str) and dropna_strategy.lower() == 'all':
            base = base.dropna(how='all').reset_index(drop=True)
        elif isinstance(dropna_strategy, str) and dropna_strategy.lower() == 'any':
            base = base.dropna(how='any').reset_index(drop=True)
        elif isinstance(dropna_strategy, float) and 0 < dropna_strategy < 1:
            thresh = int(np.ceil(base.shape[1] * float(dropna_strategy)))
            base = base.dropna(thresh=thresh).reset_index(drop=True)
        elif isinstance(dropna_strategy, int) and dropna_strategy >= 1:
            base = base.dropna(thresh=int(dropna_strategy)).reset_index(drop=True)
        else:
            val = float(dropna_strategy)
            if 0 < val < 1:
                thresh = int(np.ceil(base.shape[1] * val))
                base = base.dropna(thresh=thresh)
            elif val == 0:
                pass
            else:
                base = base.dropna(thresh=int(val)).reset_index(drop=True)
    except Exception:
        base = base.dropna(how='all').reset_index(drop=True)

    return base

def data_quality_report(
    df,
    future_horizon=1,
    verbose=True,
    top_n=20,
    clean_infs=False,
    compute_corr=False,
    corr_method='pearson',
    corr_max_cols=100,
    cols=None,
    by_day=False,
):
    """Run a set of data-quality checks on `df` and return a report dict.

    New options:
    - `clean_infs`: replace inf/-inf with NaN and report counts
    - `compute_corr`: compute correlation with future returns (limited by `corr_max_cols`)
    - `cols`: optional list of columns to focus the analysis on
    - `by_day`: if True and `trading_date` present, compute NaN fraction by day
    Returns a dict `report` and includes a `column_profile` DataFrame under that key.
    """

    df = df.copy()
    report = {}
    n_rows, n_cols = df.shape
    report['rows'] = int(n_rows)
    report['cols'] = int(n_cols)

    # Work on a subset if requested to avoid heavy work
    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        df_sub = df[cols].copy()
    else:
        df_sub = df.copy()

    # NaN fractions
    na_frac = df_sub.isna().mean().sort_values(ascending=False)
    report['na_fraction'] = na_frac.to_dict()
    report['top_na'] = na_frac.head(top_n).to_dict()

    # Constant / zero-variance columns
    nunique = df_sub.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    report['constant_columns'] = constant_cols

    # Infinite values (detect TRUE infinities only). Optionally replace them with NaN.
    numeric_cols = df_sub.select_dtypes(include=[np.number]).columns.tolist()
    inf_cols = []
    inf_counts = {}
    for c in numeric_cols:
        try:
            arr = df_sub[c].values
            inf_mask = np.isinf(arr)
            if inf_mask.any():
                inf_cols.append(c)
                inf_counts[c] = int(inf_mask.sum())
                if clean_infs:
                    df_sub.loc[inf_mask, c] = np.nan
                    # also reflect on main df
                    df.loc[inf_mask, c] = np.nan
        except Exception:
            continue
    report['infinite_columns'] = inf_cols
    report['infinite_counts'] = inf_counts

    # Fraction of zeros
    if numeric_cols:
        zero_frac = (df_sub[numeric_cols] == 0).mean().sort_values(ascending=False)
        report['top_zero_fraction'] = zero_frac.head(top_n).to_dict()
    else:
        report['top_zero_fraction'] = {}

    # Object/category uniques
    obj_uniques = df_sub.select_dtypes(include=['object', 'category']).apply(lambda s: int(s.nunique(dropna=False))).to_dict()
    report['object_unique_counts'] = obj_uniques

    # Basic describe summary (reduced)
    try:
        desc = df_sub.describe().T[['min', '25%', '50%', '75%', 'max']]
        report['describe'] = desc.to_dict(orient='index')
    except Exception:
        report['describe'] = {}

    # Correlation with future returns (if close exists)
    top_corr = {}
    if compute_corr and 'close' in df_sub.columns and numeric_cols:
        # limit numeric columns considered to avoid heavy computation
        cand_cols = [c for c in numeric_cols if c != 'close']
        if len(cand_cols) > corr_max_cols:
            variances = df_sub[cand_cols].var().sort_values(ascending=False)
            cand_cols = variances.head(corr_max_cols).index.tolist()

        try:
            future = df_sub['close'].pct_change(-int(future_horizon))
            corrs = {}
            for c in cand_cols:
                try:
                    valid = (~df_sub[c].isna()) & (~future.isna())
                    if valid.sum() < 3:
                        continue
                    if corr_method == 'pearson':
                        val = float(np.corrcoef(df_sub.loc[valid, c], future.loc[valid])[0,1])
                    else:
                        val = df_sub.loc[valid, c].corr(future.loc[valid], method=corr_method)
                    corrs[c] = float(val) if not pd.isna(val) else None
                except Exception:
                    continue
            if corrs:
                top_corr = dict(sorted(corrs.items(), key=lambda kv: abs(kv[1]) if kv[1] is not None else 0, reverse=True)[:top_n])
        except Exception:
            top_corr = {}
    report['top_corr_with_future'] = top_corr

    # Daily-like columns NaN fraction
    daily_cols = [c for c in df_sub.columns if '_daily' in c]
    daily_na = {c: float(df_sub[c].isna().mean()) for c in daily_cols}
    report['daily_columns_na_fraction'] = daily_na

    # Optional: NaN fraction by trading day (if by_day requested)
    if by_day and 'trading_date' in df.columns:
        try:
            per_day = df.groupby('trading_date').apply(lambda d: d.isna().mean()).T
            report['nan_fraction_by_day'] = per_day.mean(axis=0).to_dict()
        except Exception:
            report['nan_fraction_by_day'] = {}

    # Suggestions
    suggestions = []
    if len(na_frac) and na_frac.max() > 0.5:
        suggestions.append('Many columns have >50% NaNs; consider dropping or re-engineering them.')
    if constant_cols:
        suggestions.append(f'Drop or fix constant columns: {constant_cols[:10]}')
    if inf_cols:
        suggestions.append(f'Columns contain true infinities (use clean_infs=True to replace): {inf_cols[:20]}')
    if any(c.startswith('slope_') for c in df_sub.columns):
        suggestions.append('`rolling_slope` may be slow; consider vectorized replacement or numba.')
    suggestions.append('To avoid fragmentation, build new columns in a dict and `pd.concat` once per group.')
    report['suggestions'] = suggestions

    # Build a compact column-level DataFrame for inspection
    try:
        col_stats = pd.DataFrame({
            'na_frac': na_frac,
            'inf_count': pd.Series(inf_counts),
        }).fillna(0)
        if numeric_cols:
            zf = pd.Series(report['top_zero_fraction'])
            col_stats['zero_frac'] = zf.reindex(col_stats.index).fillna(0)
        nunique = df_sub.nunique(dropna=False)
        col_stats['nunique'] = nunique.reindex(col_stats.index)
        report['column_profile'] = col_stats
    except Exception:
        report['column_profile'] = None

    if verbose:
        print('Data Quality Report')
        print(f' - rows: {n_rows}, cols: {n_cols}')
        print(f" - top {top_n} NaN fractions:\n{pd.Series(report['top_na'])}")
        if report['infinite_columns']:
            print(' - infinite columns:', report['infinite_counts'])
        if report['top_corr_with_future']:
            print(' - top corr with future:', report['top_corr_with_future'])
        print('\nSuggestions:')
        for s in suggestions:
            print(' - ' + s)

    return report