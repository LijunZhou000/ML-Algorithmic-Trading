import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
import time
from tqdm import tqdm
from collections import OrderedDict

# Time horizon defaults (minutes-based)
MINUTES = 60
RETURN_HORIZON_MIN = MINUTES * 2
# default min_periods for rolling windows to reduce warm-up NaNs
ROLL_MIN_PERIODS = 1

def _ensure_trading_date(df):
    """Ensure DataFrame has a `trading_date` column used to reset intraday calculations.
    If `trading_date` exists it's left untouched. If `datetime` exists it's computed as
    `(datetime - 1 hour).date()` to match the project's trading-day convention.
    """
    if 'trading_date' in df.columns:
        return df
    if 'datetime' in df.columns:
        try:
            df['trading_date'] = (pd.to_datetime(df['datetime']) - pd.Timedelta(hours=1)).dt.date
        except Exception:
            df['trading_date'] = pd.to_datetime(df['datetime']).dt.date
        return df
    if 'dtyyyymmdd' in df.columns:
        df['trading_date'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d', errors='coerce').dt.date
        return df
    return df

# Media movil simple
def sma(df, n=20):
    df = df.copy()
    df[f"sma_{n}"] = df['close'].rolling(n, min_periods=ROLL_MIN_PERIODS).mean()
    return df
# Media movil exponencial
def ema(df, n=20):
    df = df.copy()
    df[f"ema_{n}"] = df['close'].ewm(span=n, adjust=False).mean()
    return df

# Convergencia y divergencia de medias moviles
def macd(df, n_fast=12, n_slow=26):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=n_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=n_slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    return df

# Índice direccional medio
def adx(df, n=14):
    df = df.copy()
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=n)
    return df

# Índice de fuerza relativa
def rsi(df, n=14):
    df = df.copy()
    df['rsi'] = ta.RSI(df['close'], timeperiod=n)
    return df

# Estocástico
def stoch(df, n=14, slowk_period=3, slowd_period=3, slowk_matype=0, slowd_matype=0):
    df = df.copy()
    df['slowk'], df['slowd'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=n, slowk_period=slowk_period, slowd_period=slowd_period, slowk_matype=slowk_matype, slowd_matype=slowd_matype)
    return df

# Índice de fuerza de elder
def elder_force_index(df, n=13):
    df = df.copy()
    df['efi'] = ta.EMA((df['close'] - df['close'].shift(1)) * df['volume'], timeperiod=n)
    return df

# Bandas de bollinger
def bollinger_bands(df, n=20, num_std_dev=2):
    df = df.copy()
    df['bb_middle'] = df['close'].rolling(n, min_periods=ROLL_MIN_PERIODS).mean()
    df['bb_std'] = df['close'].rolling(n, min_periods=ROLL_MIN_PERIODS).std()
    df['bb_upper'] = df['bb_middle'] + num_std_dev * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std_dev * df['bb_std']
    return df

# Rango verdadero medio
def true_range(df, window=14):
    # pointwise true range (not ATR)
    df = df.copy()
    df['true_range'] = ta.TRANGE(df['high'], df['low'], df['close'])
    df['range'] = df['high'] - df['low']
    df['range_mean'] = df['range'].rolling(window, min_periods=ROLL_MIN_PERIODS).mean()
    return df

def compute_atr(df, period=14, session_key=None):
    # ATR computed as rolling mean of true_range. If `session_key` or
    # `trading_date`/`datetime` present, compute ATR grouped per session
    # to avoid leakage across trading days.
    df = df.copy()
    if 'true_range' not in df.columns:
        df = true_range(df, window=period)

    if session_key is not None or any(k in df.columns for k in ('trading_date', 'datetime', 'dtyyyymmdd')):
        df = _ensure_trading_date(df)
        df['atr'] = df.groupby('trading_date')['true_range'].transform(lambda x: x.rolling(period, min_periods=ROLL_MIN_PERIODS).mean())
    else:
        df['atr'] = df['true_range'].rolling(period, min_periods=ROLL_MIN_PERIODS).mean()
    return df

# Commodity Channel Index
def cci(df, n=20):
    df = df.copy()
    df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=n)
    return df

# Volumen en balance
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
    df = df.copy()
    # If session-aware requested, build OBV per session to avoid cumsum leakage
    if session_key is not None or any(k in df.columns for k in ('trading_date', 'datetime', 'dtyyyymmdd')):
        df = _ensure_trading_date(df)
        # manual OBV per session
        dir_sign = (df['close'].diff() > 0).astype(int) - (df['close'].diff() < 0).astype(int)
        df['obv'] = (dir_sign * df['volume']).groupby(df['trading_date']).cumsum()
    else:
        # OBV clásico (TA-Lib)
        df['obv'] = ta.OBV(df['close'], df['volume'])

    # OBV manual optional preserved as per original API
    if include_manual:
        df['obv_manual'] = (
            (df['close'].diff() > 0).astype(int) * df['volume']
            - (df['close'].diff() < 0).astype(int) * df['volume']
        )
        if session_key is not None or 'trading_date' in df.columns:
            df['obv_manual'] = df['obv_manual'].groupby(df['trading_date']).cumsum()
        else:
            df['obv_manual'] = df['obv_manual'].cumsum()

    # Normalizaciones
    if normalize_by == "range":
        df['obv_norm'] = df['obv'] / (df['high'] - df['low']).replace(0, np.nan)
    elif normalize_by == "close":
        df['obv_norm'] = df['obv'] / df['close']
    else:
        df['obv_norm'] = df['obv']

    # defaults for list args to avoid mutable defaults
    if roc_windows is None:
        roc_windows = [5, 10]
    if smooth_windows is None:
        smooth_windows = [10]

    # OBV ROC
    for w in roc_windows:
        df[f'obv_roc_{w}'] = df['obv'].pct_change(w)

    # Suavizados
    for w in smooth_windows:
        df[f'obv_ema_{w}'] = df['obv'].ewm(span=w, adjust=False).mean()
        df[f'obv_sma_{w}'] = df['obv'].rolling(w, min_periods=ROLL_MIN_PERIODS).mean()

    # Z-score
    mean = df['obv'].rolling(zscore_window, min_periods=ROLL_MIN_PERIODS).mean()
    std = df['obv'].rolling(zscore_window, min_periods=ROLL_MIN_PERIODS).std()
    df['obv_z'] = (df['obv'] - mean) / std

    # Dirección del OBV
    if include_direction:
        df['obv_dir'] = df['obv'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # OBV relativo al volumen
    df['obv_rel'] = df['obv'] / df['volume'].rolling(relative_volume_window, min_periods=ROLL_MIN_PERIODS).sum()

    # Derivada del OBV
    if include_delta:
        df['obv_delta'] = df['obv'].diff()

    return df

# Volumen por precio
def vpt(df):
    df = df.copy()
    df['vpt'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
    return df

# Volumen relativo
def relative_volume(df, window=20):
    # normalize names to english snake_case
    df = df.copy()
    df['avg_volume'] = df['volume'].rolling(window=window, min_periods=ROLL_MIN_PERIODS).mean()
    df['relative_volume'] = df['volume'] / df['avg_volume']
    return df

# Acumulación/distribución
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
    df = df.copy()
    # --- CLV (Close Location Value) ---
    if include_clv:
        df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
                    (df["high"] - df["low"]).replace(0, np.nan)

    # --- A/D clásico (session-aware) ---
    if session_key is not None or any(k in df.columns for k in ('trading_date', 'datetime', 'dtyyyymmdd')):
        df = _ensure_trading_date(df)
        df["ad"] = (df["clv"] * df["volume"]).groupby(df['trading_date']).cumsum()
    else:
        df["ad"] = (df["clv"] * df["volume"]).cumsum()

    # --- Normalización ---
    if normalize_by == "range":
        df["ad_norm"] = df["ad"] / (df["high"] - df["low"]).replace(0, np.nan)
    elif normalize_by == "close":
        df["ad_norm"] = df["ad"] / df["close"]
    else:
        df["ad_norm"] = df["ad"]

    # defaults for list args
    if roc_windows is None:
        roc_windows = [5, 10]
    if smooth_windows is None:
        smooth_windows = [10]

    # --- Rate of Change (momentum del A/D) ---
    for w in roc_windows:
        df[f"ad_roc_{w}"] = df["ad"].pct_change(w)

    # --- Suavizados ---
    for w in smooth_windows:
        df[f"ad_ema_{w}"] = df["ad"].ewm(span=w, adjust=False).mean()
        df[f"ad_sma_{w}"] = df['ad'].rolling(w, min_periods=ROLL_MIN_PERIODS).mean()

    # --- Z-score (acumulación/distribución extrema) ---
    mean = df["ad"].rolling(zscore_window, min_periods=ROLL_MIN_PERIODS).mean()
    std = df["ad"].rolling(zscore_window, min_periods=ROLL_MIN_PERIODS).std()
    df["ad_z"] = (df["ad"] - mean) / std

    # --- Derivada del A/D ---
    if include_delta:
        df["ad_delta"] = df["ad"].diff()

    # --- Dirección del A/D ---
    if include_direction:
        df["ad_dir"] = df["ad"].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # --- A/D relativo al volumen reciente ---
    df["ad_rel"] = df["ad"] / df["volume"].rolling(relative_volume_window, min_periods=ROLL_MIN_PERIODS).sum()

    return df

def mfi(df, window=14):
    df = df.copy()
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    pos_sum = positive_flow.rolling(window, min_periods=ROLL_MIN_PERIODS).sum()
    neg_sum = negative_flow.rolling(window, min_periods=ROLL_MIN_PERIODS).sum()

    df['mfi'] = 100 - (100 / (1 + (pos_sum / neg_sum)))
    return df
def cmf(df, window=20):
    df = df.copy()
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low']).replace(0, np.nan)

    money_flow_volume = clv * df['volume']

    df['cmf'] = money_flow_volume.rolling(window, min_periods=ROLL_MIN_PERIODS).sum() / df['volume'].rolling(window, min_periods=ROLL_MIN_PERIODS).sum()
    return df
def vwap(df):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = typical_price * df['volume']

    # Resetear por sesión (fecha) y ticker
    group_keys = []
    if 'ticker' in df.columns:
        group_keys.append('ticker')
    if 'dtyyyymmdd' in df.columns:
        group_keys.append('dtyyyymmdd')

    if group_keys:
        df['vwap'] = (
            tp_vol.groupby([df[k] for k in group_keys], sort=False).cumsum()
            / df['volume'].groupby([df[k] for k in group_keys], sort=False).cumsum()
        )
    else:
        # fallback: sin info de fecha/ticker, cumsum global
        df['vwap'] = tp_vol.cumsum() / df['volume'].cumsum()

    return df
def volume_zscore(df, window=20):
    df = df.copy()
    mean = df['volume'].rolling(window, min_periods=ROLL_MIN_PERIODS).mean()
    std = df['volume'].rolling(window, min_periods=ROLL_MIN_PERIODS).std()
    df['volume_z'] = (df['volume'] - mean) / std
    return df
def volume_delta(df):
    df = df.copy()
    df['volume_delta'] = df['volume'].diff()
    return df
def volume_sum(df, window=20):
    df = df.copy()
    df[f'volume_sum_{window}'] = df['volume'].rolling(window).sum()
    return df
def volume_by_range(df):
    df = df.copy()
    df['volume_range'] = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    return df
def volume_by_price(df):
    df = df.copy()
    df['volume_price'] = df['volume'] / df['close']
    return df

def historical_volatility(df, window=20):
    df = df.copy()
    returns = np.log(df['close'] / df['close'].shift(1))
    df['hv'] = returns.rolling(window).std() * np.sqrt(252)
    return df
def atr_normalized(df):
    df = df.copy()
    df['atr_norm'] = df['atr'] / df['close']
    return df
def volatility_zscore(df, window=20):
    df = df.copy()
    mean = df['atr'].rolling(window).mean()
    std = df['atr'].rolling(window).std()
    df['atr_z'] = (df['atr'] - mean) / std
    return df
def dema(df, span=20):
    df = df.copy()
    ema = df['close'].ewm(span=span).mean()
    df['dema'] = 2*ema - ema.ewm(span=span).mean()
    return df
def kama(df, window=10, fast=2, slow=30):
    # Prefer TA-Lib's KAMA for correctness and performance. Fall back to
    # a safe pandas/numpy implementation if TA-Lib call fails.
    df = df.copy()
    try:
        df['kama'] = ta.KAMA(df['close'], timeperiod=window)
        return df
    except Exception:
        close = df['close'].astype(float).to_numpy()
        n = len(close)
        if n == 0:
            df['kama'] = np.nan
            return df

        # Efficiency Ratio (ER): abs(close - close.shift(window)) / sum(abs(diff(close)))
        change = np.abs(close - np.roll(close, window)).astype(float)
        change[:window] = np.nan

        vol = np.concatenate([[np.nan], np.abs(np.diff(close))])
        vol = pd.Series(vol).rolling(window).sum().to_numpy()

        # avoid division by zero
        er = np.where(vol == 0, 0.0, change / vol)

        fast_sc = 2.0 / (fast + 1.0)
        slow_sc = 2.0 / (slow + 1.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = np.full(n, np.nan, dtype=float)
        # initialize first value at the end of the warm-up window
        if n > window:
            kama[window] = close[window]
            for i in range(window + 1, n):
                kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])

        df['kama'] = kama
        return df
def roc(df, window=10):
    df = df.copy()
    df[f'roc_{window}'] = df['close'].pct_change(window)
    return df

def multi_roc(df, windows=None):
    df = df.copy()
    if windows is None:
        windows = [1, 5, 10, 20]
    for w in windows:
        df[f'roc_{w}'] = df['close'].pct_change(w)
    return df
def williams_r(df, window=14):
    df = df.copy()
    highest = df['high'].rolling(window, min_periods=ROLL_MIN_PERIODS).max()
    lowest = df['low'].rolling(window, min_periods=ROLL_MIN_PERIODS).min()
    df['williams_r'] = (highest - df['close']) / (highest - lowest)*-100
    return df
def cmo(df, window=14):
    df = df.copy()
    diff = df['close'].diff()
    up = diff.clip(lower=0).rolling(window, min_periods=ROLL_MIN_PERIODS).sum()
    down = -diff.clip(upper=0).rolling(window, min_periods=ROLL_MIN_PERIODS).sum()
    df['cmo'] = 100 * (up - down) / (up + down)
    return df
def vol_vol_ratio(df):
    df = df.copy()
    df['vol_vol_ratio'] = df['volume'] / df['atr']
    return df
def vpt_norm(df):
    df = df.copy()
    df['vpt_norm'] = df['vpt'] / df['close']
    return df
def tr_direction(df):
    df = df.copy()
    df['tr_dir'] = np.sign(df['true_range'].diff())
    return df
def candle_body(df):
    df = df.copy()
    df['body'] = (df['close'] - df['open']).abs()
    return df
def wick_ratio(df):
    df = df.copy()
    upper = df['high'] - df[['close','open']].max(axis=1)
    lower = df[['close','open']].min(axis=1) - df['low']
    df['wick_ratio'] = (upper + lower) / (df['high'] - df['low'])
    return df

def percent_return_features(df, lags=None, rolling_windows=None, sampling_minutes=1, return_horizon_min=RETURN_HORIZON_MIN):
    """Create short- and multi-horizon return features.

    - `sampling_minutes` is the minutes per sample (1 for 1-min, 30 for 30-min, etc.)
    - `return_horizon_min` is the horizon in minutes for the horizon return feature.
    """
    df = df.copy()
    if lags is None:
        lags = [1, 5, 10, 20]
    if rolling_windows is None:
        rolling_windows = [5, 10, 20]
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # horizon shift computed from minutes and sampling rate
    try:
        horizon_shift = int(return_horizon_min / max(1, int(sampling_minutes)))
    except Exception:
        horizon_shift = int(RETURN_HORIZON_MIN)

    if horizon_shift >= 1:
        df[f'return_horizon_{return_horizon_min}m'] = df['close'].pct_change(horizon_shift)
        df[f'log_return_horizon_{return_horizon_min}m'] = np.log(df['close'] / df['close'].shift(horizon_shift))

    for lag in lags:
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
    for w in rolling_windows:
        df[f'return_mean_{w}'] = df['return'].rolling(w, min_periods=ROLL_MIN_PERIODS).mean()
        df[f'return_std_{w}'] = df['return'].rolling(w, min_periods=ROLL_MIN_PERIODS).std()
        df[f'return_z_{w}'] = (df['return'] - df['return'].rolling(w, min_periods=ROLL_MIN_PERIODS).mean()) / df['return'].rolling(w, min_periods=ROLL_MIN_PERIODS).std()
    return df


def create_targets(df, return_horizon_min=RETURN_HORIZON_MIN, sampling_minutes=1, tick_size=None, ternary=False, pos_threshold=0.0):
    """Create target columns for a given horizon.

    Behavior:
    - If `tick_size` is provided the target is computed in ticks and
        classification thresholds (`pos_threshold`) are interpreted in ticks.
    - If no `tick_size` is provided the function falls back to log-returns/pct.

    Parameters
    - `return_horizon_min`: horizon in minutes (e.g., 30, 60)
    - `sampling_minutes`: minutes per sample (1 for 1-min, 30 for 30-min)
    - `tick_size`: tick size (e.g. 0.1). If provided `target_ticks_*m` is created
    - `ternary`: if True create `target_class` with 0=SELL,1=NEUTRAL,2=BUY
    - `pos_threshold`: threshold for deciding neutral vs buy/sell. Interpreted as
        ticks when `tick_size` is provided, otherwise as log-return.
    """
    df = df.copy()
    # compute shift in rows
    try:
        shift = max(1, int(return_horizon_min / max(1, int(sampling_minutes))))
    except Exception:
        shift = max(1, int(RETURN_HORIZON_MIN))

    # log return and pct return for horizon
    col_log = f'target_logret_{return_horizon_min}m'
    col_pct = f'target_ret_{return_horizon_min}m'
    df[col_log] = np.log(df['close'].shift(-shift) / df['close'])
    df[col_pct] = df['close'].shift(-shift) / df['close'] - 1

    # ticks if tick_size provided
    if tick_size is not None:
        col_ticks = f'target_ticks_{return_horizon_min}m'
        df[col_ticks] = (df['close'].shift(-shift) - df['close']) / float(tick_size)

    # Prefer ticks when tick_size is provided
    if tick_size is not None:
        # create ticks column already above; classify by ticks
        # Interpret `pos_threshold` flexibly: if user passed a small value (<1)
        # treat it as a relative price return and convert to ticks per-row.
        if pos_threshold is None:
            pos_threshold = 0.0

        if abs(pos_threshold) < 1.0 and pos_threshold != 0.0:
            # per-row threshold in ticks = (return_threshold * price) / tick_size
            thresh_ticks = (pos_threshold * df['close']).abs() / float(tick_size)
        else:
            # pos_threshold already expressed in ticks (or is 0)
            thresh_ticks = float(abs(pos_threshold))

        # Binary target: require movement in ticks beyond threshold (if threshold>0)
        if (hasattr(thresh_ticks, 'shape')):
            df['target_bin'] = (df[col_ticks].abs() > thresh_ticks).astype(int) * (df[col_ticks] > 0).astype(int)
            # This produces 1 for positive moves above threshold, 0 otherwise
        else:
            if thresh_ticks > 0:
                df['target_bin'] = (df[col_ticks] > thresh_ticks).astype(int)
            else:
                df['target_bin'] = (df[col_ticks] > 0).astype(int)

        if ternary:
            # Vectorized ternary classification using per-row thresholds when available
            if hasattr(thresh_ticks, 'shape'):
                df['target_class'] = np.where(df[col_ticks] > thresh_ticks, 2,
                                              np.where(df[col_ticks] < -thresh_ticks, 0, 1))
                # preserve NaN where appropriate
                df.loc[df[col_ticks].isna(), 'target_class'] = np.nan
            else:
                t = float(thresh_ticks)
                def _clas_ticks(x):
                    if np.isnan(x):
                        return np.nan
                    if x > t:
                        return 2
                    if x < -t:
                        return 0
                    return 1

                df['target_class'] = df[col_ticks].apply(_clas_ticks)
    else:
        # fallback to log-return based targets
        df['target_bin'] = (df[col_log] > 0).astype(int)
        if ternary:
            def _clas(val):
                if np.isnan(val):
                    return np.nan
                if val > pos_threshold:
                    return 2
                if val < -pos_threshold:
                    return 0
                return 1

            df['target_class'] = df[col_log].apply(_clas)
    df = df.iloc[:-shift]
    return df

def percent_above_below_ma(df, ma_windows=None):
    df = df.copy()
    if ma_windows is None:
        ma_windows = [20, 50]
    for w in ma_windows:
        ma = df['close'].rolling(w, min_periods=ROLL_MIN_PERIODS).mean()
        df[f'pct_above_ma_{w}'] = (df['close'] - ma) / ma
    return df

def bollinger_extra(df):
    # requires bb_upper, bb_lower, bb_middle
    df = df.copy()
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    return df

def rolling_moments(df, windows=None):
    # avoid mutable default argument; set default list inside function
    df = df.copy()
    if windows is None:
        windows = [10, 20]
    for w in windows:
        df[f'rolling_skew_{w}'] = df['return'].rolling(w, min_periods=ROLL_MIN_PERIODS).skew()
        df[f'rolling_kurt_{w}'] = df['return'].rolling(w, min_periods=ROLL_MIN_PERIODS).kurt()
    return df

def rolling_slope(df, window=10, source='close', session_key=None):
    df = df.copy()
    # If session-aware requested, compute slope per session to avoid mixing days
    if session_key is not None or any(k in df.columns for k in ('trading_date', 'datetime', 'dtyyyymmdd')):
        df = _ensure_trading_date(df)

        def _slope_series(s):
            y = s.values.astype(float)
            n = window
            # Support variable-length windows when min_periods < window.
            # For each rolling window, compute linear slope using local x (0..m-1)
            def _slope_apply(arr):
                m = len(arr)
                if m < 2:
                    return np.nan
                x_local = np.arange(m, dtype=float)
                sum_x = x_local.sum()
                sum_x2 = (x_local * x_local).sum()
                y_sum_local = float(arr.sum())
                yx_local = float(np.dot(arr, x_local))
                num = m * yx_local - sum_x * y_sum_local
                den = m * sum_x2 - sum_x * sum_x
                return float(num / den) if den != 0 else np.nan

            y_series = pd.Series(y, index=s.index)
            slopes = y_series.rolling(n, min_periods=ROLL_MIN_PERIODS).apply(_slope_apply, raw=True)
            return slopes.rename(s.name)

        df[f'slope_{window}'] = df.groupby('trading_date')[source].transform(_slope_series)
        return df

    # session-unaware (fast vectorized path)
    y = df[source].values.astype(float)
    n = window
    x = np.arange(n, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    y_sum = pd.Series(y).rolling(n, min_periods=ROLL_MIN_PERIODS).sum().values
    yx = pd.Series(y).rolling(n, min_periods=ROLL_MIN_PERIODS).apply(lambda arr: np.dot(arr, x), raw=True).values
    num = n * yx - sum_x * y_sum
    den = n * sum_x2 - sum_x * sum_x
    slope = num / den
    df[f'slope_{window}'] = slope
    return df

def time_features(df):
    df = df.copy()
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
        return df
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
        return df
    else:
        return df

def realized_volatility(df, window=20):
    # sum of squared intraday returns -> sqrt
    df = df.copy()
    ret = df['log_return'] if 'log_return' in df.columns else np.log(df['close'] / df['close'].shift(1))
    df[f'realized_vol_{window}'] = np.sqrt((ret**2).rolling(window, min_periods=ROLL_MIN_PERIODS).sum()) * np.sqrt(252 / window)
    return df

def is_doji(df, threshold=0.1):
    # small body relative to candle range
    df = df.copy()
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    df['is_doji'] = (body / rng) < threshold
    df['is_doji'] = df['is_doji'].astype(float)
    return df

def add_tick_features(df, tick_size=0.1):
    df = df.copy()

    # Ensure base cols exist
    if 'open' not in df.columns or 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
        return df

    # compute basic micro-structure if missing
    if 'body' not in df.columns:
        df['body'] = (df['close'] - df['open']).abs()

    # absolute ticks
    df['body_ticks'] = (df['close'] - df['open']) / float(tick_size)
    df['range_ticks'] = (df['high'] - df['low']) / float(tick_size)
    df['upper_wick_ticks'] = (df['high'] - df[['open', 'close']].max(axis=1)) / float(tick_size)
    df['lower_wick_ticks'] = (df[['open', 'close']].min(axis=1) - df['low']) / float(tick_size)

    # proportions (safe denom)
    denom = df['range_ticks'].replace(0, np.nan)
    df['body_pct'] = df['body_ticks'].abs() / (denom + 1e-8)
    df['upper_wick_pct'] = df['upper_wick_ticks'] / (denom + 1e-8)
    df['lower_wick_pct'] = df['lower_wick_ticks'] / (denom + 1e-8)

    # close location within candle [0..1]
    rng = (df['high'] - df['low']).replace(0, np.nan)
    df['close_location'] = (df['close'] - df['low']) / (rng + 1e-8)

    # distances in ticks to intra-day aggregates (fall back to NaN if missing)
    if 'high_cum' in df.columns:
        df['dist_high_cum_ticks'] = (df['high_cum'] - df['close']) / float(tick_size)
    else:
        df['dist_high_cum_ticks'] = np.nan

    if 'low_cum' in df.columns:
        df['dist_low_cum_ticks'] = (df['close'] - df['low_cum']) / float(tick_size)
    else:
        df['dist_low_cum_ticks'] = np.nan

    if 'open_day' in df.columns:
        df['dist_open_day_ticks'] = (df['close'] - df['open_day']) / float(tick_size)
    else:
        df['dist_open_day_ticks'] = np.nan

    # volume dynamics (safe denominators)
    df['vol_per_tick'] = df['volume'] / (df['range_ticks'].abs() + 1.0)
    df['vol_vs_body'] = df['volume'] / (df['body_ticks'].abs() + 1.0)
    if 'volume_cum' in df.columns:
        df['vol_weight_intraday'] = df['volume'] / (df['volume_cum'] + 1.0)
    else:
        df['vol_weight_intraday'] = np.nan

    return df

def generate_features(df, config_json=None, dropna_strategy='any'):
    """Genera indicadores técnicos en el DataFrame `df`.
    `config_json` puede ser un diccionario o un string JSON con configuraciones por indicador,
    por ejemplo: {'sma': {'n':20}, 'macd': {'n_fast':12,'n_slow':26}}
    Devuelve el DataFrame con nuevas columnas."""
    cfg = {}
    if config_json:
        if isinstance(config_json, str):
            cfg = json.loads(config_json)
        elif isinstance(config_json, dict):
            cfg = config_json
        else:
            raise ValueError("config_json debe ser dict o JSON string")

    # parámetros globales
    global_cfg = cfg.get("global", {})
    # allow override of dropna strategy from global config
    if 'dropna_strategy' in global_cfg and (dropna_strategy == 'any' or dropna_strategy is None):
        dropna_strategy = global_cfg.get('dropna_strategy')

    # helper para obtener parámetros del indicador
    def p(name):
        local = cfg.get(name, {})
        return {**global_cfg, **local}  # local override

    # función auxiliar
    def _call(func, name, pass_cfg=True):
        params = p(name)
        # inject session_key by default for session-aware functions
        session_funcs = {'compute_atr', 'obv', 'ad', 'rolling_slope'}
        if name in session_funcs:
            if 'session_key' not in params:
                params = {**params, 'session_key': 'trading_date'}
        # Only pass kwargs accepted by the function to avoid TypeError
        if pass_cfg and params:
            try:
                sig = inspect.signature(func)
                has_varkw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                if has_varkw:
                    allowed = params
                else:
                    allowed = {k: v for k, v in params.items() if k in sig.parameters}
            except Exception:
                # If signature inspection fails, fall back to passing params (best-effort)
                allowed = params

            if allowed:
                return func(df, **allowed)
        return func(df)
    # Aplicar indicadores (se aplican en un orden lógico)
    df = df.copy()
    # Tendencia / medias
    df = _call(sma, 'sma')
    df = _call(ema, 'ema')
    df = _call(dema, 'dema')
    df = _call(kama, 'kama')
    # Momentum / osciladores
    df = _call(macd, 'macd')
    df = _call(rsi, 'rsi')
    df = _call(stoch, 'stoch')
    df = _call(cmo, 'cmo')
    df = _call(williams_r, 'williams_r')
    # Dirección / volatilidad
    df = _call(adx, 'adx')
    df = _call(true_range, 'true_range')
    df = _call(compute_atr, 'compute_atr')
    df = _call(atr_normalized, 'atr_normalized')
    df = _call(volatility_zscore, 'volatility_zscore')
    df = _call(historical_volatility, 'historical_volatility')
    # Volumen y derivados
    df = _call(obv, 'obv')
    df = _call(vpt, 'vpt')
    df = _call(vpt_norm, 'vpt_norm')
    df = _call(relative_volume, 'relative_volume')
    df = _call(ad, 'ad')
    df = _call(mfi, 'mfi')
    df = _call(cmf, 'cmf')
    df = _call(vwap, 'vwap')
    df = _call(volume_zscore, 'volume_zscore')
    df = _call(volume_delta, 'volume_delta')
    df = _call(volume_sum, 'volume_sum')
    df = _call(volume_by_range, 'volume_by_range')
    df = _call(volume_by_price, 'volume_by_price')
    df = _call(vol_vol_ratio, 'vol_vol_ratio')
    # returns and momentum
    df = _call(percent_return_features, 'percent_return_features')
    df = _call(percent_above_below_ma, 'percent_above_below_ma')
    df = _call(multi_roc, 'multi_roc')
    df = _call(rolling_moments, 'rolling_moments')
    df = _call(rolling_slope, 'rolling_slope')
    df = _call(time_features, 'time_features')
    # Indicadores técnicos adicionales
    df = _call(cci, 'cci')
    df = _call(bollinger_bands, 'bollinger_bands')
    df = _call(bollinger_extra, 'bollinger_extra')
    df = _call(elder_force_index, 'elder_force_index')
    df = _call(candle_body, 'candle_body')
    df = _call(wick_ratio, 'wick_ratio')
    df = _call(roc, 'roc')
    # additional small utilities
    df = _call(tr_direction, 'tr_direction')
    df = _call(is_doji, 'is_doji')
    df = _call(realized_volatility, 'realized_volatility')
    # Normalizaciones / utilidades finales
    df = _call(atr_normalized, 'atr_normalized')
    df = _call(volatility_zscore, 'volatility_zscore')
    df = _call(add_tick_features, 'add_tick_features')
    # replace infinities with NaN before applying dropna strategy
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # dropna strategy options:
    # - 0 / None / False -> do not drop any rows
    # - 'all' -> drop rows where all values are NaN
    # - float between 0 and 1 -> minimum fraction of non-NA columns required
    # - int >= 1 -> minimum number of non-NA columns required
    try:
        if dropna_strategy in (0, '0', None, False):
            pass
        elif isinstance(dropna_strategy, str) and dropna_strategy.lower() == 'all':
            df = df.dropna(how='all')
            df = df.reset_index(drop=True)
        elif isinstance(dropna_strategy, str) and dropna_strategy.lower() == 'any':
            df = df.dropna(how='any')
            df = df.reset_index(drop=True)
        elif isinstance(dropna_strategy, float) and 0 < dropna_strategy < 1:
            thresh = int(np.ceil(df.shape[1] * float(dropna_strategy)))
            df = df.dropna(thresh=thresh)
            df = df.reset_index(drop=True)
        elif isinstance(dropna_strategy, int) and dropna_strategy >= 1:
            df = df.dropna(thresh=int(dropna_strategy))
            df = df.reset_index(drop=True)
        else:
            # try to coerce numeric-like strings
            val = float(dropna_strategy)
            if 0 < val < 1:
                thresh = int(np.ceil(df.shape[1] * val))
                df = df.dropna(thresh=thresh)
            elif val == 0:
                pass
            else:
                df = df.dropna(thresh=int(val))
                df = df.reset_index(drop=True)
    except Exception:
        # safe fallback: remove rows that are all-NaN
        df = df.dropna(how='all')
        df = df.reset_index(drop=True)

    return df


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


def profile_generate_features(df, config_json=None, show_progress=True):
    """Run feature generation step-by-step and time each feature using tqdm.

    Returns (df, timings) where timings is an ordered dict of {feature_name: seconds}.
    """

    cfg = {}
    if config_json:
        if isinstance(config_json, str):
            cfg = json.loads(config_json)
        elif isinstance(config_json, dict):
            cfg = config_json
        else:
            raise ValueError("config_json must be dict or JSON string")

    df = df.copy()

    global_cfg = cfg.get("global", {})
    def p(name):
        local = cfg.get(name, {})
        return {**global_cfg, **local}

    def call_with_params(func, name):
        params = p(name)
        if params:
            return func(df, **params)
        return func(df)

    steps = [
        (sma, 'sma'), (ema, 'ema'), (dema, 'dema'), (kama, 'kama'),
        (macd, 'macd'), (rsi, 'rsi'), (stoch, 'stoch'), (cmo, 'cmo'), (williams_r, 'williams_r'),
        (adx, 'adx'), (true_range, 'true_range'), (compute_atr, 'compute_atr'), (atr_normalized, 'atr_normalized'), (volatility_zscore, 'volatility_zscore'), (historical_volatility, 'historical_volatility'),
        (obv, 'obv'), (vpt, 'vpt'), (vpt_norm, 'vpt_norm'), (relative_volume, 'relative_volume'), (ad, 'ad'),
        (mfi, 'mfi'), (cmf, 'cmf'), (vwap, 'vwap'), (volume_zscore, 'volume_zscore'), (volume_delta, 'volume_delta'),
        (volume_sum, 'volume_sum'), (volume_by_range, 'volume_by_range'), (volume_by_price, 'volume_by_price'), (vol_vol_ratio, 'vol_vol_ratio'),
        (percent_return_features, 'percent_return_features'), (percent_above_below_ma, 'percent_above_below_ma'), (multi_roc, 'multi_roc'),
        (rolling_moments, 'rolling_moments'), (rolling_slope, 'rolling_slope'), (time_features, 'time_features'),
        (cci, 'cci'), (bollinger_bands, 'bollinger_bands'), (bollinger_extra, 'bollinger_extra'),
        (elder_force_index, 'elder_force_index'), (candle_body, 'candle_body'), (wick_ratio, 'wick_ratio'), (roc, 'roc')
    ]

    timings = OrderedDict()
    iterator = steps if not show_progress else tqdm(steps, desc='Profiling features')
    for func, name in iterator:
        if show_progress:
            iterator.set_description(f"Profiling: {name}")
        start = time.perf_counter()
        try:
            df = call_with_params(func, name)
        except Exception as e:
            # record error as negative time (or store exception)
            timings[name] = {'error': str(e)}
            continue
        elapsed = time.perf_counter() - start
        timings[name] = elapsed

    if show_progress:
        # print summary
        print('\nFeature timing (seconds):')
        for k, v in timings.items():
            if isinstance(v, dict) and 'error' in v:
                print(f" - {k}: ERROR -> {v['error']}")
            else:
                print(f" - {k}: {v:.4f}s")

    return df, timings

# ===== Otro intento =====

