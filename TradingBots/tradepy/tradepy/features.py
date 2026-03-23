import talib as ta
import numpy as np
import pandas as pd
import json
import time
from tqdm import tqdm
from collections import OrderedDict

# Media movil simple
def sma(df, n=20):
    df = df.copy()
    df[f"sma_{n}"] = df['close'].rolling(n).mean()
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
    df['bb_middle'] = df['close'].rolling(n).mean()
    df['bb_std'] = df['close'].rolling(n).std()
    df['bb_upper'] = df['bb_middle'] + num_std_dev * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std_dev * df['bb_std']
    return df

# Rango verdadero medio
def true_range(df, window=14):
    # pointwise true range (not ATR)
    df = df.copy()
    df['true_range'] = ta.TRANGE(df['high'], df['low'], df['close'])
    df['range'] = df['high'] - df['low']
    df['range_mean'] = df['range'].rolling(window).mean()
    return df

def compute_atr(df, period=14):
    # ATR computed as rolling mean of true_range
    df = df.copy()
    if 'true_range' not in df.columns:
        df = true_range(df, window=period)
    df['atr'] = df['true_range'].rolling(period).mean()
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
    include_delta=True
):
    df = df.copy()
    # OBV clásico
    df['obv'] = ta.OBV(df['close'], df['volume'])

    # OBV manual opcional
    if include_manual:
        df['obv_manual'] = (
            (df['close'].diff() > 0).astype(int) * df['volume']
            - (df['close'].diff() < 0).astype(int) * df['volume']
        ).cumsum()

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
        df[f'obv_sma_{w}'] = df['obv'].rolling(w).mean()

    # Z-score
    mean = df['obv'].rolling(zscore_window).mean()
    std = df['obv'].rolling(zscore_window).std()
    df['obv_z'] = (df['obv'] - mean) / std

    # Dirección del OBV
    if include_direction:
        df['obv_dir'] = df['obv'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # OBV relativo al volumen
    df['obv_rel'] = df['obv'] / df['volume'].rolling(relative_volume_window).sum()

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
    df['avg_volume'] = df['volume'].rolling(window=window).mean()
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
    include_direction=True
):
    df = df.copy()
    # --- CLV (Close Location Value) ---
    if include_clv:
        df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
                    (df["high"] - df["low"]).replace(0, np.nan)

    # --- A/D clásico ---
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
        df[f"ad_sma_{w}"] = df["ad"].rolling(w).mean()

    # --- Z-score (acumulación/distribución extrema) ---
    mean = df["ad"].rolling(zscore_window).mean()
    std = df["ad"].rolling(zscore_window).std()
    df["ad_z"] = (df["ad"] - mean) / std

    # --- Derivada del A/D ---
    if include_delta:
        df["ad_delta"] = df["ad"].diff()

    # --- Dirección del A/D ---
    if include_direction:
        df["ad_dir"] = df["ad"].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # --- A/D relativo al volumen reciente ---
    df["ad_rel"] = df["ad"] / df["volume"].rolling(relative_volume_window).sum()

    return df

def mfi(df, window=14):
    df = df.copy()
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    pos_sum = positive_flow.rolling(window).sum()
    neg_sum = negative_flow.rolling(window).sum()

    df['mfi'] = 100 - (100 / (1 + (pos_sum / neg_sum)))
    return df
def cmf(df, window=20):
    df = df.copy()
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low']).replace(0, np.nan)

    money_flow_volume = clv * df['volume']

    df['cmf'] = money_flow_volume.rolling(window).sum() / df['volume'].rolling(window).sum()
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
    mean = df['volume'].rolling(window).mean()
    std = df['volume'].rolling(window).std()
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
    highest = df['high'].rolling(window).max()
    lowest = df['low'].rolling(window).min()
    df['williams_r'] = (highest - df['close']) / (highest - lowest)*-100
    return df
def cmo(df, window=14):
    df = df.copy()
    diff = df['close'].diff()
    up = diff.clip(lower=0).rolling(window).sum()
    down = -diff.clip(upper=0).rolling(window).sum()
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

def percent_return_features(df, lags=None, rolling_windows=None):
    df = df.copy()
    if lags is None:
        lags = [1, 5, 10, 20]
    if rolling_windows is None:
        rolling_windows = [5, 10, 20]
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    for lag in lags:
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
    for w in rolling_windows:
        df[f'return_mean_{w}'] = df['return'].rolling(w).mean()
        df[f'return_std_{w}'] = df['return'].rolling(w).std()
        df[f'return_z_{w}'] = (df['return'] - df['return'].rolling(w).mean()) / df['return'].rolling(w).std()
    return df

def percent_above_below_ma(df, ma_windows=None):
    df = df.copy()
    if ma_windows is None:
        ma_windows = [20, 50]
    for w in ma_windows:
        ma = df['close'].rolling(w).mean()
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
        df[f'rolling_skew_{w}'] = df['return'].rolling(w).skew()
        df[f'rolling_kurt_{w}'] = df['return'].rolling(w).kurt()
    return df

def rolling_slope(df, window=10, source='close'):
    df = df.copy()
    y = df[source].values.astype(float)
    n = window

    # x fijo: 0..n-1
    x = np.arange(n, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()

    # rolling sum de y
    y_sum = pd.Series(y).rolling(n).sum().values

    # rolling sum de y*x (pero x se reinicia en cada ventana)
    yx = pd.Series(y).rolling(n).apply(lambda arr: np.dot(arr, x), raw=True).values

    # fórmula cerrada
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
        df['is_month_end'] = dt_series.dt.is_month_end
        return df
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        df['hour'] = idx.hour
        df['minute'] = idx.minute
        df['dayofweek'] = idx.dayofweek
        # is_month_end is an index property accessed via .is_month_end
        try:
            df['is_month_end'] = idx.is_month_end
        except Exception:
            df['is_month_end'] = False
        return df
    else:
        return df

def realized_volatility(df, window=20):
    # sum of squared intraday returns -> sqrt
    df = df.copy()
    ret = df['log_return'] if 'log_return' in df.columns else np.log(df['close'] / df['close'].shift(1))
    df[f'realized_vol_{window}'] = np.sqrt((ret**2).rolling(window).sum()) * np.sqrt(252 / window)
    return df

def is_doji(df, threshold=0.1):
    # small body relative to candle range
    df = df.copy()
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    df['is_doji'] = (body / rng) < threshold
    return df

def generate_features(df, config_json=None):
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

    # helper para obtener parámetros del indicador
    def p(name):
        local = cfg.get(name, {})
        return {**global_cfg, **local}  # local override

    # función auxiliar
    def _call(func, name, pass_cfg=True):
        params = p(name)
        if pass_cfg and params:
            return func(df, **params)
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
    # Normalizaciones / utilidades finales
    df = _call(atr_normalized, 'atr_normalized')
    df = _call(volatility_zscore, 'volatility_zscore')

    return df


def data_quality_report(df, future_horizon=1, verbose=True, top_n=20, clean_infs=False):
    """Run a set of data-quality checks on `df` and return a report dict.

    The function prints a concise report when `verbose=True` and always
    returns a dictionary with the computed metrics (useful for programmatic checks).
    """

    df = df.copy()
    report = {}
    n_rows, n_cols = df.shape
    report['rows'] = int(n_rows)
    report['cols'] = int(n_cols)

    # NaN fractions
    na_frac = df.isna().mean().sort_values(ascending=False)
    report['na_fraction'] = na_frac.to_dict()
    report['top_na'] = na_frac.head(top_n).to_dict()

    # Constant / zero-variance columns
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    report['constant_columns'] = constant_cols

    # Infinite values (detect TRUE infinities only). Optionally replace them with NaN.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_cols = []
    inf_counts = {}
    for c in numeric_cols:
        try:
            arr = df[c].values
            has_inf = np.isinf(arr).any()
            if has_inf:
                inf_cols.append(c)
                inf_counts[c] = int(np.isinf(arr).sum())
                if clean_infs:
                    # replace infinities with NaN in-place
                    df[c].replace([np.inf, -np.inf], np.nan, inplace=True)
        except Exception:
            continue
    report['infinite_columns'] = inf_cols
    report['infinite_counts'] = inf_counts

    # Fraction of zeros
    if numeric_cols:
        zero_frac = (df[numeric_cols] == 0).mean().sort_values(ascending=False)
        report['top_zero_fraction'] = zero_frac.head(top_n).to_dict()
    else:
        report['top_zero_fraction'] = {}

    # Object/category uniques
    obj_uniques = df.select_dtypes(include=['object', 'category']).apply(lambda s: int(s.nunique(dropna=False))).to_dict()
    report['object_unique_counts'] = obj_uniques

    # Basic describe summary (reduced)
    try:
        desc = df.describe().T[['min', '25%', '50%', '75%', 'max']]
        report['describe'] = desc.to_dict(orient='index')
    except Exception:
        report['describe'] = {}

    # Correlation with future returns (if close exists)
    top_corr = {}
    if 'close' in df.columns:
        future = df['close'].pct_change(-int(future_horizon))
        corrs = {}
        for c in numeric_cols:
            try:
                mask = df[c].notna() & future.notna()
                if mask.sum() > 2:
                    corr = df.loc[mask, c].corr(future[mask])
                    if np.isfinite(corr):
                        corrs[c] = abs(float(corr))
            except Exception:
                continue
        if corrs:
            s = pd.Series(corrs).sort_values(ascending=False)
            top_corr = s.head(top_n).to_dict()
    report['top_corr_with_future'] = top_corr

    # Daily-like columns NaN fraction
    daily_cols = [c for c in df.columns if '_daily' in c]
    daily_na = {c: float(df[c].isna().mean()) for c in daily_cols}
    report['daily_columns_na_fraction'] = daily_na

    # Suggestions
    suggestions = []
    if len(na_frac) and na_frac.max() > 0.5:
        suggestions.append('Many columns have >50% NaNs; consider dropping or re-engineering them.')
    if constant_cols:
        suggestions.append(f'Drop or fix constant columns: {constant_cols[:10]}')
    if inf_cols:
        suggestions.append(f'Columns contain true infinities (use clean_infs=True to replace): {inf_cols[:20]}')
    if any(c.startswith('slope_') for c in df.columns):
        suggestions.append('`rolling_slope` may be slow; consider vectorized replacement or numba.')
    suggestions.append('To avoid fragmentation, build new columns in a dict and `pd.concat` once per group.')
    report['suggestions'] = suggestions

    if verbose:
        print('Data Quality Report')
        print(f' - rows: {n_rows}, cols: {n_cols}')
        print(' - top NA columns (fraction):')
        for k, v in list(report['top_na'].items())[:top_n]:
            print(f'    {k}: {v:.3g}')
        print(f' - constant columns: {len(constant_cols)}')
        if top_corr:
            print(' - top abs correlation with future returns:')
            for k, v in list(top_corr.items())[:min(10, len(top_corr))]:
                print(f'    {k}: {v:.3f}')
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

