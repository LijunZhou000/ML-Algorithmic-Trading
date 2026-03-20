import talib as ta
import numpy as np
import pandas as pd
import json

# Media movil simple
def sma(df, n=20):
    df[f"sma_{n}"] = df['close'].rolling(n).mean()
    return df
# Media movil exponencial
def ema(df, n=20):
    df[f"ema_{n}"] = df['close'].ewm(span=n, adjust=False).mean()
    return df

# Convergencia y divergencia de medias moviles
def macd(df, n_fast=12, n_slow=26):
    df['ema_fast'] = df['close'].ewm(span=n_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=n_slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    return df

# Índice direccional medio
def adx(df, n=14):
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=n)
    return df

# Índice de fuerza relativa
def rsi(df, n=14):
    df['rsi'] = ta.RSI(df['close'], timeperiod=n)
    return df

# Estocástico
def stoch(df, n=14, slowk_period=3, slowd_period=3, slowk_matype=0, slowd_matype=0):
    df['slowk'], df['slowd'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=n, slowk_period=slowk_period, slowd_period=slowd_period, slowk_matype=slowk_matype, slowd_matype=slowd_matype)
    return df

# Índice de fuerza de elder
def elder_force_index(df, n=13):
    df['efi'] = ta.EMA((df['close'] - df['close'].shift(1)) * df['volume'], timeperiod=n)
    return df

# Bandas de bollinger
def bollinger_bands(df, n=20, num_std_dev=2):
    df['bb_middle'] = df['close'].rolling(n).mean()
    df['bb_std'] = df['close'].rolling(n).std()
    df['bb_upper'] = df['bb_middle'] + num_std_dev * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std_dev * df['bb_std']
    return df

# Rango verdadero medio
def true_range(df, window=14):
    df['atr'] = ta.TRANGE(df['high'], df['low'], df['close'])
    df["range"] = df["high"] - df["low"]
    df["range_mean"] = df["range"].rolling(window).mean()
    return df

# Commodity Channel Index
def cci(df, n=20):
    df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=n)
    return df

# Volumen en balance
def obv(
    df,
    roc_windows=[5, 10],
    smooth_windows=[10],
    zscore_window=20,
    relative_volume_window=20,
    normalize_by="range",  # "range", "close", "none"
    include_manual=False,
    include_direction=True,
    include_delta=True
):
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
    df['vpt'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
    return df

# Volumen relativo
def relative_volume(df, window=20):
    df["volumen_medio"] = df["volume"].rolling(window=window).mean()
    df["volumen_relativo"] = df["volume"] / df["volumen_medio"]
    return df

# Acumulación/distribución
def ad(
    df,
    zscore_window=20,
    roc_windows=[5, 10],
    smooth_windows=[10],
    relative_volume_window=20,
    normalize_by="range",   # "range", "close", "none"
    include_clv=True,
    include_delta=True,
    include_direction=True
):
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
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    pos_sum = positive_flow.rolling(window).sum()
    neg_sum = negative_flow.rolling(window).sum()

    df['mfi'] = 100 - (100 / (1 + (pos_sum / neg_sum)))
    return df
def cmf(df, window=20):
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low']).replace(0, np.nan)

    money_flow_volume = clv * df['volume']

    df['cmf'] = money_flow_volume.rolling(window).sum() / df['volume'].rolling(window).sum()
    return df
def vwap(df):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return df
def volume_zscore(df, window=20):
    mean = df['volume'].rolling(window).mean()
    std = df['volume'].rolling(window).std()
    df['volume_z'] = (df['volume'] - mean) / std
    return df
def volume_delta(df):
    df['volume_delta'] = df['volume'].diff()
    return df
def volume_sum(df, window=20):
    df[f'volume_sum_{window}'] = df['volume'].rolling(window).sum()
    return df
def volume_by_range(df):
    df['volume_range'] = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    return df
def volume_by_price(df):
    df['volume_price'] = df['volume'] / df['close']
    return df

def historical_volatility(df, window=20):
    returns = np.log(df['close'] / df['close'].shift(1))
    df['hv'] = returns.rolling(window).std() * np.sqrt(252)
    return df
def atr_normalized(df):
    df['atr_norm'] = df['atr'] / df['close']
    return df
def volatility_zscore(df, window=20):
    mean = df['atr'].rolling(window).mean()
    std = df['atr'].rolling(window).std()
    df['atr_z'] = (df['atr'] - mean) / std
    return df
def dema(df, span=20):
    ema = df['close'].ewm(span=span).mean()
    df['dema'] = 2*ema - ema.ewm(span=span).mean()
    return df
def kama(df, window=10, fast=2, slow=30):
    close = df['close'].values

    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, window))
    volatility = np.abs(np.diff(close))
    volatility = np.concatenate([[np.nan], volatility]).astype(float)
    volatility = pd.Series(volatility).rolling(window).sum().values

    er = np.where(volatility == 0, 0, change / volatility)

    # Smoothing Constant (SC)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # KAMA calculation (recursive)
    kama = np.zeros_like(close)
    kama[0] = close[0]

    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    df['kama'] = kama
    return df
def roc(df, window=10):
    df[f'roc_{window}'] = df['close'].pct_change(window)
    return df
def williams_r(df, window=14):
    highest = df['high'].rolling(window).max()
    lowest = df['low'].rolling(window).min()
    df['williams_r'] = (highest - df['close']) / (highest - lowest)
    return df
def cmo(df, window=14):
    diff = df['close'].diff()
    up = diff.clip(lower=0).rolling(window).sum()
    down = -diff.clip(upper=0).rolling(window).sum()
    df['cmo'] = 100 * (up - down) / (up + down)
    return df
def vol_vol_ratio(df):
    df['vol_vol_ratio'] = df['volume'] / df['atr']
    return df
def vpt_norm(df):
    df['vpt_norm'] = df['vpt'] / df['close']
    return df
def tr_direction(df):
    df['tr_dir'] = np.sign(df['true_range'].diff())
    return df
def candle_body(df):
    df['body'] = (df['close'] - df['open']).abs()
    return df
def wick_ratio(df):
    upper = df['high'] - df[['close','open']].max(axis=1)
    lower = df[['close','open']].min(axis=1) - df['low']
    df['wick_ratio'] = (upper + lower) / (df['high'] - df['low'])
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
    # Indicadores técnicos adicionales
    df = _call(cci, 'cci')
    df = _call(bollinger_bands, 'bollinger_bands')
    df = _call(elder_force_index, 'elder_force_index')
    df = _call(candle_body, 'candle_body')
    df = _call(wick_ratio, 'wick_ratio')
    df = _call(roc, 'roc')
    # Normalizaciones / utilidades finales
    df = _call(atr_normalized, 'atr_normalized')
    df = _call(volatility_zscore, 'volatility_zscore')

    return df

def resample_ohlcv(df, period="5min"):
    """
    Resamplea un dataframe OHLCV al periodo deseado.
    period puede ser: '1min', '5min', '15min', '30min', '1H', '1D', etc.
    """

    # Asegurar orden temporal
    df = df.sort_values("datetime").copy()

    # Asegurar que datetime es datetime64
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Establecer índice temporal
    df = df.set_index("datetime")

    # Diccionario OHLCV estándar
    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "openint": "last"
    }

    # Resample usando el periodo elegido
    df_resampled = df.resample(period).agg(ohlc_dict)

    # Eliminar velas vacías
    df_resampled = df_resampled.dropna(subset=["open", "high", "low", "close"])

    # Añadir columnas extra
    df_resampled["ticker"] = df["ticker"].iloc[0]
    df_resampled["per"] = period

    # Reset index
    df_resampled = df_resampled.reset_index()

    return df_resampled

# Generar datos ohlcv acumulados diarios a partir de los datos de 5 minutos
def daily_ohlcv_cummulative(df_5_min):
    df = df_5_min.copy()
    
    # Asegurar orden temporal
    df = df.sort_values('datetime')
    
    # Crear columna de día
    df['date'] = df['datetime'].dt.date
    
    # Open del día (primer valor de cada grupo)
    df['open_day'] = df.groupby('date')['open'].transform('first')
    
    # High acumulado intradía
    df['high_cum'] = df.groupby('date')['high'].cummax()
    
    # Low acumulado intradía
    df['low_cum'] = df.groupby('date')['low'].cummin()
    
    # Volume acumulado intradía
    df['volume_cum'] = df.groupby('date')['volume'].cumsum()
    
    # Open interest (último valor hasta ese momento → ya es el actual)
    df['openint_cum'] = df['openint']
    
    # (Opcional) puedes sobrescribir columnas originales
    # df['open'] = df['open_day']
    # df['high'] = df['high_cum']
    # df['low'] = df['low_cum']
    # df['volume'] = df['volume_cum']
    
    # Limpiar columnas auxiliares si quieres
    # df = df.drop(columns=['date', 'open_day', 'high_cum', 'low_cum', 'volume_cum'])
    
    return df

def prepare_accumulated_ohlcv(df):
    # 1) copia segura y ordenar
    df_acc = df.copy()
    df_acc['datetime'] = pd.to_datetime(df_acc['datetime'])
    df_acc = df_acc.sort_values('datetime')

    # 2) renombrar OHLCV originales a *_5min (para conservarlos)
    orig_ohlcv = ['open','high','low','close','volume','openint','ticker','per']
    rename_to_5min = {c: f"{c}_5min" for c in orig_ohlcv if c in df_acc.columns}
    df_acc = df_acc.rename(columns=rename_to_5min)

    # 3) mapear las columnas acumuladas a nombres estándar OHLCV
    map_accum = {
        'open_day': 'open',
        'high_cum': 'high',
        'low_cum': 'low',
        'volume_cum': 'volume',
        'openint_cum': 'openint'
    }
    df_acc = df_acc.rename(columns={k: v for k, v in map_accum.items() if k in df_acc.columns})

    # 4) asegurarse de que exista 'close' (usar la close_5min si no hay close_cum)
    if 'close' not in df_acc.columns and 'close_5min' in df_acc.columns:
        df_acc['close'] = df_acc['close_5min']

    # 5) aplicar generate_features sobre las OHLCV acumuladas
    # df_dia_cum_accum_features = generate_features(df_acc)
    return df_acc

def merge_intraday_daily(df_dia_cum_features, df_5_min_features):
    df_5_min_features = df_5_min_features.copy()
    df_dia_cum_features = df_dia_cum_features.copy()
    # columnas a excluir (ohlcv y metadata)
    exclude = ['open','high','low','close','volume','openint','ticker','per','date']

    # seleccionar sólo features útiles del 5min
    features_5min = [c for c in df_5_min_features.columns if c not in exclude and c != 'datetime']

    # join manteniendo las OHLCV acumuladas
    df_final = df_dia_cum_features.join(
        df_5_min_features.set_index('datetime')[features_5min],
        on='datetime',
        rsuffix='_5min'
    )

    # eliminar duplicados exactos (si una columna _5min es idéntica a la original)
    for col in list(df_dia_cum_features.columns):
        col5 = f"{col}_5min"
        if col5 in df_final.columns:
            try:
                if df_final[col].equals(df_final[col5]):
                    df_final.drop(columns=[col5], inplace=True)
            except Exception:
                pass

    # Opcional: revisar resultado
    # print("Columns:", df_final.columns.tolist())
    return df_final