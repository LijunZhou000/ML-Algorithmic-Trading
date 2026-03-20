# from utils_features import *
import pandas as pd
import yfinance as yf
import talib as ta
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mplfinance as mpf
import json

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from arch import arch_model

import backtrader as bt

def import_dataset(asset='gc1', format='dia'):
    """
    Importa un dataset de texto y normaliza las columnas de fecha y hora.

    Parámetros:
    - asset (str): prefijo del archivo de activo (ej. 'gc1').
    - format (str): sufijo de formato de archivo (ej. 'dia' o 'min').

    El archivo esperado está en `../data/{asset}{format}.txt` y debe contener
    al menos las columnas 'DTYYYYMMDD' y 'TIME'.

    El proceso realiza:
    1. Lectura del CSV.
    2. Conversión de 'DTYYYYMMDD' a tipo datetime.
    3. Normalización de 'TIME' (relleno a 6 dígitos) y conversión a time.
    4. Creación de la columna 'DATETIME' uniendo fecha y hora.

    Retorna:
    - pd.DataFrame con las columnas originales y la nueva columna 'DATETIME'.
    """
    file_path = f'../Data/{asset}{format}.txt'
    df = pd.read_csv(file_path)
    # Normalizar nombres de columnas a minúsculas
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'vol': 'volume'
    })
    # Convertir columna de fecha (formato YYYYMMDD) a datetime
    df['dtyyyymmdd'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d')
    # Asegurar que time tenga 6 dígitos (HHMMSS) y convertir a tipo time
    df['time'] = df['time'].astype(str).str.zfill(6)
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S').dt.time
    # Unir fecha y hora en una sola columna de tipo datetime
    df['datetime'] = pd.to_datetime(df['dtyyyymmdd'].astype(str) + ' ' + df['time'].astype(str))
    df = df.sort_values('datetime')
    return df

def load_future(asset='gc1', format='all'):
    """_summary_

    Args:
        asset (str, optional): _description_. Defaults to 'gc1'.
        format (str, optional): _description_. Defaults to 'all'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if format == 'all':
        df_dia = import_dataset(asset, 'dia')
        df_min = import_dataset(asset, 'min')
    elif format == 'dia':
        df_dia = import_dataset(asset, 'dia')
    elif format == 'min':
        df_min = import_dataset(asset, 'min')
    else:
        raise ValueError("Formato no reconocido. Use 'dia', 'min' o 'all'.")
    specs = pd.read_json("../Data/futuros_specs.json")
    spec = specs[asset[:2].upper()]
    return spec, df_dia if format in ['dia', 'all'] else None, df_min if format in ['min', 'all'] else None
    

def plot_candlestick_chart(df, title="Candlestick Chart", full_dataset=False):
    """
    Dibuja un gráfico de velas (candlestick) usando `mplfinance`.

    Parámetros:
    - df (pd.DataFrame): DataFrame que debe contener las columnas
      'OPEN','HIGH','LOW','CLOSE','VOL' y la columna 'DATETIME' creada
      por `import_dataset`.
    - title (str): título del gráfico.
    - full_dataset (bool): si es False se grafican las últimas 500 filas
      con tipo 'candle'; si es True se grafica toda la serie como 'line'.

    El DataFrame se renombra y reindexa para cumplir el formato requerido
    por `mplfinance` ('Open','High','Low','Close','Volume') y luego
    se llama a `mpf.plot` con volumen y estilo 'charles'.
    """
    df_plot = df.copy()
    # Normalizar columnas a minúsculas por si acaso
    df_plot.columns = df_plot.columns.str.lower()
    # Renombrar a formato esperado por mplfinance (Title Case)
    df_plot = df_plot.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'vol': 'Volume',
        'volume': 'Volume'
    })
    # Usar 'datetime' como índice temporal para mplfinance
    df_plot = df_plot.set_index('datetime')
    df_plot = df_plot[['Open', 'High', 'Low', 'Close', 'Volume']]
    # Mostrar solo las últimas 500 filas por defecto para no sobrecargar
    if not full_dataset:
        df_plot = df_plot.tail(500)
        plot_type = 'candle'
    else:
        plot_type = 'line'
    mpf.plot(
        df_plot,
        type=plot_type,
        volume=True,
        title=title,
        style='charles',  # Estilo opcional
        figsize=(16, 8),
        tight_layout=True
    )

def calculate_atr(df, period=14):
    """ATR auxiliar para estrategia 'atr'. Usa MAYÚSCULAS de tus datos GC."""
    # Asegurar nombres en minúsculas
    df = df.copy()
    df.columns = df.columns.str.lower()
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(period).mean()

def calculate_swings(df, strategy='atr', **kwargs):
    """
    Calcula swings en DataFrame OHLCV de tus datos GC.
    Espera MAYÚSCULAS: ['OPEN','HIGH','LOW','CLOSE'] con index 'DATETIME'
    
    Parameters:
    - df: DataFrame de import_dataset() 
    - strategy: ['atr', 'pct', 'bars']
        - 'atr': ATR adaptativo (minuto, vol variable)
        - 'pct': % fijo (diario, movimientos estables)
        - 'bars': N-barras confirmación
    """
    # Verificar index datetime (de tu import_dataset). Normalizar columnas a minúsculas
    df = df.copy()
    df.columns = df.columns.str.lower()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('datetime')

    if strategy == 'atr':
        atr_period = kwargs.get('atr_period', 14)
        atr_mult = kwargs.get('atr_mult', 1.5)
        atr = calculate_atr(df, atr_period)
        
    swings = []
    direction = 0  # 0: indefinida, 1: up, -1: down
    last_swing_idx = 0
    last_swing_price = df['close'].iloc[0]
    
    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        
        # Cálculo threshold por estrategia
        if strategy == 'pct':
            threshold = kwargs.get('pct_threshold', 2.0) / 100
            reversal = abs((price - last_swing_price) / last_swing_price)

        elif strategy == 'atr':
            # NaN handling para ATR inicial
            if pd.isna(atr.iloc[i]):
                continue
            threshold = atr_mult * atr.iloc[i]
            reversal = abs(price - last_swing_price)
            
        elif strategy == 'bars':
            n_bars = kwargs.get('n_bars', 3)
            if i - last_swing_idx < n_bars:
                continue
            reversal_pct = kwargs.get('pct_threshold', 1.0) / 100
            reversal = abs((price - last_swing_price) / last_swing_price)
        
        if direction == 0:  # Primer swing
            direction = 1 if price > last_swing_price else -1
            continue
            
        # Detectar reversión
        if direction == 1 and reversal >= threshold:  # Uptrend -> HIGH
            swings.append({
                'datetime': df.index[last_swing_idx],
                'price': last_swing_price,
                'type': 'high'
            })
            last_swing_idx = i
            last_swing_price = price
            direction = -1
            
        elif direction == -1 and reversal >= threshold:  # Downtrend -> LOW
            swings.append({
                'datetime': df.index[last_swing_idx],
                'price': last_swing_price,
                'type': 'low'
            })
            last_swing_idx = i
            last_swing_price = price
            direction = 1
    
    # Último swing
    swings.append({
        'datetime': df.index[last_swing_idx],
        'price': last_swing_price,
        'type': 'high' if direction == 1 else 'low'
    })
    
    return pd.DataFrame(swings)

def add_indicators(df, fast=20, slow=50, atr_period=14):
    df = df.copy()
    df.columns = df.columns.str.lower()
    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()
    df['atr'] = calculate_atr(df, period=atr_period)
    return df.dropna()

def compute_order_levels(row, atr_mult_sl=2.0, rr_ratio=2.0):
    atr = row.get('atr', row.get('ATR'))
    price = row.get('close', row.get('CLOSE'))
    if row.get('signal', row.get('SIGNAL')) == 1:   # long
        entry = price
        stop  = entry - atr_mult_sl * atr
        tp    = entry + rr_ratio * atr_mult_sl * atr
    elif row.get('signal', row.get('SIGNAL')) == -1:  # short
        entry = price
        stop  = entry + atr_mult_sl * atr
        tp    = entry - rr_ratio * atr_mult_sl * atr
    else:
        return pd.Series({'entry': np.nan, 'stop': np.nan, 'tp': np.nan})
    return pd.Series({'entry': entry, 'stop': stop, 'tp': tp})

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

def linear_regression_model(df):
    # Regresion lineal
    df = df.copy()
    X = df.drop(columns=["close"])
    Y = df["close"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    return model

def logistic_regression_model(df):
    # Regresion logistica
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df.drop(columns=["close", "target"])
    Y = df["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    preccision = accuracy_score(Y_test, Y_pred)
    matriz_confusion = confusion_matrix(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, model.predict(X_test)[:, 1])

    print(f"Accuracy: {preccision}")
    print(f"Confusion Matrix:\n{matriz_confusion}")
    print(f"ROC AUC Score: {roc_auc}")
    return model

def arima_model(df):
    # ARIMA
    df = df.copy()
    df.reset_index()
    dates = df["date"]
    close = df["close"]
    close.index = range(len(close))
    model = ARIMA(close, order=(5, 1, 2))
    model_fit = model.fit()

    print(model_fit.summary())

    in_sample_pred = model_fit.predict(start=0, end=len(close)-1)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, close, label='Actual')
    plt.plot(dates, in_sample_pred, label='Predicted', alpha=0.7)
    plt.title('ARIMA In-Sample Prediction') 
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    return model_fit

def garch_model(df):
    # GARCH
    df = df.copy()
    df['returns'] = df['close'].pct_change().dropna()*100
    model = arch_model(df['returns'].dropna(), vol='Garch', p=1, q=1)
    model_fit = model.fit()
    
    print(model_fit.summary())
    
    in_sample_pred = model_fit.conditional_volatility
    
    oos_pred = model_fit.forecast(horizon=10)
    forecast_variance = oos_pred.variance.iloc[-1]
    forecast_volatility_oos = np.sqrt(forecast_variance)
    
    df["historical_volatility"] = df['returns'].rolling(window=20).std()
    
    future_dates = pd.date_range(start=df['datetime'].iloc[-1] + pd.Timedelta(minutes=5), periods=10, freq='5min')
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['historical_volatility'], label='Historical Volatility')
    plt.plot(future_dates, forecast_volatility_oos.values, label='Forecasted Volatility', marker='o')
    
    plt.plot(df['datetime'], in_sample_pred, label='In-Sample Volatility', alpha=0.7)
    plt.title('GARCH Volatility Forecast')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()  
    return model_fit

def sarima_model(df):
    # SARIMA
    df = df.copy()
    df.reset_index()
    dates = df["date"]
    close = df["close"]
    close.index = range(len(close))
    model = SARIMAX(close, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    
    print(model_fit.summary())
    
    in_sample_pred = model_fit.predict(start=0, end=len(close)-1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, close, label='Actual')
    plt.plot(dates, in_sample_pred, label='Predicted', alpha=0.7)
    plt.title('SARIMA In-Sample Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    return model_fit

def decision_tree_model(df):
    # Decision Tree Classifier
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df.drop(columns=["close", "target"])
    Y = df["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
    return model

def random_forest_model(df):
    # Random forests
    # df["SMA_10"]
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df.drop(columns=["close", "target"])
    Y = df["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid={
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }, cv=3, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, Y_train)

    print(f"Best Hyperparameters: {grid_search.best_params_}")

    best_rf = grid_search.best_estimator_
    Y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
    return best_rf

def lstm_model(df):
    # LSTM
    df = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(df.drop(columns=["close"]))
    X_train = []
    Y_train = []
    window_size = 60
    for i in range(window_size, len(X_scaled)):
        X_train.append(X_scaled[i-window_size:i, 0])
        Y_train.append(X_scaled[i, 0])
        
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, Y_train, epochs=50, batch_size=32)
    
    df_test = df[-window_size:].copy()
    df_total = pd.concat([df, df_test], axis=0)
    inputs = df_total[len(df_total) - len(df_test) - window_size:]['close'].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    X_test = []
    for i in range(window_size, len(inputs)):
        X_test.append(inputs[i-window_size:i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['close'], label='Actual Price')
    plt.plot(df_test['datetime'], predicted_price, label='Predicted Price', alpha=0.7)
    plt.title('LSTM Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(df_test['close'], predicted_price))
    mae = mean_absolute_error(df_test['close'], predicted_price)
    mape = mean_absolute_percentage_error(df_test['close'], predicted_price)
    
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    return model

def kmeans_clustering(df, n_clusters=3):
    # Clustering
    # Kmeans
    df = df.copy()
    X = df.drop(columns=["close"])
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    sns.scatterplot(data=df, x='rsi', y='adx', hue='cluster', palette='Set1')
    plt.title('Clustering de Indicadores Técnicos')
    plt.xlabel('RSI')
    plt.ylabel('ADX')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['close'], label='Close Price')
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['datetime'], cluster_data['close'], label=f'Cluster {cluster}', alpha=0.6)
    plt.title('Price Movement by Cluster')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    return kmeans

def dbscan_clustering(df, eps=0.5, min_samples=5):
    # DBSCAN
    df = df.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=["close"]))
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)
    
    plt.figure(figsize=(12, 6))
    noise = df[df['dbscan_cluster'] == -1]
    clusters = df[df['dbscan_cluster'] != -1]
    
    plt.scatter(noise['datetime'], noise['close'], label='Noise', color='red', alpha=0.6)
    plt.scatter(clusters['datetime'], clusters['close'], c=clusters['dbscan_cluster'], cmap='Set1', label='Clusters', alpha=0.6)
    plt.title('DBSCAN Clustering of Price Movements')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    return dbscan