import pandas as pd
import mplfinance as mpf
import numpy as np
import talib as ta

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
# Repensar la función anterior

# Rango verdadero medio
def true_range(df):
    df['atr'] = ta.TRANGE(df['high'], df['low'], df['close'])
    return df
# Repensar la función anterior

# Commodity Channel Index
def cci(df, n=20):
    df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=n)
    return df

# Volumen en balance
def obv(df):
    df['obv'] = ta.OBV(df['close'], df['volume'])
    return df
# Repensar la función anterior

# Volumen relativo
def vpt(df):
    df['vpt'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
    return df
# Repensar la función anterior

# Acumulación/distribución
def ad(df):
    df['ad'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
    return df
# Repensar la función anterior

def generate_features(df, config_json=None):
    """Genera indicadores técnicos en el DataFrame `df`.
    `config_json` puede ser un diccionario o un string JSON con configuraciones por indicador,
    por ejemplo: {'sma': {'n':20}, 'macd': {'n_fast':12,'n_slow':26}}
    Devuelve el DataFrame con nuevas columnas."""
    import json
    cfg = {}
    if config_json:
        if isinstance(config_json, str):
            try:
                cfg = json.loads(config_json)
            except Exception as e:
                raise ValueError('config_json string no es JSON válido: ' + str(e))
        elif isinstance(config_json, dict):
            cfg = config_json
        else:
            raise ValueError('config_json debe ser dict o JSON string')
    # helper para obtener parámetros (devuelve dict vacío si no hay configuración)
    def p(name):
        return cfg.get(name, {}) if cfg else {}
    # función auxiliar para llamadas seguras donde se pasan kwargs sólo si existen
    def _call(func, name, pass_cfg=True):
        params = p(name)
        if pass_cfg and params:
            return func(df, **params)
        return func(df)
    # Aplicar indicadores (pasar kwargs sólo cuando la función los acepta)
    df = df.copy()
    df = _call(sma, 'sma')
    df = _call(ema, 'ema')
    df = _call(macd, 'macd')
    df = _call(adx, 'adx')
    df = _call(rsi, 'rsi')
    df = _call(stoch, 'stoch')
    df = _call(elder_force_index, 'elder_force_index')
    df = _call(bollinger_bands, 'bollinger_bands')
    # true_range en este notebook no recibe kwargs, llamarla directamente
    df = true_range(df)
    df = _call(cci, 'cci')
    df = _call(obv, 'obv', pass_cfg=False)
    df = _call(vpt, 'vpt', pass_cfg=False)
    df = _call(ad, 'ad', pass_cfg=False)
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