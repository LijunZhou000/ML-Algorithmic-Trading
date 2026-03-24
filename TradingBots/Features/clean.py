import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

def import_dataset(asset='gc1', format='min'):
    file_path = f'../Data/{asset}{format}.txt'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'vol': 'volume'
    })
    df['dtyyyymmdd'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d')
    # Asegurar que time tenga 6 dígitos (HHMMSS) y convertir a tipo time
    df['time'] = df['time'].astype(str).str.zfill(6)
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S').dt.time
    # Unir fecha y hora en una sola columna de tipo datetime
    df['datetime'] = pd.to_datetime(df['dtyyyymmdd'].astype(str) + ' ' + df['time'].astype(str))
    df = df.sort_values('datetime')
    print(f"Dataset '{asset}' imported with {len(df)} rows")
    specs = pd.read_json("../Data/futuros_specs.json")
    spec = specs[asset[:2].upper()]
    return df, spec

def clean(df, asset='gc1'):
    df = df.copy()
    # 1. Eliminar Sábados (Día 5) - Ruido absoluto detectado en el EDA
    df = df[df['datetime'].dt.weekday != 5]
    # 2. Definir el inicio de la sesión (Trading Day)
    # En tus datos el hueco es de 00:00 a 01:00. 
    # Restamos 1 hora para que la vela de la 01:05 sea la primera del "día"
    df['trading_date'] = (df['datetime'] - pd.Timedelta(hours=1)).dt.date
    # 3. Filtrar el ruido del "Break" (15 min antes y 5 min después)
    # Minutos totales del día:
    minutes = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    # Filtramos: antes de las 00:00 (cierre) y justo al abrir (01:00)
    # Esto elimina spreads gigantes que la LSTM no puede predecir
    mask_noise = (minutes >= 1425) | (minutes == 60) 
    df = df[~mask_noise]
    # 4. Eliminar duplicados y asegurar orden temporal
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
    # 5. Resetear índice para que la secuencia de la LSTM sea continua
    df = df.reset_index(drop=True)
    print(f"Limpieza {asset} completada. Filas restantes: {len(df)}")
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

def daily_ohlcv_cummulative(df_5_min):
    df = df_5_min.copy()
    df = df.sort_values('datetime')
    # Usamos el desplazamiento de 1 hora que definimos en la limpieza
    df['trading_date'] = (df['datetime'] - pd.Timedelta(hours=1)).dt.date
    
    # Ahora agrupamos por 'trading_date' en lugar de 'date'
    df['open_day'] = df.groupby('trading_date')['open'].transform('first')
    df['high_cum'] = df.groupby('trading_date')['high'].cummax()
    df['low_cum'] = df.groupby('trading_date')['low'].cummin()
    df['volume_cum'] = df.groupby('trading_date')['volume'].cumsum()
    
    return df

def add_time_features(df):
    """
    Optimizada para 30 minutos: Elimina redundancias y añade el día de la semana,
    crucial para los futuros del Oro.
    """
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. CICLO INTRADIARIO (48 bloques de 30 min)
    # ---------------------------------------------------------
    # Sustituye a la hora. Un solo ciclo de 48 pasos es perfecto para la LSTM.
    step_30m = (df['datetime'].dt.hour * 2) + (df['datetime'].dt.minute // 30)
    df['step_sin'] = np.sin(2 * np.pi * step_30m / 48)
    df['step_cos'] = np.cos(2 * np.pi * step_30m / 48)

    # ---------------------------------------------------------
    # 2. CICLO SEMANAL (NUEVO)
    # ---------------------------------------------------------
    # El Oro se comporta diferente los lunes (gaps del finde) y viernes (cierres)
    day_of_week = df['datetime'].dt.dayofweek # Lunes=0, Domingo=6
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # ---------------------------------------------------------
    # 3. SESIONES INSTITUCIONALES (Basado en UTC/GMT)
    # ---------------------------------------------------------
    # NOTA: Ajusta estos horarios si tus datos tienen otro Timezone (ej. EST)
    df['is_london'] = df['datetime'].dt.hour.isin(range(8, 17)).astype(int)
    df['is_ny'] = df['datetime'].dt.hour.isin(range(13, 22)).astype(int)
    df['is_overlap'] = ((df['is_london'] == 1) & (df['is_ny'] == 1)).astype(int)

    return df

def add_tick_features(df, tick_size=0.1): # El tick size del GC suele ser 0.1
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. MICRO-ESTRUCTURA: Ticks Absolutos (Mantenidos)
    # ---------------------------------------------------------
    df['body_ticks'] = (df['close'] - df['open']) / tick_size
    df['range_ticks'] = (df['high'] - df['low']) / tick_size
    df['upper_wick_ticks'] = (df['high'] - df[['open', 'close']].max(axis=1)) / tick_size
    df['lower_wick_ticks'] = (df[['open', 'close']].min(axis=1) - df['low']) / tick_size
    
    # ---------------------------------------------------------
    # 2. MICRO-ESTRUCTURA: Proporciones (NUEVO)
    # ---------------------------------------------------------
    # Usamos 1e-8 para evitar divisiones por cero en dojis perfectos
    df['body_pct'] = abs(df['body_ticks']) / (df['range_ticks'] + 1e-8)
    df['upper_wick_pct'] = df['upper_wick_ticks'] / (df['range_ticks'] + 1e-8)
    df['lower_wick_pct'] = df['lower_wick_ticks'] / (df['range_ticks'] + 1e-8)
    
    # Dónde cierra la vela: 1 = En el máximo (Bullish max), 0 = En el mínimo (Bearish max)
    df['close_location'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-8)
    
    # ---------------------------------------------------------
    # 3. MACRO-CONTEXTO DEL DÍA (Mantenidos)
    # ---------------------------------------------------------
    df['dist_high_cum_ticks'] = (df['high_cum'] - df['close']) / tick_size
    df['dist_low_cum_ticks'] = (df['close'] - df['low_cum']) / tick_size
    df['dist_open_day_ticks'] = (df['close'] - df['open_day']) / tick_size
    
    # ---------------------------------------------------------
    # 4. DINÁMICAS DE VOLUMEN (Modificados / NUEVO)
    # ---------------------------------------------------------
    # Esfuerzo por tick de rango total (liquidez de la vela)
    df['vol_per_tick'] = df['volume'] / (df['range_ticks'] + 1)
    
    # Esfuerzo vs Resultado Direccional: Volumen usado para mover el precio netamente (cuerpo)
    # Si hay mucho volumen y poco cuerpo, indica posible absorción o freno.
    df['vol_vs_body'] = df['volume'] / (abs(df['body_ticks']) + 1)
    
    # Porcentaje de participación: Qué peso tiene esta vela en el día actual
    df['vol_weight_intraday'] = df['volume'] / (df['volume_cum'] + 1)
    
    return df

def add_indicator_features(df, tick_size=0.1):
    df = df.copy()
    
    # Log returns de la vela actual (Momentum base)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # ---------------------------------------------------------
    # 1. TARGETS CORREGIDOS (Para DF de 30 minutos)
    # ---------------------------------------------------------
    # Shift(-1) = Próximos 30 min | Shift(-2) = Próximos 60 min
    df['target_ret_30m'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_ret_1h'] = np.log(df['close'].shift(-2) / df['close'])

    # ---------------------------------------------------------
    # 2. INDICADORES DE MOMENTUM Y TENDENCIA
    # ---------------------------------------------------------
    # RSI de 14 periodos (Cálculo real de Wilder usando EWM)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # com=13 es el equivalente a la media suavizada de 14 periodos de Wilder
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8) # Evitar división por cero
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Distancia a la Media Móvil (SMA 20 -> 10 horas de trading en 30m)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['dist_sma_20_ticks'] = (df['close'] - df['sma_20']) / tick_size
    
    # NUEVO: Volatilidad (Proxy de ATR / Desviación Estándar)
    # Vital para que la LSTM sepa si el mercado está rápido o lento
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    
    # Limpieza de columnas temporales de cálculo
    df.drop(['sma_20'], axis=1, inplace=True, errors='ignore')
    
    return df

def generate_features(df, spec):
    df = df.copy()
    
    # 1. Tiempo
    df = add_time_features(df)
    
    # 2. Ticks (usando tu spec)
    df = add_tick_features(df, spec['tick_size'])
    
    # 3. Indicadores
    df = add_indicator_features(df)
    
    # 4. Limpieza final de NaNs (creados por las medias móviles)
    df = df.dropna()
    
    return df

def final_cleanup(df):
    # Columnas que NO deben ir al modelo (solo sirven para tracking)
    drop_cols = [
        'datetime', 'ticker', 'per', 'trading_date', 
        'open', 'high', 'low', 'close', 'volume', # Usamos sus versiones en ticks/logs
        'open_day', 'high_cum', 'low_cum', 'volume_cum', # Usamos sus distancias
        'openint' # Está a 0 siempre
    ]
    
    # 1. Eliminar lo innecesario
    df_clean = df.drop(columns=drop_cols, errors='ignore')
    
    # 2. Manejo de infinitos (por si acaso en las divisiones de volumen)
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # 3. DROP NA (Aquí es donde limpias los bordes de los indicadores y targets)
    df_clean = df_clean.dropna()
    
    return df_clean

def prepare_classification_targets(df, threshold_ticks=5, tick_size=0.1):
    """
    Convierte retornos logarítmicos en clases:
    0: Sell (Caída > threshold)
    1: Neutral (Rango lateral)
    2: Buy (Subida > threshold)
    """
    # Calculamos el movimiento en ticks
    # (precio_futuro - precio_actual) / tick_size
    future_move_ticks = (np.exp(df['target_ret_30m']) - 1) * df['close'] / tick_size
    
    conditions = [
        (future_move_ticks <= -threshold_ticks), # Sell
        (future_move_ticks >= threshold_ticks)   # Buy
    ]
    choices = [0, 2] # 0=Sell, 2=Buy
    
    df['target_class'] = np.select(conditions, choices, default=1) # 1=Neutral
    return df

def create_lstm_dataset(df, features, target_col, lookback=14):
    """
    Crea el dataset para LSTM usando NumPy strides (ultra rápido).
    """
    # 1. Convertimos a NumPy array (importante para velocidad)
    feature_array = df[features].values
    target_array = df[target_col].values
    
    # 2. Calculamos las dimensiones
    num_samples = len(df) - lookback
    num_features = len(features)
    
    # 3. Magia de NumPy Strides: Creamos ventanas sin bucles
    # (samples, lookback, features)
    shape = (num_samples, lookback, num_features)
    strides = (feature_array.strides[0], feature_array.strides[0], feature_array.strides[1])
    
    X = np.lib.stride_tricks.as_strided(feature_array, shape=shape, strides=strides)
    
    # 4. El target es simplemente el valor DESPUÉS de la ventana
    y = target_array[lookback:]
    
    return X, y

def build_lstm_model_30m(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True), # Más neuronas para 30m
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dense(3, activation='softmax') 
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005), # LR más bajo para mayor estabilidad
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model