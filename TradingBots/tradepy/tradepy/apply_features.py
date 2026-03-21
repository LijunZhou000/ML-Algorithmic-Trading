import pandas as pd

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
    # Use time-aware merge_asof to avoid lookahead: for each intraday row, attach
    # the most recent daily cumulative features whose timestamp <= intraday timestamp.
    left = df_5_min_features.copy()
    right = df_dia_cum_features.copy()

    # ensure datetime and sorting
    left['datetime'] = pd.to_datetime(left['datetime'])
    right['datetime'] = pd.to_datetime(right['datetime'])
    left = left.sort_values('datetime')
    right = right.sort_values('datetime')

    # perform asof merge (left rows keep their timestamps, right provides last-known daily state)
    df_merged = pd.merge_asof(left, right, on='datetime', direction='backward', suffixes=('','_daily'))

    # keep both sets of columns: intraday columns stay, daily cumulative columns have '_daily' suffix
    # This avoids accidentally using future daily aggregates for intraday rows.
    return df_merged