import pandas as pd

def resample_ohlcv(df, period="4H", include_incomplete=True):
    """
    Resamplea un dataframe OHLCV al periodo deseado.

    Parameters
    ----------
    df : DataFrame con columnas [datetime, open, high, low, close, volume...]
    period : str (ej: '1H', '4H', '1D')
    include_incomplete : bool
        - True  → incluye la última vela (aunque esté en formación)
        - False → solo velas cerradas
    """

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("datetime").copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "openint": "last",
    }

    # Resample correcto para trading
    df_resampled = df.resample(
        period,
        label="right",
        closed="left"
    ).agg(ohlc_dict)

    # Eliminar velas vacías
    df_resampled = df_resampled.dropna(subset=["open", "high", "low", "close"])

    # Control de vela incompleta
    if not include_incomplete and len(df_resampled) > 0:
        df_resampled = df_resampled.iloc[:-1]

    # Añadir columnas extra si existen
    if "ticker" in df.columns:
        df_resampled["ticker"] = df["ticker"].iloc[0]

    df_resampled["per"] = period

    return df_resampled.reset_index()

def daily_ohlcv_cummulative(df_min):
    df = df_min.copy()
    df = df.sort_values('datetime')
    # Usamos el desplazamiento de 1 hora que definimos en la limpieza
    df['trading_date'] = (df['datetime'] - pd.Timedelta(hours=1)).dt.date
    
    # Ahora agrupamos por 'trading_date' en lugar de 'date'
    df['open_day'] = df.groupby('trading_date')['open'].transform('first')
    df['high_cum'] = df.groupby('trading_date')['high'].cummax()
    df['low_cum'] = df.groupby('trading_date')['low'].cummin()
    df['volume_cum'] = df.groupby('trading_date')['volume'].cumsum()
    
    return df