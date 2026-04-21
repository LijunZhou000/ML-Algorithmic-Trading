import pandas as pd
import numpy as np
import pywt
import joblib
import os
import torch

# Funciones de carga, limpieza, resampleo y análisis para futuros
def import_dataset(asset='gc1', format='min'):
    if asset != 'ibex1':
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
        if asset == 'gc1':
            df['datetime'] = df['datetime'] - pd.Timedelta(hours=3)
        else:
            df['datetime'] = df['datetime'] - pd.Timedelta(hours=2)
        print(f"Dataset '{asset}' imported with {len(df)} rows")
        specs = pd.read_json("../Data/futuros_specs.json")
        spec = specs[asset[:2].upper()]
    else:
        asset = 'mfxi1'
        file_path = f'../Data/{asset}{format}.txt'
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(r'[<>]', '', regex=True)
        df = df.rename(columns={
            'vol': 'volume',
        })
        df['dtyyyymmdd'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d')
        # Asegurar que time tenga 6 dígitos (HHMMSS) y convertir a tipo time
        df['time'] = df['time'].astype(str).str.zfill(6)
        df['time'] = pd.to_datetime(df['time'], format='%H%M%S').dt.time
        # Unir fecha y hora en una sola columna de tipo datetime
        df['datetime'] = pd.to_datetime(df['dtyyyymmdd'].astype(str) + ' ' + df['time'].astype(str))
        df = df.sort_values('datetime')
        df['datetime'] = df['datetime'] - pd.Timedelta(hours=1)
        # if asset == 'gc1':
        #     df['datetime'] = df['datetime'] - pd.Timedelta(hours=3)
        # else:
        #     df['datetime'] = df['datetime'] - pd.Timedelta(hours=2)
        print(f"Dataset '{asset}' imported with {len(df)} rows")
        specs = pd.read_json("../Data/futuros_specs.json")
        asset = "ibex1"
        spec = specs[asset[:4].upper()]
        # df["openint"] = 0  # El futuro del IBEX no tiene open interest en el dataset, así que lo rellenamos con ceros
        # df["per"] = "1min"  # Añadimos la columna de periodo para mantener consistencia con los otros datasets
        # df["ticker"] = asset  # Añadimos la columna de ticker para mantener consistencia con los otros datasets
    return df, spec

def clean(df, asset='gc1', break_hour=21):
    df = df.copy()

    # 1. Días de la semana: Lunes a viernes y domingos. Elimino los sábados
    df = df[df['datetime'].dt.weekday != 5]

    # 2. Eliminar la hora del break dinámicamente.
    df = df[df['datetime'].dt.hour != break_hour]

    # 3. Filtrar ruido alrededor del break (15 min antes y 15 min después).
    # Calculamos la hora previa y la hora posterior teniendo en cuenta el ciclo de 24h.
    prev_hour = (break_hour - 1) % 24
    next_hour = (break_hour + 1) % 24
    
    # Máscara para 15 minutos antes del cierre (ej: 22:45 a 22:59)
    mask_noise_before = (df['datetime'].dt.hour == prev_hour) & (df['datetime'].dt.minute >= 45)
    
    # Máscara para 15 minutos después de la apertura (ej: 00:00 a 00:15 o 01:00 a 01:15)
    mask_noise_after = (df['datetime'].dt.hour == next_hour) & (df['datetime'].dt.minute <= 15)
    
    # Aplicamos el filtro para quitar ese ruido
    df = df[~(mask_noise_before | mask_noise_after)]

    # 4. Definir trading day continuo.
    df['trading_date'] = (df['datetime'] + pd.Timedelta(hours=2)).dt.date

    # 5. Eliminar duplicados y asegurar orden temporal continuo.
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
    df = df.reset_index(drop=True)

    print(f"Limpieza {asset} completada. Filas restantes: {len(df)}")
    return df

def wavelet_denoising(x, wavelet='db4', level=1):
    # Descomponer la señal
    coeffs = pywt.wavedec(x, wavelet, mode='per')
    # Aplicar umbral para eliminar ruido (detalles de alta frecuencia)
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:]]
    # Reconstruir
    y = pywt.waverec(coeffs, wavelet, mode='per')

    # Ajustar longitud
    if len(y) > len(x):
        y = y[:len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)), mode='edge')

    return y

def resample_ohlcv(df, period="5min"):
    """
    Resamplea un dataframe OHLCV al periodo deseado.
    period puede ser: '1min', '5min', '15min', '30min', '1H', '1D', etc.
    """
    if df.empty:
        return pd.DataFrame()
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
        "openint": "last",
        "dtyyyymmdd": "first"
    }

    # Resample usando el periodo elegido
    df_resampled = df.resample(period, label='right', closed='left').agg(ohlc_dict)
    # df_resampled = df.resample(period, label='right', closed='right').agg(ohlc_dict)

    # Eliminar velas vacías
    df_resampled = df_resampled.dropna(subset=["open", "high", "low", "close"])
    # df_resampled = df_resampled.dropna(subset=["open", "high", "low", "close"], how='all')
    # df_resampled_cerradas = df_resampled.iloc[:-1].dropna(subset=["open", "high", "low", "close"])
    # df_resampled = pd.concat([df_resampled_cerradas, df_resampled.iloc[[-1]]]).reset_index()

    # Añadir columnas extra
    df_resampled["ticker"] = df["ticker"].iloc[0]
    df_resampled["per"] = period
    
    # Reset index
    df_resampled = df_resampled.reset_index()

    return df_resampled

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