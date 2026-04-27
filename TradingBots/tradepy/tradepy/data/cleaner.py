import numpy as np
import pywt
import pandas as pd

def add_trading_date_by_gap(df, gap_minutes=60):
    """
    Detecta cambios de sesión automáticamente usando gaps temporales.

    Parameters
    ----------
    gap_minutes : int
        Si el gap entre velas supera este valor → nueva sesión
    """

    df = df.sort_values("datetime").copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Diferencia entre timestamps
    df["time_diff"] = df["datetime"].diff()

    # Detectar gaps grandes
    df["new_session"] = df["time_diff"] > pd.Timedelta(minutes=gap_minutes)

    # Primera fila siempre nueva sesión
    df["new_session"] = df["new_session"].fillna(True)

    # Crear ID de sesión acumulado
    df["session_id"] = df["new_session"].cumsum()

    # Crear trading_date (puedes usar date o session_id)
    df["trading_date"] = df.groupby("session_id")["datetime"].transform("first").dt.date

    return df.drop(columns=["time_diff", "new_session"])

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