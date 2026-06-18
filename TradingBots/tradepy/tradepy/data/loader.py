"""
loader.py
---------
Carga de datos históricos desde parquet y specs de futuros.

Uso:
    from tradepy.data.loader import load_future, load_all_futures

    df, spec = load_future('ES')
    futures  = load_all_futures(['ES', 'GC'])
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from tradepy.paths import PARQUET_DIR, FUTURES_STATIC, FUTURES_DYNAMIC, TIME_OFFSET

log = logging.getLogger(__name__)

# ── Specs ─────────────────────────────────────────────────────────────────────

def load_specs() -> dict:
    """
    Carga y fusiona futures_static y futures_dynamic en un único dict.
    Las keys son los tickers en mayúsculas (ES, GC, etc.).
    """
    with open(FUTURES_STATIC) as f:
        static = json.load(f)
    with open(FUTURES_DYNAMIC) as f:
        dynamic = json.load(f)

    # Fusionar por key — dynamic sobreescribe static si hay solapamiento
    specs = {}
    for ticker in static:
        specs[ticker] = {**static[ticker], **dynamic.get(ticker, {})}

    return specs


def _ticker_from_filename(filename: str) -> str:
    """Extrae el ticker del nombre del parquet usando regex. ej. 'es_1min.parquet' → 'ES'"""
    match = re.match(r'^([a-zA-Z]+)_\d+min\.parquet$', filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"No se puede extraer ticker de '{filename}'")
    return match.group(1).upper()


def _find_parquet(ticker: str) -> Path:
    """Localiza el parquet de un ticker usando regex."""
    pattern = re.compile(rf'^{re.escape(ticker.lower())}\d+min\.parquet$', re.IGNORECASE)
    matches = [f for f in PARQUET_DIR.iterdir() if pattern.match(f.name)]
    if not matches:
        raise FileNotFoundError(f"No se encuentra parquet para '{ticker}' en {PARQUET_DIR}")
    return matches[0]


# ── Carga individual ──────────────────────────────────────────────────────────

def load_future(ticker: str) -> tuple[pd.DataFrame, dict]:
    """
    Carga el parquet de un futuro y sus specs combinadas (static + dynamic).

    Parámetros
    ----------
    ticker : str
        Clave del futuro, ej. 'ES', 'GC'.

    Retorna (DataFrame, spec_dict).
    DataFrame con columnas originales y tipos correctos.
    El ajuste a UTC se hace en cleaner.py.
    """
    parquet_path = _find_parquet(ticker)

    log.info(f"[{ticker}] Cargando {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    df = df.rename(columns={
            'vol': 'volume'
        })
    # Ajustar tipos
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['time']       = df['time'].astype(str).str.zfill(6)
    df['time']       = pd.to_datetime(df['time'], format='%H%M%S').dt.time
    df['datetime']   = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['time'].astype(str)
    )
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Cargar offset del JSON
    with open(TIME_OFFSET, 'r') as f:
        offsets = json.load(f)

    ticker_u = ticker.upper()
    if ticker_u not in offsets:
        raise KeyError(f"No offset definido para '{ticker_u}' en time_offset.json")

    offset_hours = offsets[ticker_u]['time_offset']

    # Convertir a UTC
    df['datetime_utc'] = df['datetime'] - pd.to_timedelta(offset_hours, unit='h')

    # Ordenar y limpiar duplicados
    df = df.sort_values('datetime_utc').drop_duplicates(subset='datetime_utc').reset_index(drop=True)

    log.info(f"[{ticker}] {len(df):,} filas cargadas (UTC).")

    log.info(f"[{ticker}] {len(df):,} filas cargadas.")

    # Cargar specs
    specs = load_specs()
    ticker = ticker.upper()
    if ticker not in specs:
        raise KeyError(f"No se encuentra spec para '{ticker}' en los JSON de info.")
    n_before = len(df)
    df = df.sort_values('datetime').drop_duplicates(subset='datetime').reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes:
        log.warning(f"[{ticker}] {n_dupes} filas duplicadas eliminadas.")
    return df, specs[ticker]


# ── Carga múltiple ────────────────────────────────────────────────────────────

def load_all_futures(
    tickers: list[str] | None = None,
) -> dict[str, tuple[pd.DataFrame, dict]]:
    """
    Carga múltiples futuros secuencialmente.

    Parámetros
    ----------
    tickers : list[str] | None
        Lista de tickers a cargar. Si None, carga todos los parquet disponibles.

    Retorna dict {ticker: (DataFrame, spec)}.
    """
    if tickers is None:
        tickers = [
            _ticker_from_filename(f.name)
            for f in PARQUET_DIR.iterdir()
            if f.suffix == '.parquet'
        ]

    results: dict[str, tuple[pd.DataFrame, dict]] = {}
    failed: list[str] = []

    for ticker in sorted(tickers):
        try:
            results[ticker] = load_future(ticker)
        except Exception as e:
            log.error(f"[{ticker}] Error al cargar: {e}")
            failed.append(ticker)

    log.info(
        f"Carga completada: {len(results)} OK · {len(failed)} fallidos"
        + (f" → {failed}" if failed else "")
    )
    return results