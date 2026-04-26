"""
to_parquet.py
-------------
Conversión de datos históricos de futuros de txt a parquet.

Estructura esperada:
    Data/
    ├── txt/        ← archivos originales *1min.txt
    └── parquet/    ← parquet generados aquí

Uso:
    from tradepy.data.to_parquet import convert_all, convert_one

    convert_one('ES')           # convierte solo ES
    convert_all()               # convierte todos los txt disponibles
    convert_all(force=True)     # reconvierte aunque el parquet ya exista
"""

import logging
import re
from pathlib import Path

import pandas as pd

from tradepy.paths import TXT_DIR, PARQUET_DIR

log = logging.getLogger(__name__)

# ── Columnas ──────────────────────────────────────────────────────────────────
# El txt tiene cabecera con <> que se limpian al leer
COLUMN_MAP = {
    'ticker':    'ticker',
    'per':       'per',
    'dtyyyymmdd': 'date',
    'time':      'time',
    'open':      'open',
    'high':      'high',
    'low':       'low',
    'close':     'close',
    'vol':       'vol',
    'openint':   'openint',
}


def convert_one(ticker: str, force: bool = False) -> Path:
    """
    Convierte el txt de un ticker a parquet.

    Parámetros
    ----------
    ticker : str
        Clave del futuro, ej. 'ES', 'GC'.
    force : bool
        Si True, reconvierte aunque el parquet ya exista.

    Retorna la ruta del parquet generado.
    """
    # Localizar txt con regex
    pattern = re.compile(rf'^{re.escape(ticker.lower())}\d+min\.txt$', re.IGNORECASE)
    matches = [f for f in TXT_DIR.iterdir() if pattern.match(f.name)]
    if not matches:
        raise FileNotFoundError(f"No se encuentra txt para '{ticker}' en {TXT_DIR}")
    txt_path = matches[0]

    # Ruta de salida
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PARQUET_DIR / f"{ticker.lower()}1min.parquet"

    if parquet_path.exists():
        if not force:
            log.info(f"[{ticker}] Parquet ya existe, omitiendo.")
            return parquet_path
        parquet_path.unlink()
        log.info(f"[{ticker}] Parquet eliminado para reconversión.")

    log.info(f"[{ticker}] Convirtiendo {txt_path.name}...")

    # Leer csv — cabecera con <> se limpia con strip
    df = pd.read_csv(txt_path, header=0, low_memory=False)
    df.columns = [c.strip('<>').lower() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)

    # Guardar parquet
    df.to_parquet(parquet_path, index=False, compression='snappy')

    size_mb = parquet_path.stat().st_size / 1_048_576
    log.info(f"[{ticker}] Guardado en {parquet_path.name} ({size_mb:.1f} MB)")

    return parquet_path


def convert_all(force: bool = False) -> dict[str, Path]:
    """
    Convierte todos los txt disponibles en TXT_DIR a parquet.

    Detecta automáticamente todos los archivos *1min.txt presentes.

    Parámetros
    ----------
    force : bool
        Si True, reconvierte aunque el parquet ya exista.

    Retorna dict {ticker: parquet_path} para los convertidos correctamente.
    """
    pattern = re.compile(r'^([a-zA-Z]+)\d+min\.txt$', re.IGNORECASE)
    txt_files = [f for f in TXT_DIR.iterdir() if pattern.match(f.name)]

    if not txt_files:
        log.warning(f"No se encontraron archivos txt en {TXT_DIR}")
        return {}

    log.info(f"Encontrados {len(txt_files)} archivos txt.")

    results: dict[str, Path] = {}
    failed: list[str] = []

    for txt_file in sorted(txt_files):
        ticker = pattern.match(txt_file.name).group(1).upper()
        try:
            results[ticker] = convert_one(ticker, force=force)
        except Exception as e:
            log.error(f"[{ticker}] Error: {e}")
            failed.append(ticker)

    log.info(
        f"Conversión completada: {len(results)} OK · {len(failed)} fallidos"
        + (f" → {failed}" if failed else "")
    )
    return results