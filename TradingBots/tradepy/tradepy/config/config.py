"""
config.py
---------
Fuente única de verdad para la configuración del sistema y specs de futuros.

Uso:
    from tradepy.config import load_config, load_specs, load_symbols

    cfg     = load_config()
    specs   = load_specs()
    symbols = load_symbols()
"""

import json
import logging
from typing import Optional

from tradepy.paths import SYSTEM_CONFIG, FUTURES_STATIC, FUTURES_DYNAMIC, FEATURE_CONFIG, EXCLUDE_CONFIG

log = logging.getLogger(__name__)


def load_config() -> dict:
    """
    Carga system.json y añade campos derivados para evitar repetir
    cálculos en cada módulo.

    Campos derivados añadidos:
        - return_horizon_min : sampling_minutes * return_horizon_multiplier
        - sampling_str       : '240min' para usar en pd.resample()
    """
    with open(SYSTEM_CONFIG) as f:
        cfg = json.load(f)

    cfg['return_horizon_min'] = cfg['sampling_minutes'] * cfg['return_horizon_multiplier']
    cfg['sampling_str']       = f"{cfg['sampling_minutes']}min"

    return cfg


def load_specs() -> dict:
    """
    Carga y fusiona futures_static.json y futures_dynamic.json.
    Las keys son los tickers en mayúsculas (ES, GC, etc.).
    dynamic sobreescribe static en campos compartidos (tick_size, márgenes, etc.)

    Retorna dict {ticker: {**static, **dynamic}}.
    """
    with open(FUTURES_STATIC) as f:
        static = json.load(f)

    try:
        with open(FUTURES_DYNAMIC) as f:
            dynamic = json.load(f)
    except FileNotFoundError:
        log.warning(
            f"futures_dynamic.json no encontrado en {FUTURES_DYNAMIC}. "
            "Usando solo datos estáticos. Ejecuta fetch_futures_dynamic.py para generarlo."
        )
        dynamic = {}

    return {
        ticker: {**static[ticker], **dynamic.get(ticker, {})}
        for ticker in static
    }


def load_symbols(exclude: bool = True) -> list[str]:
    """
    Devuelve la lista de tickers disponibles desde futures_static.json.
    Si exclude=True, filtra los tickers en 'excluded_symbols' de system.json.

    Parámetros
    ----------
    exclude : bool
        Si True, aplica la lista 'excluded_symbols' de system.json.

    Retorna lista de tickers en mayúsculas, ej. ['AD', 'BP', 'CL', ...]
    """
    with open(FUTURES_STATIC) as f:
        static = json.load(f)

    symbols = list(static.keys())

    if exclude:
        cfg      = load_config()
        excluded = cfg.get('excluded_symbols', [])
        if excluded:
            log.info(f"Símbolos excluidos por system.json: {excluded}")
        symbols = [s for s in symbols if s not in excluded]

    return symbols


def load_feature_config() -> dict:
    """
    Carga el archivo features.json con la configuración de las features a generar.
    """
    with open(FEATURE_CONFIG) as f:
        return json.load(f)
 
def load_exclude_config(minutes: Optional[int] = None) -> dict:
    """
    Carga el archivo exclude.json con las listas de columnas a excluir.

    Parámetros
    ----------
    minutes : int | None
        Si se proporciona, sustituye los placeholders `{MINUTES}` y `__MINUTES__`
        por este valor (útil para nombres de target como `target_ret_5`).

    Retorna
    -------
    dict
        Contenido del JSON con las sustituciones aplicadas cuando procede.
    """
    with open(EXCLUDE_CONFIG) as f:
        cfg = json.load(f)

    if minutes is None:
        return cfg

    mins = str(minutes)

    def _replace(obj):
        if isinstance(obj, str):
            return obj.replace("{MINUTES}", mins).replace("__MINUTES__", mins)
        if isinstance(obj, list):
            return [_replace(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _replace(v) for k, v in obj.items()}
        return obj

    return _replace(cfg)
if __name__ == '__main__':
    import pprint
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    print("── CONFIG ──────────────────────────────")
    pprint.pprint(load_config())

    print("\n── SYMBOLS ─────────────────────────────")
    print(load_symbols())

    print("\n── SPECS (keys) ────────────────────────")
    specs = load_specs()
    for ticker, spec in specs.items():
        print(f"  {ticker}: {list(spec.keys())}")