"""
paths.py
--------
Fuente única de verdad para todas las rutas del proyecto.
Importar desde cualquier módulo:

    from tradepy.paths import FUTURES_STATIC, LOGS_DIR, ...

ROOT apunta a TradingBots/ independientemente de desde dónde se ejecute.
"""

from pathlib import Path

# ── Raíz del proyecto ─────────────────────────────────────────────────────────
# __file__ = TradingBots/tradepy/paths.py → .parent = tradepy/ → .parent = TradingBots/
ROOT = Path(__file__).resolve().parent.parent.parent

# ── Directorios principales ───────────────────────────────────────────────────
CONFIG_DIR   = ROOT / 'Config' # Configuración de bot, de las features
DATA_DIR     = ROOT / 'Data' # Datos históricos
LOGS_DIR     = ROOT / 'Logs' # Logs de la operación
MODELS_DIR   = ROOT / 'Models' # .pkl de los modelos
NOTEBOOKS_DIR = ROOT / 'Notebooks' # Notebooks de ejecución
SCRIPTS_DIR  = ROOT / 'Scripts' # Scripts de python
INFO_DIR = ROOT / 'Info' # Información de los futuros

# ── Config ────────────────────────────────────────────────────────────────────
FUTURES_STATIC  = INFO_DIR / 'base_info.json'
FUTURES_DYNAMIC = INFO_DIR / 'dynamic_info.json'
TIME_OFFSET = INFO_DIR / 'time_offset.json'
SYSTEM_CONFIG   = CONFIG_DIR / 'system.json'
FEATURE_CONFIG   = CONFIG_DIR / 'features.json'
EXCLUDE_CONFIG   = CONFIG_DIR / 'exclude.json'

# ── Data ────────────────────────────────────────────────────────────────────
TXT_DIR     = DATA_DIR / 'txt'
PARQUET_DIR = DATA_DIR / 'parquet'
BT_DIR      = DATA_DIR / 'bt'

# ── Logs / estado dinámico ────────────────────────────────────────────────────
POSITIONS_OPEN    = LOGS_DIR / 'positions_open.json'
POSITIONS_HISTORY = LOGS_DIR / 'positions_history.json'

# ── Modelos ───────────────────────────────────────────────────────────────────
MODELS_DIR_ARTEFACTS = MODELS_DIR / 'models'   # artefactos entrenados (.h5, .pkl)

# ── Helpers ───────────────────────────────────────────────────────────────────

def model_dir(ticker: str) -> Path:
    """Devuelve la carpeta de modelos de un activo concreto, creándola si no existe."""
    path = MODELS_DIR_ARTEFACTS / ticker.upper()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs() -> None:
    """Crea todos los directorios necesarios si no existen. Llamar al arrancar el bot."""
    dirs = [CONFIG_DIR, DATA_DIR, LOGS_DIR, MODELS_DIR, MODELS_DIR_ARTEFACTS]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # Verificación rápida: imprime todas las rutas resueltas
    print(f"ROOT:             {ROOT}")
    print(f"CONFIG_DIR:       {CONFIG_DIR}")
    print(f"DATA_DIR:         {DATA_DIR}")
    print(f"LOGS_DIR:         {LOGS_DIR}")
    print(f"MODELS_DIR:       {MODELS_DIR}")
    print(f"FUTURES_STATIC:   {FUTURES_STATIC}")
    print(f"FUTURES_DYNAMIC:  {FUTURES_DYNAMIC}")
    print(f"SYSTEM_CONFIG:    {SYSTEM_CONFIG}")
    print(f"POSITIONS_OPEN:   {POSITIONS_OPEN}")
    print(f"POSITIONS_HISTORY:{POSITIONS_HISTORY}")