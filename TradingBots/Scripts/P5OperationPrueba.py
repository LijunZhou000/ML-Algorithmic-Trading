"""
P6OperationMulti.py — Bot de operación multi-activo (13 futuros)
=================================================================
Diferencias respecto a P5Operation.py (single-asset):
  • Lee best_per_asset.json para saber qué activos operar.
  • Carga modelos LSTM y GRU de 3 niveles para TODOS los activos al arranque.
  • Obtiene specs (ib_symbol, exchange, currency) vía load_specs.
  • Descarga histórico inicial de forma SECUENCIAL con sleep(4) entre símbolos.
  • Opera cada 4 horas (velas UTC: 0, 4, 8, 12, 16, 20) con ventana de gracia de 15 min.
  • En cada ciclo: predice todos → ordena por confianza → ejecuta respetando riesgo.
  • Soft-voting ponderado: el peso de cada modelo (LSTM vs GRU) se deriva del
    valor absoluto de la predicción de log-return de L3.
  • Gestión de riesgo de cartera:
      - Risk per trade   : 0.01 % del NLV (pérdida máxima asumida si salta el SL)
      - Nocional por trade: máx 10 % del NLV
      - Nocional total   : máx 50 % del NLV
"""

# ==================== IMPORTS ====================
import asyncio
import json
import logging
import math
import datetime
import pytz
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from ib_async import IB, Future, MarketOrder, LimitOrder, util

from tradepy.supervised.posttrain import (
    load_trading_model_3level,
)
from tradepy.data.loader import load_specs
from tradepy.data.resampler import resample_ohlcv, daily_ohlcv_cummulative

from tradepy.features.generate import generate_features

from tradepy.supervised.models import (
    BasicLSTM_L1_Move, BasicLSTM_L2_Dir, BasicLSTM_L3_Regression,
    BasicGRU_L1_Move,  BasicGRU_L2_Dir,  BasicGRU_L3_Regression,
)
from tradepy.logging.logger import log_movement, log_balance

from tradepy.paths import MODELS_DIR

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ==================== RUTAS ====================

# MODELS_DIR   = Path("../Models")          # raíz con subcarpetas por símbolo
# DATA_DIR     = Path("../Data")
# JSON_MULTI   = DATA_DIR / "posiciones_multi.json"
# JSON_LOGS    = DATA_DIR / "predicciones_multi.csv"
# SPECS_PATH   = DATA_DIR / "futuros_specs.json"
# FEATURES_CFG = DATA_DIR / "features_config.json"

# ==================== CONEXIÓN IB ====================
IB_HOST   = '127.0.0.1'
IB_PORT   = 7497
CLIENT_ID = 10            # ID único para este bot multi-activo

# ==================== SCHEDULING ====================
HOURS_4H          = {0, 4, 8, 12, 16, 20}   # horas UTC en las que se predice
GRACE_MINUTES     = 15                        # ventana de gracia tras la hora exacta
IB_REQUEST_SLEEP  = 4.0                       # segundos entre peticiones históricas al arranque
LOOP_SLEEP        = 60                        # segundos entre iteraciones del loop principal

# ==================== DATOS HISTÓRICOS ====================
HIST_DURATION = "7 D"     # 1 semana de historia para arranque
HIST_TIMEFRAME = "30 mins" # barras de 30 min  →  se resamplea a 4h en prepare_features
RESAMPLE_MIN  = 240        # minutos objetivo: 4 horas
RETURN_HORIZON = 240       # horizonte de retorno para L3 (en minutos)
LOOKBACK_DEFAULT = 48      # velas de lookback si model_params no lo especifica

# ==================== RIESGO POR CARTERA ====================
RISK_PER_TRADE_PCT  = 0.0001   # 0.01 % del NLV como pérdida máxima por trade
NOTIONAL_PER_TRADE  = 0.10     # máx nocional por trade (10 % NLV)
NOTIONAL_TOTAL      = 0.50     # máx nocional total (50 % NLV)
MAX_DRAWDOWN_PCT    = 5.0      # drawdown máximo antes de cerrar todo
MAX_DAILY_LOSS_PCT  = 2.0      # pérdida diaria máxima
MARGIN_BUFFER_USD   = 5000.0   # colchón mínimo de margen disponible

# ==================== ATR / SL / TP ====================
SL_ATR_MULT  = 1.2
TP_ATR_MULT  = 1.8
ATR_PERIOD   = 14
SL_FALLBACK  = 0.005   # fallback SL como % del precio si no hay ATR
TP_FALLBACK  = 0.010

# Señal codificada (igual que en entrenamiento)
SIGNAL_SELL = 0
SIGNAL_HOLD = 1
SIGNAL_LONG = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tick sizes por símbolo para snap_price (añade/ajusta si es necesario)
TICK_SIZE_MAP = {
    'AD':   0.00005, 'BP': 0.0001,  'CL': 0.01,     'EC': 0.00005,
    'ES':   0.25,    'GC': 0.10,    'MFXI': 0.0001,  'NG': 0.001,
    'NQ':   0.25,    'YM': 1.0,     'ZB':  0.03125,  'ZN': 0.015625,
    'ZS':   0.0025,
}

# Tick value (USD/punto) y multiplicador por símbolo
# Fuente: tabla de introducción del TFG
TICK_VALUE_MAP = {
    'AD': 5.0,    'BP': 6.25,   'CL': 10.0,    'EC': 6.25,
    'ES': 12.5,   'GC': 10.0,   'MFXI': 1.25,  'NG': 10.0,
    'NQ': 5.0,    'YM': 5.0,    'ZB': 31.25,   'ZN': 15.625,
    'ZS': 12.5,
}

MULTIPLIER_MAP = {
    'AD': 100000, 'BP': 62500,  'CL': 1000,    'EC': 125000,
    'ES': 50,     'GC': 100,    'MFXI': 12500, 'NG': 10000,
    'NQ': 20,     'YM': 5,      'ZB': 1000,    'ZN': 1000,
    'ZS': 5000,
}

# ==================== REGISTRO DE OPERACIONES ====================

def _cargar_registro() -> List[dict]:
    if not JSON_MULTI.exists():
        return []
    try:
        with open(JSON_MULTI, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _guardar_registro(registro: List[dict]):
    JSON_MULTI.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_MULTI, 'w') as f:
        json.dump(registro, f, indent=2, default=str)

def get_operaciones_abiertas(symbol: Optional[str] = None) -> List[dict]:
    reg = _cargar_registro()
    abiertas = [op for op in reg if op['estado'] == 'ABIERTA']
    if symbol:
        abiertas = [op for op in abiertas if op['symbol'] == symbol]
    return abiertas

def registrar_entrada(
    symbol: str, contract_local: str, signal: str,
    precio_entrada: float, sl: float, tp: float, atr: float,
    parent_order_id: int, sl_order_id: int, tp_order_id: int,
    num_contratos: int = 1,
) -> str:
    registro = _cargar_registro()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    idx = len([r for r in registro if r['estado'] == 'ABIERTA']) + 1
    trade_id = f"{symbol}_{ts}_{idx}"
    entrada = {
        "trade_id": trade_id, "symbol": symbol,
        "contract_local": contract_local, "signal": signal,
        "num_contratos": num_contratos,
        "precio_entrada": round(precio_entrada, 5),
        "sl": round(sl, 5), "tp": round(tp, 5),
        "atr_entrada": round(atr, 5) if not np.isnan(atr) else None,
        "hora_entrada": datetime.datetime.utcnow().isoformat(),
        "precio_max_intraop": round(precio_entrada, 5),
        "precio_min_intraop": round(precio_entrada, 5),
        "hora_cierre": None, "precio_cierre": None,
        "duracion_minutos": None, "pnl_usd": None,
        "mae_puntos": None, "mfe_puntos": None,
        "equity_al_cierre": None, "motivo_cierre": None,
        "estado": "ABIERTA", "origen": "NORMAL",
        "parent_order_id": parent_order_id,
        "sl_order_id": sl_order_id, "tp_order_id": tp_order_id,
    }
    registro.append(entrada)
    _guardar_registro(registro)
    logger.info(f"📝 [{symbol}] Registrada {trade_id} | {signal} @ {precio_entrada} | SL={sl} TP={tp}")
    return trade_id

def cerrar_operacion(
    trade_id: str, precio_cierre: float, motivo: str,
    tick_value: float = 10.0, multiplier: float = 1.0,
    equity_al_cierre: Optional[float] = None,
):
    registro = _cargar_registro()
    for op in registro:
        if op['trade_id'] != trade_id or op['estado'] != 'ABIERTA':
            continue
        op['estado'] = 'CERRADA'
        op['hora_cierre'] = datetime.datetime.utcnow().isoformat()
        op['precio_cierre'] = round(precio_cierre, 5)
        op['motivo_cierre'] = motivo
        op['equity_al_cierre'] = round(equity_al_cierre, 2) if equity_al_cierre else None
        try:
            hora_entrada = datetime.datetime.fromisoformat(op['hora_entrada'])
            op['duracion_minutos'] = round(
                (datetime.datetime.utcnow() - hora_entrada).total_seconds() / 60, 1
            )
        except Exception:
            op['duracion_minutos'] = None
        precio_entrada = op['precio_entrada']
        if multiplier > 1 and precio_entrada > precio_cierre * 10:
            precio_entrada = precio_entrada / multiplier
        direccion = 1 if op['signal'] == 'BUY' else -1
        op['pnl_usd'] = round(
            (precio_cierre - precio_entrada) * direccion * tick_value * op['num_contratos'], 2
        )
        p_max = op.get('precio_max_intraop', precio_entrada)
        p_min = op.get('precio_min_intraop', precio_entrada)
        if multiplier > 1 and p_max > precio_cierre * 10:
            p_max, p_min = p_max / multiplier, p_min / multiplier
        if op['signal'] == 'BUY':
            op['mfe_puntos'] = round(p_max - precio_entrada, 5)
            op['mae_puntos'] = round(precio_entrada - p_min, 5)
        else:
            op['mfe_puntos'] = round(precio_entrada - p_min, 5)
            op['mae_puntos'] = round(p_max - precio_entrada, 5)
        logger.info(
            f"📕 [{op['symbol']}] Cerrada {trade_id} | {motivo} | "
            f"PnL={op['pnl_usd']} USD | dur={op['duracion_minutos']}min"
        )
        break
    _guardar_registro(registro)

# ==================== HELPERS IB ====================

async def get_nlv(ib: IB) -> Optional[float]:
    try:
        summary = await ib.accountSummaryAsync()
        nlv = float(next((v.value for v in summary if v.tag == 'NetLiquidation'), 0))
        return nlv if nlv > 0 else None
    except Exception:
        return None

async def get_account_summary(ib: IB) -> dict:
    summary = await ib.accountSummaryAsync()
    return {v.tag: float(v.value) for v in summary}

def _snap_price(price: float, symbol: str) -> float:
    tick = TICK_SIZE_MAP.get(symbol.upper(), 0.01)
    factor = 1.0 / tick
    snapped = math.floor(price * factor + 0.5) / factor
    decimals = max(0, -int(math.floor(math.log10(tick))))
    return round(snapped, decimals)

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high_low   = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close  = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def normalize_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    date_col = next((c for c in ['date', 'datetime', 'time', 'timestamp'] if c in df.columns), None)
    if date_col:
        df['datetime'] = pd.to_datetime(df[date_col])
        df.set_index('datetime', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            df.index.name = 'datetime'
        except Exception:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    return df[cols].astype(float)

async def get_active_future_contract(ib: IB, symbol: str, exchange: str, currency: str):
    """Devuelve el contrato de futuros con mayor volumen para el símbolo dado."""
    template = Future(symbol=symbol, exchange=exchange, currency=currency)
    cds = await ib.reqContractDetailsAsync(template)
    if not cds:
        raise ValueError(f"No se encontraron contratos para {symbol} en {exchange}")
    contracts = [c.contract for c in cds]
    tickers = await ib.reqTickersAsync(*contracts)
    best, max_vol = None, -1
    for ticker in tickers:
        vol = ticker.volume or 0
        logger.debug(f"  {ticker.contract.localSymbol} vol={vol}")
        if vol > max_vol:
            max_vol, best = vol, ticker.contract
    return best

async def close_position(ib: IB, contract, position: float):
    if position == 0:
        return
    action = 'SELL' if position > 0 else 'BUY'
    ib.placeOrder(contract, MarketOrder(action, abs(position), tif='DAY'))
    logger.info(f"  → MarketOrder cierre: {action} {abs(position)}")

async def close_all_positions(ib: IB, asset_states: dict):
    """Cierra todas las posiciones abiertas en todos los activos."""
    for sym, state in asset_states.items():
        contract = state.get('contract')
        if contract is None:
            continue
        pos = next((p.position for p in ib.positions()
                    if p.contract.conId == contract.conId), 0)
        if pos != 0:
            await close_position(ib, contract, pos)
            for op in get_operaciones_abiertas(sym):
                cerrar_operacion(
                    op['trade_id'],
                    precio_cierre=state.get('last_price', op['precio_entrada']),
                    motivo='CIERRE_RISK_OFF',
                    tick_value=TICK_VALUE_MAP.get(sym, 10.0),
                    multiplier=MULTIPLIER_MAP.get(sym, 1.0),
                )

# ==================== RIESGO DE CARTERA ====================

async def check_risk(ib: IB, daily_start_nlv: Optional[float],
                     max_equity: Optional[float]) -> Tuple[bool, float, float]:
    """
    Devuelve (puede_operar, daily_start_nlv, max_equity).
    Para si se supera el drawdown máximo o la pérdida diaria máxima.
    """
    acct = await get_account_summary(ib)
    nlv              = acct.get('NetLiquidation', 0)
    excess_liquidity = acct.get('ExcessLiquidity', 0)
    realized_pnl     = acct.get('RealizedPnL', 0)

    if daily_start_nlv is None:
        daily_start_nlv = nlv
    if max_equity is None or nlv > max_equity:
        max_equity = nlv

    daily_pnl    = nlv - daily_start_nlv
    drawdown_pct = (max_equity - nlv) / max_equity * 100 if max_equity > 0 else 0.0
    max_daily_loss_usd = -daily_start_nlv * MAX_DAILY_LOSS_PCT / 100.0

    logger.info(
        f"Risk | NLV={nlv:,.0f} | DailyPnL={daily_pnl:+,.0f} | "
        f"DD={drawdown_pct:.2f}% | Excess={excess_liquidity:,.0f}"
    )
    if daily_pnl < max_daily_loss_usd and realized_pnl < max_daily_loss_usd:
        logger.critical("🚨 PÉRDIDA DIARIA MÁXIMA → RISK OFF")
        return False, daily_start_nlv, max_equity
    if drawdown_pct > MAX_DRAWDOWN_PCT:
        logger.critical("🚨 MAX DRAWDOWN → RISK OFF")
        return False, daily_start_nlv, max_equity
    if excess_liquidity < MARGIN_BUFFER_USD:
        logger.warning("⚠️ Margen insuficiente → no se operará este ciclo")
        return False, daily_start_nlv, max_equity
    return True, daily_start_nlv, max_equity

def calcular_contratos(
    precio: float, atr: float, nlv: float, symbol: str,
    total_notional_used: float,
) -> int:
    """
    Calcula el número de contratos a operar respetando:
      - Risk per trade = 0.01 % NLV (pérdida máxima si salta el SL)
      - Nocional por trade ≤ 10 % NLV
      - Nocional total    ≤ 50 % NLV (no se abre si ya supera el límite)
    Devuelve 0 si no cabe dentro del límite total.
    """
    tick_val  = TICK_VALUE_MAP.get(symbol, 10.0)
    tick_size = TICK_SIZE_MAP.get(symbol, 0.01)
    mult      = MULTIPLIER_MAP.get(symbol, 1.0)

    # Riesgo en USD que queremos asumir por trade
    risk_usd = nlv * RISK_PER_TRADE_PCT

    # SL en precio
    sl_dist = atr * SL_ATR_MULT if (not np.isnan(atr) and atr > 0) else precio * SL_FALLBACK
    # USD de pérdida por contrato si salta el SL
    pnl_per_contract_sl = (sl_dist / tick_size) * tick_val
    if pnl_per_contract_sl <= 0:
        return 0

    contracts_risk = max(1, int(risk_usd / pnl_per_contract_sl))

    # Nocional por contrato
    notional_per = precio * mult
    max_notional_trade  = nlv * NOTIONAL_PER_TRADE
    max_notional_total  = nlv * NOTIONAL_TOTAL
    contracts_notional  = max(1, int(max_notional_trade / notional_per))

    # Contratos que caben en el límite de exposición total
    espacio_nocional = max_notional_total - total_notional_used
    if espacio_nocional <= 0:
        logger.info(f"  [{symbol}] Límite de nocional total alcanzado → skip")
        return 0
    contracts_total = max(1, int(espacio_nocional / notional_per))

    contratos = min(contracts_risk, contracts_notional, contracts_total)
    logger.info(
        f"  [{symbol}] Sizing → risk={contracts_risk} notional={contracts_notional} "
        f"total={contracts_total} → final={contratos}"
    )
    return contratos

def nocional_total_abierto(ib: IB, asset_states: dict) -> float:
    """Suma el nocional de todas las posiciones abiertas actualmente en IB."""
    total = 0.0
    for sym, state in asset_states.items():
        contract = state.get('contract')
        if contract is None:
            continue
        pos = next((p.position for p in ib.positions()
                    if p.contract.conId == contract.conId), 0)
        last_price = state.get('last_price', 0)
        mult = MULTIPLIER_MAP.get(sym, 1.0)
        total += abs(pos) * last_price * mult
    return total

# ==================== CARGA DE MODELOS ====================

def load_asset_models(symbol: str) -> Optional[dict]:
    """
    Carga los modelos LSTM y GRU de 3 niveles para el símbolo dado.
    Devuelve None si no existe la carpeta del modelo.

    Se asume que load_trading_model_3level devuelve:
        (model_l1, model_l2, model_l3, scaler, scaler_l3, features)
    y que best_thresholds.pkl y model_params.pkl se cargan por separado.
    Ajusta si la firma de tu función difiere.
    """
    sym_lower = symbol.lower()

    for arch, (cls_l1, cls_l2, cls_l3) in [
        ('lstm', (BasicLSTM_L1_Move, BasicLSTM_L2_Dir, BasicLSTM_L3_Regression)),
        ('gru',  (BasicGRU_L1_Move,  BasicGRU_L2_Dir,  BasicGRU_L3_Regression)),
    ]:
        path = MODELS_DIR / sym_lower / f"trading_model_{arch}_3level"
        if not path.exists():
            logger.warning(f"⚠️ [{symbol}] No existe {path} — activo descartado")
            return None

    bundle = {}
    for arch, (cls_l1, cls_l2, cls_l3) in [
        ('lstm', (BasicLSTM_L1_Move, BasicLSTM_L2_Dir, BasicLSTM_L3_Regression)),
        ('gru',  (BasicGRU_L1_Move,  BasicGRU_L2_Dir,  BasicGRU_L3_Regression)),
    ]:
        sym_lower = symbol.lower()
        path = str(MODELS_DIR / sym_lower / f"trading_model_{arch}_3level")
        try:
            # Ajusta el desempaquetado si load_trading_model_3level tiene otra firma
            model_l1, model_l2, model_l3, scaler, scaler_l3, features = \
                load_trading_model_3level(cls_l1, cls_l2, cls_l3, path=path, device=DEVICE)
        except Exception as e:
            logger.error(f"❌ [{symbol}] Error cargando {arch}: {e}")
            return None

        # Cargar thresholds y params por separado (están en el mismo directorio)
        path_obj = Path(path)
        try:
            thresholds = joblib.load(path_obj / "best_thresholds.pkl")
        except Exception:
            thresholds = {'threshold_move': 0.65, 'threshold_dir': 0.50}
            logger.warning(f"  [{symbol}][{arch}] Usando thresholds por defecto: {thresholds}")

        try:
            model_params = joblib.load(path_obj / "model_params.pkl")
        except Exception:
            model_params = {'lookback': LOOKBACK_DEFAULT}
            logger.warning(f"  [{symbol}][{arch}] Usando model_params por defecto: {model_params}")

        bundle[arch] = {
            'model_l1':   model_l1,
            'model_l2':   model_l2,
            'model_l3':   model_l3,
            'scaler':     scaler,
            'scaler_l3':  scaler_l3,
            'features':   features,
            'thresholds': thresholds,
            'params':     model_params,
        }
        logger.info(f"  ✅ [{symbol}][{arch}] Modelos cargados | lookback={model_params.get('lookback')}")

    return bundle

# ==================== PREPARACIÓN DE FEATURES ====================

def prepare_features_multi(
    df: pd.DataFrame, symbol: str, tick_size: float,
) -> pd.DataFrame:
    """
    Adapta el DataFrame de barras 30-min de IB a features de 4h.
    Equivalente a prepare_features del bot single-asset pero símbolo-agnóstico.
    """
    df = df.copy()

    # Normalizar columna datetime
    if 'datetime' not in df.columns:
        df = df.reset_index()
    df['datetime'] = pd.to_datetime(df.get('datetime', df.get('date')), errors='coerce')
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)

    df['close']        = wavelet_denoising(df['close'].values.copy())
    df['openint']      = 0
    df['dtyyyymmdd']   = df['datetime'].dt.strftime('%Y%m%d').astype(int)
    df['ticker']       = symbol.upper()
    df['per']          = RESAMPLE_MIN

    df_resampled   = resample_ohlcv(df, period=f"{RESAMPLE_MIN}min")
    logger.info(f"  [{symbol}] Filas tras resample {RESAMPLE_MIN}min: {len(df_resampled)}")

    if len(df_resampled) < LOOKBACK_DEFAULT + 20:
        logger.warning(
            f"  [{symbol}] Sólo {len(df_resampled)} velas de {RESAMPLE_MIN}min — "
            "puede no ser suficiente para los modelos"
        )

    df_cumulative = daily_ohlcv_cummulative(df_resampled)

    with open(FEATURES_CFG) as f:
        config = json.load(f)
    config['global']['sampling_minutes']  = RESAMPLE_MIN
    config['global']['return_horizon_min'] = RETURN_HORIZON
    config['global']['tick_size']          = tick_size

    df_final = generate_features(df_cumulative, config_json=config, dropna_strategy='all')
    logger.info(f"  [{symbol}] Features: {len(df_final)} filas, {df_final.shape[1]} columnas")
    return df_final

# ==================== PREDICCIÓN 3-LEVEL CON WEIGHTED SOFT VOTING ====================

def generar_prediccion_multi(
    df_features: pd.DataFrame,
    models_bundle: dict,
    symbol: str,
) -> Tuple[Literal['BUY', 'SELL', 'HOLD'], float, float]:
    """
    Predicción de 3 niveles con weighted soft voting entre LSTM y GRU.
    El peso de cada modelo se calcula a partir del valor absoluto del log-return
    predicho por L3: mayor retorno predicho → mayor peso en L1 y L2.

    Devuelve (señal, confianza_dirección, log_return_l3_promedio).
    """
    # Usar el lookback del primer modelo disponible
    lookback = (
        models_bundle.get('lstm', {}).get('params', {}).get('lookback') or
        models_bundle.get('gru',  {}).get('params', {}).get('lookback') or
        LOOKBACK_DEFAULT
    )

    if len(df_features) < lookback:
        logger.warning(f"  [{symbol}] Insuficientes filas ({len(df_features)} < {lookback})")
        return 'HOLD', 0.0, 0.0

    # --- Obtener thresholds (promedio entre LSTM y GRU si difieren) ---
    thresholds_list = [
        models_bundle[k]['thresholds']
        for k in ('lstm', 'gru') if k in models_bundle
    ]
    threshold_move = np.mean([t['threshold_move'] for t in thresholds_list])
    threshold_dir  = np.mean([t['threshold_dir']  for t in thresholds_list])

    probs_move_per_arch = {}
    probs_dir_per_arch  = {}
    l3_logret_per_arch  = {}

    for arch in ('lstm', 'gru'):
        if arch not in models_bundle:
            continue
        info     = models_bundle[arch]
        features = info['features']
        scaler   = info['scaler']
        sc_l3    = info['scaler_l3']
        m_l1     = info['model_l1']
        m_l2     = info['model_l2']
        m_l3     = info['model_l3']

        missing = [f for f in features if f not in df_features.columns]
        if missing:
            logger.warning(f"  [{symbol}][{arch}] Features faltantes ({len(missing)}): {missing[:5]}…")
            return 'HOLD', 0.0, 0.0

        window = df_features[features].iloc[-lookback:].values.astype(float)
        if window.shape[0] < lookback:
            return 'HOLD', 0.0, 0.0

        nan_ratio = np.isnan(window).sum() / window.size
        if nan_ratio > 0.10:
            logger.warning(f"  [{symbol}][{arch}] {nan_ratio:.1%} NaN en ventana → HOLD")
            return 'HOLD', 0.0, 0.0

        window_scaled = scaler.transform(window)             # (lookback, features)
        X = torch.FloatTensor(window_scaled).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # L1: probabilidad de movimiento  [p_hold, p_move]
            probs_move = torch.softmax(m_l1(X), dim=1).cpu().numpy()[0]
            # L2: dirección                   [p_sell, p_buy]
            probs_dir  = torch.softmax(m_l2(X), dim=1).cpu().numpy()[0]
            # L3: log-return predicho (regresión, escalar)
            # scaler_l3 puede ser 1-D → usamos el output directamente si no hay scaler
            raw_l3 = m_l3(X).cpu().numpy().flatten()
            try:
                logret = float(sc_l3.inverse_transform(raw_l3.reshape(-1, 1))[0, 0])
            except Exception:
                logret = float(raw_l3[0])

        probs_move_per_arch[arch] = probs_move
        probs_dir_per_arch[arch]  = probs_dir
        l3_logret_per_arch[arch]  = abs(logret)          # peso = magnitud del retorno

        logger.debug(
            f"  [{symbol}][{arch}] L1_move={probs_move[1]:.3f} "
            f"L2_buy={probs_dir[1]:.3f} L3={logret:.6f}"
        )

    if not probs_move_per_arch:
        return 'HOLD', 0.0, 0.0

    # --- Weighted soft voting ---
    total_weight = sum(l3_logret_per_arch.values()) + 1e-9
    weights = {arch: w / total_weight for arch, w in l3_logret_per_arch.items()}

    avg_move = sum(weights[a] * probs_move_per_arch[a] for a in weights)  # vector
    avg_dir  = sum(weights[a] * probs_dir_per_arch[a]  for a in weights)

    prob_move = float(avg_move[1])          # P(hay movimiento)
    prob_buy  = float(avg_dir[1])           # P(long)
    prob_sell = float(avg_dir[0])           # P(short)
    avg_logret = sum(
        weights[a] * list(l3_logret_per_arch.values())[i]
        for i, a in enumerate(weights)
    )

    logger.info(
        f"  [{symbol}] Voting → move={prob_move:.3f} (thr={threshold_move:.2f}) "
        f"buy={prob_buy:.3f} sell={prob_sell:.3f} (thr_dir={threshold_dir:.2f})"
    )

    # L1: ¿hay movimiento?
    if prob_move < threshold_move:
        return 'HOLD', prob_move, 0.0

    # L2: ¿qué dirección?
    if prob_buy < threshold_dir and prob_sell < threshold_dir:
        return 'HOLD', prob_move, 0.0

    if prob_buy >= prob_sell:
        return 'BUY', prob_buy, avg_logret
    else:
        return 'SELL', prob_sell, avg_logret

# ==================== SCHEDULING ====================

def is_prediction_window(now_utc: datetime.datetime) -> bool:
    """True si estamos dentro de los GRACE_MINUTES posteriores a una vela de 4h."""
    return (now_utc.hour % 4 == 0) and (now_utc.minute < GRACE_MINUTES)

def get_4h_candle_id(now_utc: datetime.datetime) -> str:
    """Identificador único de la vela de 4h actual (ej: '2026-06-23-08')."""
    hour_slot = (now_utc.hour // 4) * 4
    return f"{now_utc.strftime('%Y-%m-%d')}-{hour_slot:02d}"

def is_market_open(contract_details, now_utc: datetime.datetime) -> bool:
    """
    Comprueba si el mercado del contrato está abierto en este momento.
    Usa tradingHours de IB. Devuelve True si no se puede determinar (conservador).
    """
    try:
        tz_id = contract_details.timeZoneId
        tz    = pytz.timezone(tz_id)
        now_ex = now_utc.replace(tzinfo=pytz.utc).astimezone(tz)
        today  = now_ex.date()

        for seg in contract_details.tradingHours.split(';'):
            seg = seg.strip()
            if not seg or ':' not in seg:
                continue
            d_str, hours = seg.split(':', 1)
            try:
                seg_date = datetime.datetime.strptime(d_str, '%Y%m%d').date()
            except Exception:
                continue
            if seg_date != today or hours.upper() == 'CLOSED':
                continue
            for rng in hours.split(','):
                rng = rng.strip()
                if '-' not in rng:
                    continue
                open_str, close_str = rng.split('-', 1)
                try:
                    open_dt  = datetime.datetime.strptime(open_str,  '%H%M').replace(
                        year=today.year, month=today.month, day=today.day, tzinfo=tz)
                    close_dt = datetime.datetime.strptime(close_str, '%H%M').replace(
                        year=today.year, month=today.month, day=today.day, tzinfo=tz)
                    if open_dt <= now_ex <= close_dt:
                        return True
                except Exception:
                    continue
    except Exception:
        pass
    return False   # fuera de horario o no se pudo determinar

# ==================== LOGGING DE PREDICCIONES ====================

def log_prediccion(
    symbol: str, signal: str, confidence: float,
    prob_move: float, prob_buy: float, prob_sell: float, logret: float,
):
    import csv
    fila = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'symbol': symbol, 'signal': signal,
        'confidence': round(confidence, 4),
        'prob_move': round(prob_move, 4),
        'prob_buy': round(prob_buy, 4),
        'prob_sell': round(prob_sell, 4),
        'logret_l3': round(logret, 6),
    }
    write_header = not JSON_LOGS.exists()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_LOGS, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fila.keys())
        if write_header:
            w.writeheader()
        w.writerow(fila)

# ==================== RECONCILIACIÓN (al arranque) ====================

async def reconciliar_simbolo(ib: IB, symbol: str, contract, tick_value: float):
    """Versión simplificada de reconciliación para un activo concreto."""
    abiertas = get_operaciones_abiertas(symbol)
    if not abiertas:
        return

    multiplier = float(contract.multiplier) if contract.multiplier else 1.0
    pos_ib = next((p for p in ib.positions() if p.contract.conId == contract.conId), None)
    qty_ib = int(pos_ib.position) if pos_ib else 0
    nlv = await get_nlv(ib)

    open_trades = await ib.reqAllOpenOrdersAsync()
    active_ids = {t.order.orderId for t in open_trades
                  if t.contract.conId == contract.conId}

    for op in abiertas:
        parent_id, sl_id, tp_id = (
            op.get('parent_order_id'), op.get('sl_order_id'), op.get('tp_order_id')
        )
        orden_activa = any(
            oid in active_ids for oid in [parent_id, sl_id, tp_id] if oid
        )
        if not orden_activa:
            cerrar_operacion(
                op['trade_id'],
                precio_cierre=op['precio_entrada'],
                motivo='CIERRE_RECONCILIACION_ARRANQUE',
                tick_value=tick_value,
                equity_al_cierre=nlv,
                multiplier=multiplier,
            )

    # Zombie check
    abiertas_post = get_operaciones_abiertas(symbol)
    qty_json = sum(op['num_contratos'] * (1 if op['signal'] == 'BUY' else -1)
                   for op in abiertas_post)
    exceso = abs(qty_json) - abs(qty_ib)
    if exceso > 0:
        logger.warning(f"[{symbol}] {exceso} trade(s) zombie → marcando cerrados")
        for op in sorted(abiertas_post, key=lambda x: x.get('hora_entrada', ''))[:exceso]:
            cerrar_operacion(
                op['trade_id'],
                precio_cierre=op['precio_entrada'],
                motivo='CIERRE_ZOMBIE',
                tick_value=tick_value,
                equity_al_cierre=nlv,
                multiplier=multiplier,
            )

# ==================== MAIN ====================

async def main():
    # ── 1. Leer activos a operar ─────────────────────────────────────────────
    best_path = MODELS_DIR / "best_per_asset.json"
    if not best_path.exists():
        raise FileNotFoundError(f"No se encuentra {best_path}")
    with open(best_path) as f:
        best_per_asset: Dict[str, dict] = json.load(f)
    symbols_to_trade = list(best_per_asset.keys())
    logger.info(f"Activos a operar ({len(symbols_to_trade)}): {symbols_to_trade}")

    # ── 2. Conectar a IB ─────────────────────────────────────────────────────
    ib = IB()
    await ib.connectAsync(IB_HOST, IB_PORT, clientId=CLIENT_ID, readonly=False)
    logger.info(f"✅ Conectado a IB (clientId={CLIENT_ID})")

    async def ensure_connected():
        if not ib.isConnected():
            await ib.connectAsync(IB_HOST, IB_PORT, clientId=CLIENT_ID, readonly=False)

    # ── 3. Cargar modelos para todos los activos ──────────────────────────────
    logger.info("Cargando modelos…")
    all_models: Dict[str, dict] = {}
    for sym in list(symbols_to_trade):
        logger.info(f"  Cargando modelos para {sym}…")
        bundle = load_asset_models(sym)
        if bundle is None:
            logger.warning(f"⚠️ [{sym}] Sin modelos — excluido de la operación")
            symbols_to_trade.remove(sym)
        else:
            all_models[sym] = bundle
    logger.info(f"✅ Modelos cargados para: {list(all_models.keys())}")

    # ── 4. Cargar specs y obtener contratos activos (secuencial + sleep) ──────
    logger.info("Obteniendo contratos y specs…")
    # asset_states guarda por símbolo: contract, contract_details, df_bars, last_price, etc.
    asset_states: Dict[str, dict] = {}

    for sym in list(symbols_to_trade):
        logger.info(f"  [{sym}] Cargando specs…")
        try:
            # load_specs devuelve dict con al menos: ib_symbol, exchange, currency, tick_size
            specs = load_specs(sym.lower())
            ib_symbol = specs.get('ib_symbol', sym)
            exchange  = specs.get('exchange', 'CME')
            currency  = specs.get('currency', 'USD')
            tick_size = specs.get('tick_size', TICK_SIZE_MAP.get(sym, 0.01))
        except Exception as e:
            logger.warning(f"  [{sym}] load_specs falló ({e}), usando defaults del mapa")
            ib_symbol = sym
            exchange  = 'CME'
            currency  = 'USD'
            tick_size = TICK_SIZE_MAP.get(sym, 0.01)

        await asyncio.sleep(IB_REQUEST_SLEEP)

        try:
            contract = await get_active_future_contract(ib, ib_symbol, exchange, currency)
        except Exception as e:
            logger.error(f"  [{sym}] Error obteniendo contrato: {e}")
            symbols_to_trade.remove(sym)
            continue

        # Contract details (para trading hours)
        try:
            cds = await ib.reqContractDetailsAsync(contract)
            contract_details = cds[0] if cds else None
        except Exception:
            contract_details = None

        asset_states[sym] = {
            'contract':         contract,
            'contract_details': contract_details,
            'specs':            specs if 'specs' in dir() else {},
            'tick_size':        tick_size,
            'df_bars':          pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']),
            'df_features':      None,
            'last_price':       0.0,
            'current_pos':      0,
            'active_sl_order':  None,
            'active_tp_order':  None,
        }
        logger.info(
            f"  ✅ [{sym}] Contrato: {contract.localSymbol} "
            f"(vence {contract.lastTradeDateOrContractMonth})"
        )
        await asyncio.sleep(IB_REQUEST_SLEEP)

    # ── 5. Descarga de histórico inicial (secuencial, sleep entre símbolos) ───
    logger.info("Descargando histórico inicial…")
    for sym, state in asset_states.items():
        contract = state['contract']
        logger.info(f"  [{sym}] Solicitando {HIST_DURATION} de barras {HIST_TIMEFRAME}…")
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract, '', HIST_DURATION, HIST_TIMEFRAME,
                'TRADES', False, 1, keepUpToDate=False
            )
            df = normalize_bars_df(util.df(bars)) if bars else pd.DataFrame()
            if not df.empty:
                df['atr'] = calculate_atr(df)
                state['df_bars'] = df
                state['last_price'] = float(df['close'].iloc[-1])
                logger.info(f"  ✅ [{sym}] {len(df)} barras cargadas")
            else:
                logger.warning(f"  ⚠️ [{sym}] Sin datos históricos")
        except Exception as e:
            logger.error(f"  [{sym}] Error descargando histórico: {e}")
        await asyncio.sleep(IB_REQUEST_SLEEP)

    # ── 6. Calcular features iniciales ───────────────────────────────────────
    logger.info("Calculando features iniciales…")
    for sym, state in asset_states.items():
        if state['df_bars'].empty:
            continue
        try:
            tick_size = state['tick_size']
            df_feat = prepare_features_multi(state['df_bars'], sym, tick_size)
            state['df_features'] = df_feat
        except Exception as e:
            logger.error(f"  [{sym}] Error calculando features: {e}")

    # ── 7. Reconciliar registro al arranque ───────────────────────────────────
    for sym, state in asset_states.items():
        await reconciliar_simbolo(
            ib, sym, state['contract'],
            tick_value=TICK_VALUE_MAP.get(sym, 10.0)
        )

    # ── 8. Suscribir barras en tiempo real ────────────────────────────────────
    logger.info("Suscribiendo barras en tiempo real…")
    live_bars: Dict[str, object] = {}
    for sym, state in asset_states.items():
        contract = state['contract']
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract, '', '1 D', HIST_TIMEFRAME,
                'TRADES', False, 1, keepUpToDate=True
            )
            live_bars[sym] = bars
        except Exception as e:
            logger.error(f"  [{sym}] Error suscribiendo tiempo real: {e}")
        await asyncio.sleep(IB_REQUEST_SLEEP)

    # ── 9. Estado del risk manager ────────────────────────────────────────────
    daily_start_nlv: Optional[float] = None
    max_equity:      Optional[float] = None
    last_candle_id:  str = ""          # ID de la última vela 4h en la que se predijo

    logger.info("🚀 Bot multi-activo iniciado. Esperando ventanas de predicción…")

    # ── 10. Loop principal ────────────────────────────────────────────────────
    try:
        while True:
            await ensure_connected()
            now_utc = datetime.datetime.utcnow()

            # 10.a Actualizar barras y precio para cada activo
            for sym, state in asset_states.items():
                bars = live_bars.get(sym)
                if bars:
                    new_data = normalize_bars_df(util.df(bars))
                    if not new_data.empty:
                        df_bars = pd.concat([state['df_bars'], new_data])
                        df_bars = df_bars[~df_bars.index.duplicated(keep='last')].sort_index()
                        df_bars['atr'] = calculate_atr(df_bars)
                        state['df_bars'] = df_bars
                        state['last_price'] = float(df_bars['close'].iloc[-1])

            # 10.b Risk check global
            can_trade, daily_start_nlv, max_equity = await check_risk(
                ib, daily_start_nlv, max_equity
            )
            if not can_trade:
                await close_all_positions(ib, asset_states)
                await asyncio.sleep(LOOP_SLEEP)
                continue

            nlv = await get_nlv(ib)
            if nlv is None:
                logger.warning("⚠️ No se pudo obtener NLV → esperando")
                await asyncio.sleep(LOOP_SLEEP)
                continue

            # 10.c Verificar si estamos en ventana de predicción (vela 4h)
            candle_id = get_4h_candle_id(now_utc)
            if not is_prediction_window(now_utc) or candle_id == last_candle_id:
                await asyncio.sleep(LOOP_SLEEP)
                continue

            logger.info(f"🕐 Ventana de predicción: {candle_id} ({now_utc.strftime('%H:%M')} UTC)")
            last_candle_id = candle_id

            # 10.d Recalcular features para todos los activos
            for sym, state in asset_states.items():
                if state['df_bars'].empty:
                    state['df_features'] = None
                    continue
                try:
                    state['df_features'] = prepare_features_multi(
                        state['df_bars'], sym, state['tick_size']
                    )
                except Exception as e:
                    logger.error(f"[{sym}] Error recalculando features: {e}")
                    state['df_features'] = None

            # 10.e Predicciones para todos los activos
            predictions: List[dict] = []

            for sym, state in asset_states.items():
                df_features = state['df_features']
                contract_details = state.get('contract_details')

                # Si el mercado está cerrado → HOLD
                if contract_details and not is_market_open(contract_details, now_utc):
                    logger.info(f"  [{sym}] Mercado cerrado → HOLD")
                    predictions.append({
                        'symbol': sym, 'signal': 'HOLD',
                        'confidence': 0.0, 'logret': 0.0,
                    })
                    continue

                if df_features is None or df_features.empty:
                    logger.warning(f"  [{sym}] Sin features → HOLD")
                    predictions.append({
                        'symbol': sym, 'signal': 'HOLD',
                        'confidence': 0.0, 'logret': 0.0,
                    })
                    continue

                signal, confidence, logret = generar_prediccion_multi(
                    df_features, all_models[sym], sym
                )
                log_prediccion(
                    sym, signal, confidence,
                    prob_move=confidence if signal != 'HOLD' else 0.0,
                    prob_buy=confidence if signal == 'BUY' else 0.0,
                    prob_sell=confidence if signal == 'SELL' else 0.0,
                    logret=logret,
                )
                logger.info(f"  [{sym}] → {signal} conf={confidence:.3f} l3={logret:.5f}")
                predictions.append({
                    'symbol': sym, 'signal': signal,
                    'confidence': confidence, 'logret': logret,
                })

            # 10.f Filtrar non-HOLD y ordenar por confianza (descendente)
            actionable = [p for p in predictions if p['signal'] != 'HOLD']
            actionable.sort(key=lambda x: x['confidence'], reverse=True)
            logger.info(
                f"Señales accionables: {len(actionable)}/{len(predictions)} "
                f"→ {[(p['symbol'], p['signal'], round(p['confidence'], 3)) for p in actionable]}"
            )

            # 10.g Ejecutar en orden de confianza respetando límites de riesgo
            total_notional_used = nocional_total_abierto(ib, asset_states)

            for pred in actionable:
                sym    = pred['symbol']
                signal = pred['signal']
                state  = asset_states[sym]
                contract = state['contract']
                last_price = state['last_price']

                if last_price <= 0:
                    logger.warning(f"  [{sym}] Precio no disponible → skip")
                    continue

                # Posición actual en IB
                current_pos = next(
                    (p.position for p in ib.positions()
                     if p.contract.conId == contract.conId), 0
                )

                # Si ya tenemos posición en la misma dirección → no abrir de nuevo
                if (signal == 'BUY' and current_pos > 0) or \
                   (signal == 'SELL' and current_pos < 0):
                    logger.info(f"  [{sym}] Ya en posición {current_pos:+.0f} → manteniendo")
                    continue

                # Si hay posición contraria → cerrar primero
                if current_pos != 0:
                    logger.info(f"  [{sym}] Reversal: cerrando posición {current_pos:+.0f}")
                    await close_position(ib, contract, current_pos)
                    for op in get_operaciones_abiertas(sym):
                        cerrar_operacion(
                            op['trade_id'], last_price, motivo='CIERRE_REVERSAL',
                            tick_value=TICK_VALUE_MAP.get(sym, 10.0),
                            multiplier=MULTIPLIER_MAP.get(sym, 1.0),
                            equity_al_cierre=nlv,
                        )
                    current_pos = 0
                    await asyncio.sleep(1)

                # ATR para SL/TP
                df_bars = state['df_bars']
                atr_val = (
                    float(df_bars['atr'].iloc[-1])
                    if 'atr' in df_bars.columns and not df_bars.empty
                    and not pd.isna(df_bars['atr'].iloc[-1])
                    and df_bars['atr'].iloc[-1] > 0
                    else float('nan')
                )

                # Sizing
                num_contracts = calcular_contratos(
                    last_price, atr_val, nlv, sym, total_notional_used
                )
                if num_contracts <= 0:
                    continue

                # SL / TP
                if not np.isnan(atr_val) and atr_val > 0:
                    sl_dist = atr_val * SL_ATR_MULT
                    tp_dist = atr_val * TP_ATR_MULT
                else:
                    sl_dist = last_price * SL_FALLBACK
                    tp_dist = last_price * TP_FALLBACK

                if signal == 'BUY':
                    sl = _snap_price(last_price - sl_dist, sym)
                    tp = _snap_price(last_price + tp_dist, sym)
                else:  # SELL
                    sl = _snap_price(last_price + sl_dist, sym)
                    tp = _snap_price(last_price - tp_dist, sym)

                action = 'BUY' if signal == 'BUY' else 'SELL'

                # Bracket order
                try:
                    bracket = ib.bracketOrder(
                        action=action,
                        quantity=num_contracts,
                        limitPrice=_snap_price(last_price, sym),
                        takeProfitPrice=tp,
                        stopLossPrice=sl,
                    )
                    parent, tp_order, sl_order = bracket
                    for o in bracket:
                        o.orderType = o.orderType if hasattr(o, 'orderType') else 'MKT'
                        o.tif = 'DAY'
                        o.transmit = True
                    parent.orderType = 'MKT'

                    ib.placeOrder(contract, parent)
                    ib.placeOrder(contract, tp_order)
                    ib.placeOrder(contract, sl_order)

                    state['active_sl_order'] = sl_order
                    state['active_tp_order'] = tp_order

                    registrar_entrada(
                        symbol=sym,
                        contract_local=contract.localSymbol,
                        signal=action,
                        precio_entrada=last_price,
                        sl=sl, tp=tp, atr=atr_val,
                        parent_order_id=parent.orderId,
                        sl_order_id=sl_order.orderId,
                        tp_order_id=tp_order.orderId,
                        num_contratos=num_contracts,
                    )

                    # Actualizar nocional usado
                    total_notional_used += num_contracts * last_price * MULTIPLIER_MAP.get(sym, 1.0)
                    logger.info(
                        f"✅ [{sym}] {action} {num_contracts} @ {last_price:.4f} "
                        f"SL={sl} TP={tp} | nocional_total={total_notional_used:,.0f}"
                    )
                    try:
                        log_movement(sym, action.lower(), last_price, num_contracts)
                    except Exception:
                        pass

                except Exception as e:
                    logger.error(f"❌ [{sym}] Error enviando orden: {e}")

            # Log de balance
            try:
                log_balance(daily_start_nlv or nlv, nlv, (nlv - (daily_start_nlv or nlv)))
            except Exception:
                pass

            logger.info(f"Ciclo completado. NLV={nlv:,.0f} | Nocional usado={total_notional_used:,.0f}")
            await asyncio.sleep(LOOP_SLEEP)

    except KeyboardInterrupt:
        logger.info("🛑 Interrupción manual — cerrando posiciones…")
        await close_all_positions(ib, asset_states)
    except Exception as e:
        import traceback
        logger.error(f"❌ Error crítico: {e}\n{traceback.format_exc()}")
    finally:
        ib.disconnect()
        logger.info("✅ Desconectado de IB")


if __name__ == "__main__":
    util.run(main())