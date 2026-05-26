import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Literal, Tuple, Optional
from attr import ib
import pandas as pd
import datetime
import pytz
import torch
import torch.nn as nn
import joblib
from ib_async import *
from load import import_dataset, clean, load_trading_model, wavelet_denoising, resample_ohlcv, daily_ohlcv_cummulative, load_trading_model_2level
from features import generate_features
import os
from ib_async import BracketOrder, MarketOrder
import numpy as np
from models_def import GoldLSTM_L1_Move, GoldLSTM_L2_Dir, GoldGRU_L1_Move, GoldGRU_L2_Dir
from dataclasses import dataclass
from tradepy.logging.logger import log_movement, log_balance


# ==================== LOGGING (preparado para Grafana + Alertmanager) ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==================== CLI ====================
parser = argparse.ArgumentParser(description="Bot intraday async PyTorch + ATR trailing + risk + scaler")
parser.add_argument('--symbol', type=str, required=True)
parser.add_argument('--exchange', type=str, required=True, help='Ej: COMEX')
parser.add_argument('--currency', type=str, default='USD')
# parser.add_argument('--expiration', type=str, default='')
parser.add_argument('--client-id', type=int, required=True)
parser.add_argument('--model-path', type=str, required=True, help='Ej: models/lstm_gc.pth')
args = parser.parse_args()

SYMBOL = args.symbol
# EXPIRATION = args.expiration
EXCHANGE = args.exchange
CURRENCY = args.currency
CLIENT_ID = args.client_id
MODEL_PATH = args.model_path
# PREDICCION_CADA_MINUTOS = 1
PREDICCION_CADA_MINUTOS = 60        # Intervalo con posición abierta (producción: 60)
PREDICCION_SIN_POSICION_MINUTOS = 15  # Intervalo sin posición (búsqueda de entrada)

# ==================== CONFIG RIESGO + ATR ====================
MAX_DAILY_LOSS_USD = -5000.0
MAX_DRAWDOWN_PCT = 3.0
MARGIN_BUFFER_USD = 2000.0

ATR_PERIOD = 14
ATR_TRAIL_MULTIPLIER = 1.5

SL_ATR_MULT = 1.2
TP_ATR_MULT = 1.8
SL_PUNTOS_FALLBACK = 20.0
TP_PUNTOS_FALLBACK = 40.0

CANTIDAD_BASE = 1
MAX_CONTRATOS = 5
SL_PUNTOS = 20.0
TP_PUNTOS = 40.0
CIERRE_VENTANA_INICIO_MIN = 15
CIERRE_VENTANA_FIN_MIN = 5
MAX_TRADE_DURATION_HORAS = 2

TIMEFRAME = '1 min'
DURACION_HISTORICO = '15 D'

JSON_POS = Path(f'../Data/posiciones_{SYMBOL}.json')
JSON_BARS = Path(f'../Data/bars_{SYMBOL}.json')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== REGISTRO DE OPERACIONES (punto 2) ====================
 
def _cargar_registro():
    """Carga el JSON de operaciones. Devuelve lista vacía si no existe o está corrupto."""
    if not JSON_POS.exists():
        return []
    try:
        with open(JSON_POS, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []
 
def _guardar_registro(registro):
    """Persiste la lista de operaciones en el JSON."""
    JSON_POS.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_POS, 'w') as f:
        json.dump(registro, f, indent=2, default=str)
 
def registrar_entrada(
    symbol: str,
    contract_local: str,
    signal: str,
    precio_entrada: float,
    sl: float,
    tp: float,
    atr: float,
    parent_order_id: int,
    sl_order_id: int,
    tp_order_id: int,
) -> str:
    """
    Añade una nueva entrada al registro JSON.
    Cada contrato (incluyendo scale-in) tiene su propio trade_id.
    Devuelve el trade_id generado.
    """
    registro = _cargar_registro()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Sufijo incremental para evitar colisiones si hay scale-in en el mismo segundo
    idx = len([r for r in registro if r['estado'] == 'ABIERTA']) + 1
    trade_id = f"{symbol}_{ts}_{idx}"
 
    entrada = {
        "trade_id": trade_id,
        "symbol": symbol,
        "contract_local": contract_local,
        "signal": signal,
        "num_contratos": 1,
        "precio_entrada": round(precio_entrada, 4),
        "sl": round(sl, 4),
        "tp": round(tp, 4),
        "atr_entrada": round(atr, 4),
        "hora_entrada": datetime.datetime.now().isoformat(),
        # ── Campos para MAE/MFE (se actualizan en el loop) ──
        "precio_max_intraop": round(precio_entrada, 4),   # high más alto visto
        "precio_min_intraop": round(precio_entrada, 4),   # low más bajo visto
        # ── Campos de cierre ──
        "hora_cierre": None,
        "precio_cierre": None,
        "duracion_minutos": None,
        "pnl_usd": None,
        "mae_puntos": None,    # Maximum Adverse Excursion
        "mfe_puntos": None,    # Maximum Favorable Excursion
        "equity_al_cierre": None,
        "motivo_cierre": None,
        "estado": "ABIERTA",
        "origen": "NORMAL",
        "parent_order_id": parent_order_id,
        "sl_order_id": sl_order_id,
        "tp_order_id": tp_order_id,
    }
    registro.append(entrada)
    _guardar_registro(registro)
    logger.info(f"📝 Operación registrada: {trade_id} | {signal} @ {precio_entrada} | SL={sl} TP={tp}")
    return trade_id

def actualizar_mae_mfe(df_bars: pd.DataFrame):
    """
    Llamar en cada ciclo del loop mientras hay operaciones abiertas.
    Actualiza precio_max_intraop y precio_min_intraop con el high/low
    de la vela actual, para poder calcular MAE y MFE al cierre.
    """
    if df_bars.empty:
        return
    abiertas = get_operaciones_abiertas()
    if not abiertas:
        return

    high_actual = float(df_bars['high'].iloc[-1])
    low_actual  = float(df_bars['low'].iloc[-1])

    registro = _cargar_registro()
    modificado = False
    for op in registro:
        if op['estado'] != 'ABIERTA':
            continue
        if high_actual > op.get('precio_max_intraop', op['precio_entrada']):
            op['precio_max_intraop'] = round(high_actual, 4)
            modificado = True
        if low_actual < op.get('precio_min_intraop', op['precio_entrada']):
            op['precio_min_intraop'] = round(low_actual, 4)
            modificado = True

    if modificado:
        _guardar_registro(registro)

def cerrar_operacion(
    trade_id: str,
    precio_cierre: float,
    motivo: str,
    tick_value: float = 100.0,
    equity_al_cierre: Optional[float] = None,
    multiplier: float = 1.0,  # <-- nuevo parámetro, pasar contract.multiplier para GBP
):
    """
    Marca una operación como cerrada y calcula:
    - PnL en USD
    - Duración en minutos
    - MAE (Maximum Adverse Excursion) en puntos
    - MFE (Maximum Favorable Excursion) en puntos
    - Equity de la cuenta en el momento del cierre

    tick_value: USD por punto (GC=$100, SI=$50, GBP=$62500, etc.)
    multiplier: multiplicador del contrato, para normalizar precio_entrada si viene de avgCost
    """
    registro = _cargar_registro()
    for op in registro:
        if op['trade_id'] != trade_id or op['estado'] != 'ABIERTA':
            continue

        op['estado'] = 'CERRADA'
        op['hora_cierre'] = datetime.datetime.now().isoformat()
        op['precio_cierre'] = round(precio_cierre, 4)
        op['motivo_cierre'] = motivo
        op['equity_al_cierre'] = round(equity_al_cierre, 2) if equity_al_cierre is not None else None

        # Duración
        try:
            hora_entrada = datetime.datetime.fromisoformat(op['hora_entrada'])
            duracion = datetime.datetime.now() - hora_entrada
            op['duracion_minutos'] = round(duracion.total_seconds() / 60, 1)
        except Exception:
            op['duracion_minutos'] = None

        # Normalizar precio_entrada si viene corrupto (avgCost en vez de precio)
        # avgCost de IB para futuros = precio * multiplier, típicamente >> 10
        precio_entrada = op['precio_entrada']
        if multiplier > 1 and precio_entrada > precio_cierre * 10:
            precio_entrada_norm = precio_entrada / multiplier
            logger.warning(
                f"⚠️ precio_entrada={precio_entrada} parece avgCost, "
                f"normalizando → {precio_entrada_norm:.4f} (÷{multiplier})"
            )
        else:
            precio_entrada_norm = precio_entrada

        # PnL
        direccion = 1 if op['signal'] == 'BUY' else -1
        op['pnl_usd'] = round(
            (precio_cierre - precio_entrada_norm) * direccion * tick_value * op['num_contratos'], 2
        )

        # MAE y MFE — usar precio_entrada normalizado y precio_max/min de barras reales
        precio_max = op.get('precio_max_intraop', precio_entrada_norm)
        precio_min = op.get('precio_min_intraop', precio_entrada_norm)

        # Si precio_max/min también están corruptos (misma escala que avgCost), normalizar
        if multiplier > 1 and precio_max > precio_cierre * 10:
            precio_max = precio_max / multiplier
        if multiplier > 1 and precio_min > precio_cierre * 10:
            precio_min = precio_min / multiplier

        if op['signal'] == 'BUY':
            op['mfe_puntos'] = round(precio_max - precio_entrada_norm, 4)
            op['mae_puntos'] = round(precio_entrada_norm - precio_min, 4)
        else:
            op['mfe_puntos'] = round(precio_entrada_norm - precio_min, 4)
            op['mae_puntos'] = round(precio_max - precio_entrada_norm, 4)

        logger.info(
            f"📕 Cerrada {trade_id} | motivo={motivo} | precio={precio_cierre} | "
            f"PnL={op['pnl_usd']} USD | dur={op['duracion_minutos']}min | "
            f"MAE={op['mae_puntos']} | MFE={op['mfe_puntos']} | "
            f"equity={op['equity_al_cierre']}"
        )
        break

    _guardar_registro(registro)
 
def get_operaciones_abiertas():
    """Devuelve solo las operaciones con estado ABIERTA."""
    return [op for op in _cargar_registro() if op['estado'] == 'ABIERTA']

async def get_nlv(ib):
    """Obtiene el Net Liquidation Value actual de la cuenta."""
    try:
        summary = await ib.accountSummaryAsync()
        nlv = float(next((v.value for v in summary if v.tag == 'NetLiquidation'), 0))
        return nlv if nlv > 0 else None
    except Exception:
        return None

# Mapa de tick sizes por símbolo (añade los tuyos si es necesario)
TICK_SIZE_MAP = {
    'GBP':    0.00005,
    'GC':     0.10,
    'SI':     0.005,
    'IBEX':   1.0,
    'IBEX35': 1.0,
    'IB':     1.0,
}

def _snap_price(price: float, symbol: str) -> float:
    """
    Redondea price al tick size del símbolo.
    GBP/CME (6B): 0.00005  |  IBEX35: 1.0  |  GC: 0.10
    Fallback: 4 decimales.
    """
    tick = TICK_SIZE_MAP.get(symbol.upper())
    if tick:
        import math
        factor = 1.0 / tick
        snapped = math.floor(price * factor + 0.5) / factor
        # Limitar decimales para evitar floating point noise
        decimals = max(0, -int(math.floor(math.log10(tick))))
        return round(snapped, decimals)
    return round(price, 4)

async def _get_fills_contrato(ib, contract) -> dict:
    """
    Devuelve un dict orderId → {'precio': float, 'side': str, 'time': datetime|None}
    con todos los fills históricos del día para este contrato.
    """
    executions = await ib.reqExecutionsAsync()
    fills = {}
    for ex in executions:
        if ex.contract.conId != contract.conId:
            continue
        oid = ex.execution.orderId
        t_str = ex.execution.time  # "20260409  21:19:29"
        dt = None
        try:
            dt = datetime.datetime.strptime(t_str.strip(), "%Y%m%d  %H:%M:%S")
        except Exception:
            pass
        fills[oid] = {
            'precio': ex.execution.avgPrice,
            'side':   ex.execution.side,   # 'BOT' / 'SLD'
            'time':   dt,
        }
    return fills

async def reconciliar_registro(ib, contract, tick_value: float = 100.0):
    """
    Al arrancar:
    1. Obtiene fills reales del día y órdenes activas.
    2. Para cada trade ABIERTO en el JSON comprueba si SL o TP ejecutó,
       determinando el motivo exacto (CIERRE_TP / CIERRE_SL / CIERRE_EXTERNO).
    3. Zombie check: si la posición neta real en IB < registros abiertos en JSON
       → marca el exceso como CERRADO provisionalmente (evita trades fantasma).
    """
    TICK_VALUE = tick_value
    abiertas = get_operaciones_abiertas()
    if not abiertas:
        logger.info("📋 Sin operaciones abiertas en el registro. Nada que reconciliar.")
        return

    # ── Datos de IB ──────────────────────────────────────────────────────────
    fills       = await _get_fills_contrato(ib, contract)
    open_trades = await ib.reqAllOpenOrdersAsync()
    active_ids  = {
        t.order.orderId for t in open_trades
        if t.contract.conId == contract.conId
    }
    nlv        = await get_nlv(ib)
    MULTIPLIER = float(contract.multiplier) if contract.multiplier else 1.0

    # Posición neta real en IB
    pos_ib = next((p for p in ib.positions() if p.contract.conId == contract.conId), None)
    qty_ib = int(pos_ib.position) if pos_ib else 0

    # ── Cerrar trades cuyas órdenes ya no están activas ──────────────────────
    cerrados_en_esta_pasada = 0
    for op in abiertas:
        parent_id = op.get('parent_order_id')
        sl_id     = op.get('sl_order_id')
        tp_id     = op.get('tp_order_id')

        orden_activa = any(
            oid in active_ids for oid in [parent_id, sl_id, tp_id] if oid
        )
        if orden_activa:
            continue  # Sigue vivo → no tocar

        # Buscar precio de cierre en fills con prioridad TP > SL > parent
        precio_cierre = None
        motivo        = 'CIERRE_EXTERNO_DETECTADO_AL_ARRANCAR'

        if tp_id and tp_id in fills:
            precio_cierre = fills[tp_id]['precio']
            motivo        = 'CIERRE_TP'
        elif sl_id and sl_id in fills:
            precio_cierre = fills[sl_id]['precio']
            motivo        = 'CIERRE_SL'
        elif parent_id and parent_id in fills:
            precio_cierre = fills[parent_id]['precio']
            motivo        = 'CIERRE_EXTERNO_POST_ENTRADA'

        if precio_cierre is None:
            precio_cierre = op['precio_entrada']
            motivo        = 'CIERRE_SIN_FILL_PROVISIONAL'
            logger.warning(
                f"⚠️ No se encontró fill para {op['trade_id']} "
                f"(SL={sl_id}, TP={tp_id}). Usando precio_entrada como provisional."
            )

        cerrar_operacion(
            op['trade_id'], precio_cierre,
            motivo=motivo,
            tick_value=TICK_VALUE,
            equity_al_cierre=nlv,
            multiplier=MULTIPLIER,
        )
        cerrados_en_esta_pasada += 1

    # ── Zombie check: JSON dice más contratos que IB ──────────────────────────
    abiertas_post = get_operaciones_abiertas()
    qty_json_post = sum(
        op['num_contratos'] * (1 if op['signal'] == 'BUY' else -1)
        for op in abiertas_post
    )
    exceso = abs(qty_json_post) - abs(qty_ib)

    if exceso > 0:
        logger.warning(
            f"⚠️ Zombie check: JSON={qty_json_post:+d} vs IB={qty_ib:+d} "
            f"→ {exceso} contrato(s) zombie. Marcando como cerrados provisionalmente."
        )
        # Cerrar los más antiguos primero
        abiertas_sorted = sorted(abiertas_post, key=lambda op: op.get('hora_entrada', ''))
        for op in abiertas_sorted[:exceso]:
            cerrar_operacion(
                op['trade_id'],
                precio_cierre=op['precio_entrada'],
                motivo='CIERRE_ZOMBIE_DESINCRONIZACION',
                tick_value=TICK_VALUE,
                equity_al_cierre=nlv,
                multiplier=MULTIPLIER,
            )

    logger.info(
        f"✅ Reconciliación completada. "
        f"Cerrados: {cerrados_en_esta_pasada + max(0, exceso)}"
    )

# ==================== CIERRE POR TIEMPO (punto 3) ====================
 
async def cerrar_operaciones_por_tiempo(ib, contract, tick_value: float = 100.0, nlv: Optional[float] = None):
    """
    Cierra con MarketOrder los contratos que llevan más de
    MAX_TRADE_DURATION_HORAS abiertos.
    """
    abiertas = get_operaciones_abiertas()
    ahora = datetime.datetime.now()
    for op in abiertas:
        hora_entrada = datetime.datetime.fromisoformat(op['hora_entrada'])
        duracion = ahora - hora_entrada
        if duracion >= datetime.timedelta(hours=MAX_TRADE_DURATION_HORAS):
            logger.info(
                f"⏰ Cierre por tiempo: {op['trade_id']} lleva {duracion} abierto. "
                "Enviando MarketOrder de cierre."
            )
            open_trades_ib = await ib.reqAllOpenOrdersAsync()
            for t in open_trades_ib:
                if t.contract.conId == contract.conId and t.order.orderId in [
                    op.get('sl_order_id'), op.get('tp_order_id')
                ]:
                    ib.cancelOrder(t.order)

            action_cierre = 'SELL' if op['signal'] == 'BUY' else 'BUY'
            ib.placeOrder(contract, MarketOrder(action_cierre, op['num_contratos'], tif='DAY'))

            # Precio provisional — se corrige en reconciliación si el fill llega después
            MULTIPLIER = float(contract.multiplier) if contract.multiplier else 1.0
            cerrar_operacion(
                op['trade_id'],
                precio_cierre=op['precio_entrada'],
                motivo='CIERRE_POR_TIEMPO_2H',
                tick_value=tick_value,
                equity_al_cierre=nlv,
                multiplier=MULTIPLIER
            )

async def detectar_huerfanas(ib, contract, df_bars: pd.DataFrame):
    """
    Detecta posiciones activas en IB que no tienen registro ABIERTO en el JSON.

    Mejoras vs versión original:
    - Descarta contratos cuyo TP o SL ya ejecutó (no registra posiciones fantasma).
    - Usa la hora real del fill de entrada para que el contador de
      MAX_TRADE_DURATION_HORAS sea correcto desde el arranque.
    - Pasa TICK_VALUE y MULTIPLIER a cerrar_operacion si detecta fills de cierre.
    """
    pos_ib = next((p for p in ib.positions() if p.contract.conId == contract.conId), None)
    if not pos_ib or pos_ib.position == 0:
        return

    cantidad_ib         = int(pos_ib.position)
    signal_ib           = 'BUY' if cantidad_ib > 0 else 'SELL'
    abiertas            = get_operaciones_abiertas()
    contratos_registrados = sum(op['num_contratos'] for op in abiertas if op['symbol'] == SYMBOL)
    contratos_huerfanos = abs(cantidad_ib) - contratos_registrados

    if contratos_huerfanos <= 0:
        logger.info("✅ Todas las posiciones IB tienen registro en el JSON.")
        return

    logger.warning(
        f"⚠️ Detectados {contratos_huerfanos} contrato(s) huérfano(s) para {SYMBOL} "
        f"(IB={abs(cantidad_ib)}, JSON={contratos_registrados}). Iniciando recuperación..."
    )

    # ── Datos de IB ──────────────────────────────────────────────────────────
    fills       = await _get_fills_contrato(ib, contract)
    open_trades = await ib.reqAllOpenOrdersAsync()
    active_ids  = {t.order.orderId for t in open_trades if t.contract.conId == contract.conId}

    ordenes_cont = [t for t in open_trades if t.contract.conId == contract.conId]
    sl_orders    = [t.order for t in ordenes_cont if t.order.orderType == 'STP']
    tp_orders    = [t.order for t in ordenes_cont if t.order.orderType == 'LMT']

    MULTIPLIER = float(contract.multiplier) if contract.multiplier else 1.0
    nlv        = await get_nlv(ib)

    # Precio de referencia
    if not df_bars.empty:
        precio_ref = float(df_bars['close'].iloc[-1])
    else:
        avg_cost_raw = float(pos_ib.avgCost)
        precio_ref   = avg_cost_raw / MULTIPLIER if avg_cost_raw > 10 else avg_cost_raw

    # SL/TP dinámicos de fallback
    atr_value = (
        float(df_bars['atr'].iloc[-1])
        if ('atr' in df_bars.columns and not df_bars.empty
            and not pd.isna(df_bars['atr'].iloc[-1])
            and df_bars['atr'].iloc[-1] > 0)
        else None
    )
    if atr_value:
        sl_dist = atr_value * SL_ATR_MULT
        tp_dist = atr_value * TP_ATR_MULT
    else:
        logger.warning("⚠️ ATR no disponible para huérfana, usando fallback fijo.")
        sl_dist = SL_PUNTOS_FALLBACK
        tp_dist = TP_PUNTOS_FALLBACK

    sl_calculado = _snap_price(precio_ref - sl_dist if signal_ib == 'BUY' else precio_ref + sl_dist, SYMBOL)
    tp_calculado = _snap_price(precio_ref + tp_dist if signal_ib == 'BUY' else precio_ref - tp_dist, SYMBOL)

    # IDs de orden de entrada ya vinculados a trades registrados (para no reutilizar)
    ya_usados = {op.get('parent_order_id') for op in abiertas if op.get('parent_order_id')}

    registrados = 0
    for i in range(contratos_huerfanos):
        sl_order = sl_orders[i] if i < len(sl_orders) else None
        tp_order = tp_orders[i] if i < len(tp_orders) else None

        sl_id = sl_order.orderId if sl_order else None
        tp_id = tp_order.orderId if tp_order else None

        # ── Si TP o SL ya ejecutó → esta posición ya cerró, no registrar ────
        if tp_id and tp_id in fills and tp_id not in active_ids:
            logger.info(
                f"   → Huérfana #{i+1}: TP {tp_id} ya ejecutó @ {fills[tp_id]['precio']:.5f}. "
                "No se registra (ya cerrada)."
            )
            continue
        if sl_id and sl_id in fills and sl_id not in active_ids:
            logger.info(
                f"   → Huérfana #{i+1}: SL {sl_id} ya ejecutó @ {fills[sl_id]['precio']:.5f}. "
                "No se registra (ya cerrada)."
            )
            continue

        # ── Reconstruir hora de entrada desde fill real ───────────────────
        hora_entrada_real = None
        side_entrada = 'BOT' if signal_ib == 'BUY' else 'SLD'
        candidates = [
            (oid, info) for oid, info in fills.items()
            if info['side'] == side_entrada and oid not in ya_usados
        ]
        if candidates:
            candidates.sort(key=lambda x: x[1]['time'] or datetime.datetime.min)
            oid_usado, fill_info = candidates[i] if i < len(candidates) else candidates[-1]
            hora_entrada_real = fill_info['time']
            ya_usados.add(oid_usado)

        hora_entrada_str = (
            hora_entrada_real.isoformat()
            if hora_entrada_real
            else datetime.datetime.now().isoformat()
        )
        if not hora_entrada_real:
            logger.warning(
                f"   → Huérfana #{i+1}: no se encontró fill de entrada. "
                "Usando datetime.now() — el contador de 2h puede ser incorrecto."
            )

        # ── Colocar TP si no existe ───────────────────────────────────────
        nueva_tp_order = None
        if tp_order is None:
            tp_action      = 'SELL' if signal_ib == 'BUY' else 'BUY'
            nueva_tp_order = LimitOrder(
                action=tp_action,
                totalQuantity=1,
                lmtPrice=tp_calculado,
                tif='DAY',
                transmit=True,
            )
            if sl_order:
                nueva_tp_order.parentId = sl_order.orderId
            ib.placeOrder(contract, nueva_tp_order)
            logger.info(
                f"   → TP colocado para huérfana #{i+1}: {tp_action} @ {tp_calculado} "
                f"(ATR={round(atr_value, 5) if atr_value else 'N/A'})"
            )

        # ── Precio de entrada normalizado ─────────────────────────────────
        avg_cost_raw        = float(pos_ib.avgCost)
        precio_entrada_norm = avg_cost_raw / MULTIPLIER if avg_cost_raw > 10 else avg_cost_raw

        # ── Registrar en JSON con origen='RECUPERADA' ─────────────────────
        registro = _cargar_registro()
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        idx      = len([r for r in registro if r['estado'] == 'ABIERTA']) + 1
        trade_id = f"{SYMBOL}_{ts}_{idx}"

        entrada = {
            "trade_id":           trade_id,
            "symbol":             SYMBOL,
            "contract_local":     contract.localSymbol,
            "signal":             signal_ib,
            "num_contratos":      1,
            "precio_entrada":     round(precio_entrada_norm, 5),
            "sl":                 round(sl_order.auxPrice, 5) if sl_order else round(sl_calculado, 5),
            "tp":                 round(tp_order.lmtPrice, 5) if tp_order else round(tp_calculado, 5),
            "atr_entrada":        round(atr_value, 5) if atr_value else None,
            "hora_entrada":       hora_entrada_str,       # ← hora real del fill
            "precio_max_intraop": round(precio_ref, 5),
            "precio_min_intraop": round(precio_ref, 5),
            "hora_cierre":        None,
            "precio_cierre":      None,
            "duracion_minutos":   None,
            "pnl_usd":            None,
            "mae_puntos":         None,
            "mfe_puntos":         None,
            "equity_al_cierre":   None,
            "motivo_cierre":      None,
            "estado":             "ABIERTA",
            "origen":             "RECUPERADA",
            "parent_order_id":    None,
            "sl_order_id":        sl_order.orderId if sl_order else None,
            "tp_order_id":        (nueva_tp_order.orderId if nueva_tp_order
                                   else (tp_order.orderId if tp_order else None)),
        }
        registro.append(entrada)
        _guardar_registro(registro)
        registrados += 1
        logger.info(
            f"📝 Huérfana registrada: {trade_id} | {signal_ib} | "
            f"entrada≈{precio_entrada_norm:.5f} | hora_entrada={hora_entrada_str} | "
            f"origen=RECUPERADA"
        )

    logger.info(f"✅ detectar_huerfanas completado. Registradas: {registrados}")

# Obtener el contrato futuro con más volumen para el símbolo dado
async def get_active_future_contract(ib, symbol, exchange='COMEX', currency='USD'):
    # OJO: exchange='NYMEX' y secType='FUT'
    template = Future(
        symbol=symbol,
        exchange=exchange,
        currency=currency
    )
    cds = await ib.reqContractDetailsAsync(template)
    if not cds:
        raise ValueError(f"No se encontraron contratos para {symbol}")

    contracts = [c.contract for c in cds]
    tickers = await ib.reqTickersAsync(*contracts)

    best_contract = None
    max_vol = -1
    for ticker in tickers:
        vol = ticker.volume or 0
        print(f"Contrato: {ticker.contract.localSymbol} | Vol: {vol}")
        if vol > max_vol:
            max_vol = vol
            best_contract = ticker.contract

    return best_contract

# Guardar el DataFrame con los datos usados para la predicción (para análisis posterior)
def save_live_data(df: pd.DataFrame):
    # Supongamos que df solo trae la última vela cerrada o las nuevas
    df_reset = df.reset_index()
    df_reset['datetime'] = df_reset['datetime'].astype(str)
    new_records = df_reset.to_dict('records')

    existing_data = []
    if os.path.exists(JSON_BARS):
        with open(JSON_BARS, 'r') as f:
            try:
                existing_data = json.load(f)
            except:
                existing_data = []

    # Crear un set de timestamps existentes para no duplicar
    existing_timestamps = {str(r['datetime']) for r in existing_data}

    # Solo añadir si el timestamp no está en el archivo
    for record in new_records:
        if str(record['datetime']) not in existing_timestamps:
            existing_data.append(record)

    with open(JSON_BARS, 'w') as f:
        json.dump(existing_data, f, indent=2)

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ==================== PREDICCIÓN PyTorch + SCALER ====================
def ensure_datetime_column(df: pd.DataFrame, col_name: str = 'datetime') -> pd.DataFrame:
    """Ensure column exists and is proper datetime, handling IB data formats."""
    df = df.copy()
    
    # Handle IB util.df() output
    if 'date' in df.columns and col_name not in df.columns:
        df[col_name] = pd.to_datetime(df['date'])
    
    # Handle index-based datetime
    if col_name not in df.columns:
        df = df.reset_index()
        if 'datetime' in df.columns:
            df[col_name] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df[col_name] = pd.to_datetime(df['date'])
        else:
            df[col_name] = pd.to_datetime(df.index)
    
    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
    return df

def normalize_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza un DataFrame de barras IB para que tenga index datetime y columnas OHLCV."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['open','high','low','close','volume'])
    
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Buscar la columna de fecha/tiempo
    date_col = None
    for candidate in ['date', 'datetime', 'time', 'timestamp']:
        if candidate in df.columns:
            date_col = candidate
            break
    
    if date_col:
        df['datetime'] = pd.to_datetime(df[date_col])
        df.set_index('datetime', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Intentar convertir el índice directamente
        try:
            df.index = pd.to_datetime(df.index)
            df.index.name = 'datetime'
        except Exception:
            logger.warning("⚠️ No se pudo establecer un índice datetime en el DataFrame de barras.")
            return pd.DataFrame(columns=['open','high','low','close','volume'])
    
    # Quedarnos solo con las columnas OHLCV que existan
    cols = [c for c in ['open','high','low','close','volume'] if c in df.columns]
    return df[cols].astype(float)
@dataclass
class PyramidConfig:
    # Señal
    base_threshold: float = 0.70        # Umbral mínimo de confianza para la 1ª posición
    threshold_mult: float = 1.05        # Multiplicador por cada posición adicional
    max_positions: int = 5              # Máximo de posiciones en una dirección
 
    # Horizonte / lookback
    lookback: int = 48                  # Ventana de secuencia que esperan los modelos
    pred_interval: int = 1              # Cada cuántas velas se re-predice (1 = cada vela)
 
    # TP / SL (en múltiplos de ATR)
    tp_atr_mult: float = TP_ATR_MULT            # Multiplicador ATR para Take Profit
    sl_atr_mult: float = SL_ATR_MULT            # Multiplicador ATR para Stop Loss
    atr_col: str = "atr"               # Nombre de la columna ATR en el DataFrame
 
    # Costos (se calculan desde spec si se pasan, pero se pueden sobreescribir)
    cost_ticks: Optional[float] = None  # Si None, se calcula desde spec
 
    # Comportamiento avanzado
    close_on_hold: bool = True          # Si True, HOLD cierra todas las posiciones
    close_on_reversal: bool = True      # Si True, señal contraria cierra y revierte
    slippage_pct: float = 0.01         # Slippage adicional como % del precio de entrada

def prepare_features(df, minutes, return_horizon_min, json_config_path, spec):
    # === Features exactas con las que se entrenó ===
    df = df.copy()
    if 'datetime' not in df.columns:
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        else:
            df = df.reset_index()
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            elif df.columns[0] == 'date':
                df['datetime'] = pd.to_datetime(df.iloc[:, 0])
            else:
                df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
                df['datetime'] = pd.to_datetime(df['datetime'])

    # CRITICAL: Convert to datetime FIRST, THEN check timezone
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = ensure_datetime_column(df)
    # Now .dt accessor is safe
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    # Convertir a tz-naive para tradepy (eliminar timezone info)
    df["close"] = wavelet_denoising(df['close'].values.copy())
    df["openint"] = 0
    df["dtyyyymmdd"] = df['datetime'].dt.strftime('%Y%m%d').astype(int)
    df["ticker"] = SYMBOL
    df["per"] = 120
    df_resampled = resample_ohlcv(df, period=f"{minutes}min")
    logger.info(f"Filas tras resample 60min: {len(df_resampled)}")
    if len(df_resampled) < 70: # 20 de indicadores + 48 de lookback
        logger.warning(f"⚠️ ¡Alerta! Solo hay {len(df_resampled)} velas de {minutes}min. " 
                       "Los indicadores necesitan al menos 20 y el modelo 48.")
    df_cumulative = daily_ohlcv_cummulative(df_resampled)
    with open(json_config_path) as f:
        config = json.load(f)

    config["global"]["sampling_minutes"] = minutes
    config["global"]["return_horizon_min"] = return_horizon_min
    config["global"]["tick_size"] = spec["tick_size"]
    
    df_final = generate_features(df_cumulative, config_json=config, dropna_strategy='all')
    logger.info(f"NaNs por columna:\n{df_final.isna().sum().sort_values(ascending=False).head(10)}")
    logger.info(f"raw len={len(df)} | resampled len={len(df_resampled)} | cumulative len={len(df_cumulative)} | final len={len(df_final)}")
    logger.info(f"Filas finales tras generar features: {len(df_final)}")
    return df_final

def log_prediccion_csv(symbol: str, prob_move: float, prob_buy: float, prob_sell: float, signal: str, confidence: float):
    """Guarda cada predicción en ../Data/{SYMBOL}log.csv para analizar deriva temporal."""
    csv_path = Path(f"../Data/{symbol}log.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ahora = datetime.datetime.now()
    fila = {
        "timestamp": ahora.isoformat(),
        "prob_move": round(prob_move, 4),
        "prob_buy":  round(prob_buy, 4),
        "prob_sell": round(prob_sell, 4),
        "signal":    signal,
        "confidence": round(confidence, 4),
    }
    escribir_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = pd.io.common  # solo para no importar csv aparte
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=fila.keys())
        if escribir_header:
            w.writeheader()
        w.writerow(fila)
        
def generar_prediccion(df: pd.DataFrame,
    models_bundle: dict,       # {'lstm': {l1, l2, scaler, features}, 'gru': {...}}
    minutes,
    json_config_path,
    return_horizon_min,
    spec,
    lookback: int = 24,
    threshold_move: float = 0.85,
    device: str = "cpu",
) -> Tuple[Literal['BUY', 'SELL', 'HOLD'], float]:
    if len(df) < lookback:
        return 'HOLD', 0.0
    all_probs_move = []
    all_probs_dir  = []
    seq_df = df.iloc[-60:]
    if seq_df.empty or seq_df.isna().all().all():
        logger.warning("⚠️ No valid features available - insufficient data or all NaN")
        return 'HOLD', 0.0
    
    seq_df = seq_df.select_dtypes(include=[np.number])
    seq = seq_df.values.astype(float)
    if seq.shape[0] == 0:
        logger.warning("⚠️ Empty feature sequence after selection")
        return 'HOLD', 0.0
    
    # Check for sufficient non-NaN data
    if seq_df.isna().sum().sum() > seq.shape[0] * seq.shape[1] * 0.5:  # >50% NaN
        logger.warning("⚠️ Too many NaN values in features")
        return 'HOLD', 0.0
    for key in ['lstm', 'gru']:
        m_info   = models_bundle[key]
        features = m_info['features']
        scaler   = m_info['scaler']
        model_l1 = m_info['model_l1']
        model_l2 = m_info['model_l2']

        # Validar features
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.warning(f"Modelo '{key}' features faltantes: {missing}")
            return 'HOLD', 0.0

        # Ventana de lookback
        window = df[features].iloc[-lookback:].values.astype(float)
        if window.shape[0] < lookback:
            return 'HOLD', 0.0

        # NaN check
        nan_ratio = np.isnan(window).sum() / window.size
        if nan_ratio > 0.1:
            logger.warning(f"⚠️ {key}: {nan_ratio:.1%} NaN en ventana — HOLD")
            return 'HOLD', 0.0

        # Escalar (shape 2D para RobustScaler)
        window_scaled = scaler.transform(window)                      # (lookback, features)
        X = torch.FloatTensor(window_scaled).unsqueeze(0).to(device)  # (1, lookback, features)

        with torch.no_grad():
            # L1: ¿hay movimiento?
            probs_move = torch.softmax(model_l1(X), dim=1).cpu().numpy()[0]  # [p_hold, p_move]
            # L2: ¿qué dirección?
            probs_dir  = torch.softmax(model_l2(X), dim=1).cpu().numpy()[0]  # [p_sell, p_buy]

        all_probs_move.append(probs_move)
        all_probs_dir.append(probs_dir)
    
    # Soft voting entre LSTM y GRU
    avg_move = np.mean(all_probs_move, axis=0)  # [p_hold, p_move]
    avg_dir  = np.mean(all_probs_dir,  axis=0)  # [p_sell, p_buy]

    prob_move = float(avg_move[1])
    prob_buy  = float(avg_dir[1])
    prob_sell = float(avg_dir[0])

    logger.info(f"L1 prob_move={prob_move:.3f} | L2 p_buy={prob_buy:.3f} p_sell={prob_sell:.3f}")

    if prob_move < threshold_move:
        log_prediccion_csv(SYMBOL, prob_move, prob_buy, prob_sell, 'HOLD', prob_move)
        return 'HOLD', prob_move

    # L2 decide dirección
    if prob_buy > prob_sell:
        log_prediccion_csv(SYMBOL, prob_move, prob_buy, prob_sell, 'BUY', prob_buy)
        return 'BUY', prob_buy
    else:
        log_prediccion_csv(SYMBOL, prob_move, prob_buy, prob_sell, 'SELL', prob_sell)
        return 'SELL', prob_sell
    
# ==================== CHECK MARGEN + DAILY LOSS + DRAWDOWN ====================
async def check_margin_and_risk(ib, daily_start_nlv, max_equity) -> Tuple[bool, float, float]:
    summary = await ib.accountSummaryAsync()
    excess_liquidity = float(next((v.value for v in summary if v.tag == 'ExcessLiquidity'), 0))
    maint_margin_req = float(next((v.value for v in summary if v.tag == 'MaintMarginReq'), 0))
    nlv = float(next((v.value for v in summary if v.tag == 'NetLiquidation'), 0))

    if daily_start_nlv is None:
        daily_start_nlv = nlv
    if max_equity is None or nlv > max_equity:
        max_equity = nlv

    daily_pnl = nlv - daily_start_nlv
    drawdown_pct = ((max_equity - nlv) / max_equity * 100) if max_equity > 0 else 0

    logger.info(f"Risk → Excess: ${excess_liquidity:,.0f} | MaintReq: ${maint_margin_req:,.0f} | "
                f"DailyPnL: ${daily_pnl:,.0f} | DD: {drawdown_pct:.2f}%")
    realized_pnl = float(next((v.value for v in summary if v.tag == 'RealizedPnL'), 0))
    if daily_pnl < MAX_DAILY_LOSS_USD and realized_pnl < MAX_DAILY_LOSS_USD:
        logger.critical(f"🚨 PÉRDIDA DIARIA MÁXIMA ALCANZADA → CERRANDO TODO")
        return False, daily_start_nlv, max_equity
    if drawdown_pct > MAX_DRAWDOWN_PCT:
        logger.critical(f"🚨 MAX DRAWDOWN ALCANZADO → CERRANDO TODO")
        return False, daily_start_nlv, max_equity
    if excess_liquidity < MARGIN_BUFFER_USD:
        logger.warning(f"⚠️ Margen insuficiente → no se operará")
        return False, daily_start_nlv, max_equity

    return True, daily_start_nlv, max_equity

async def close_all_positions(ib, contract):
    positions = ib.positions()
    pos = next((p for p in positions if p.contract.conId == contract.conId), None)
    if pos and pos.position != 0:
        action = 'SELL' if pos.position > 0 else 'BUY'
        ib.placeOrder(contract, MarketOrder(action, abs(pos.position), tif='DAY'))
        
# ==================== MAIN ASYNC (TODO INTEGRADO) ====================
async def main():
    ib = IB()
    await ib.connectAsync('127.0.0.1', 7497, clientId=CLIENT_ID, readonly=False)
    logger.info(f"✅ Conectado {SYMBOL} (clientId {CLIENT_ID})")
 
    async def ensure_connected():
        if not ib.isConnected():
            await ib.connectAsync('127.0.0.1', 7497, clientId=CLIENT_ID, readonly=False)
 
    logger.info(f"Buscando contrato con más volumen para {SYMBOL}...")
    contract = await get_active_future_contract(ib, SYMBOL, EXCHANGE, CURRENCY)
 
    if not contract:
        logger.error("No se pudo determinar el contrato activo.")
        return
 
    logger.info(f"✅ Contrato seleccionado: {contract.localSymbol} (Vence: {contract.lastTradeDateOrContractMonth})")
 
    # ── Tick value por símbolo para cálculo de PnL ───────────────────────────
    # Ajusta este mapa según los futuros que operes.
    TICK_VALUE_MAP = {'GC': 100.0, 'SI': 50.0, 'GBP': 62500.0, 'IBEX': 1.0, 'IBEX35': 1.0, 'IB': 1.0}
    TICK_VALUE = TICK_VALUE_MAP.get(SYMBOL, 100.0)

    # ── Reconciliar registro al arrancar ─────────────────────────────────────
    await reconciliar_registro(ib, contract, tick_value=TICK_VALUE)
 
    # Cierre de sesión
    details = (await ib.reqContractDetailsAsync(contract))[0]
    tz = pytz.timezone(details.timeZoneId)
    trading_hours = details.tradingHours
    now_ex = datetime.datetime.now(tz)
    today = now_ex.date()
    closes = []
 
    def parse_ib_datetime(s: str, tz):
        s = s.strip()
        if ':' in s:
            return datetime.datetime.strptime(s, '%Y%m%d:%H%M').replace(tzinfo=tz)
        return None
 
    for seg in trading_hours.split(';'):
        seg = seg.strip()
        if not seg or ':' not in seg:
            continue
        d_str, hours = seg.split(':', 1)
        seg_date = datetime.datetime.strptime(d_str, '%Y%m%d').date()
        if hours.upper() == 'CLOSED':
            continue
        for r in hours.split(','):
            r = r.strip()
            if '-' not in r:
                continue
            open_str, close_str = r.split('-', 1)
            if ':' in close_str:
                close_dt = datetime.datetime.strptime(close_str, '%Y%m%d:%H%M').replace(tzinfo=tz)
            else:
                close_time = datetime.datetime.strptime(close_str, '%H%M').time()
                close_dt = datetime.datetime.combine(seg_date, close_time, tzinfo=tz)
            if close_dt.date() >= today:
                closes.append(close_dt)
 
    session_close = max(closes) if closes else now_ex + datetime.timedelta(hours=24)
 
    # Cargar modelo PyTorch + SCALER
    if SYMBOL == "GBP":
        symbol = "bp"
    elif SYMBOL in ("IBEX", "IBEX35", "IB"):
        symbol = "ibex"
    else:
        symbol = SYMBOL.lower()
    model_l1_lstm, model_l2_lstm, sc_lstm, features_lstm = load_trading_model_2level(
        GoldLSTM_L1_Move, GoldLSTM_L2_Dir,
        path=f"../Models/{symbol}1/trading_model_lstm_2level",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model_l1_gru, model_l2_gru, sc_gru, features_gru = load_trading_model_2level(
        GoldGRU_L1_Move, GoldGRU_L2_Dir,
        path=f"../Models/{symbol}1/trading_model_gru_2level",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    models_bundle = {
        "lstm": {"model_l1": model_l1_lstm, "model_l2": model_l2_lstm,
                 "scaler": sc_lstm, "features": features_lstm},
        "gru":  {"model_l1": model_l1_gru,  "model_l2": model_l2_gru,
                 "scaler": sc_gru,  "features": features_gru},
    }
    logger.info(f"✅ Modelo PyTorch + Scaler cargados → {MODEL_PATH}")
 
    # Estado
    daily_start_nlv = None
    max_equity = None
    last_prediction_time = datetime.datetime.now() - datetime.timedelta(minutes=100)
    df_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    df_bars.index.name = 'datetime'
    consecutive_same = 0
    last_signal: Literal['BUY', 'SELL', None] = None
    active_sl_order = None
    active_tp_order = None
 
    # Carga histórica
    logger.info(f"Solicitando histórico estático de {DURACION_HISTORICO}...")
    try:
        # WHAT_TO_SHOW = 'MIDPOINT' if SYMBOL.upper() in ('GBP', 'EUR', 'JPY') else 'TRADES'
        WHAT_TO_SHOW = 'TRADES'
        static_bars = await ib.reqHistoricalDataAsync(
            contract, '', DURACION_HISTORICO, TIMEFRAME,
            WHAT_TO_SHOW, False, 1, keepUpToDate=False
        )
        logger.info("Activando suscripción en tiempo real (keepUpToDate)...")
        bars = await ib.reqHistoricalDataAsync(
            contract, '', '1 D', TIMEFRAME,
            WHAT_TO_SHOW, False, 1, keepUpToDate=True
        )
        df_static = normalize_bars_df(util.df(static_bars)) if static_bars else pd.DataFrame()
        df_live   = normalize_bars_df(util.df(bars))         if bars         else pd.DataFrame()
        new_df = pd.concat([df_static, df_live])
        if not new_df.empty:
            df_bars = new_df[~new_df.index.duplicated(keep='last')].sort_index()
            df_bars['atr'] = calculate_atr(df_bars)
            logger.info(f"✅ Datos inicializados: {len(df_bars)} filas.")
        else:
            logger.warning("⚠️ No se pudieron cargar datos iniciales.")
    except Exception as e:
        logger.error(f"❌ Error crítico en carga inicial: {e}")
    
    await detectar_huerfanas(ib, contract, df_bars)
    
    logger.info(f"🚀 Bot {SYMBOL} PyTorch + Scaler iniciado")
    # Prueba: escribir un par de logs para que Promtail los capture (movements + balance)
    try:
        log_movement(SYMBOL, "long", 1234.56, 1)
        log_balance(100000.0, 100100.0, 0.1)
        logger.info("✅ Test logs escritos: movements & balance")
    except Exception as e:
        logger.warning(f"⚠️ No se pudieron escribir logs de prueba: {e}")
 
    # Recuperar órdenes SL/TP activas tras reconexión
    current_pos = next((p.position for p in ib.positions() if p.contract.conId == contract.conId), 0.0)
    open_trades = await ib.reqAllOpenOrdersAsync()
    for t in open_trades:
        if t.contract.conId == contract.conId:
            order = t.order
            if order.orderType == 'STP' and order.action != ('BUY' if current_pos > 0 else 'SELL'):
                active_sl_order = order
                logger.info(f"✅ SL recuperado: ID {order.orderId} a {order.auxPrice}")
            elif order.orderType == 'LMT' and order.action != ('BUY' if current_pos > 0 else 'SELL'):
                active_tp_order = order
                logger.info(f"✅ TP recuperado: ID {order.orderId} a {order.lmtPrice}")
    # await detectar_huerfanas(ib, contract, df_bars)
    try:
        while True:
            await ensure_connected()
            ahora = datetime.datetime.now()
            now_ex = datetime.datetime.now(tz)
            min_to_close = (session_close - now_ex).total_seconds() / 60.0
 
            # Risk check
            can_trade, daily_start_nlv, max_equity = await check_margin_and_risk(ib, daily_start_nlv, max_equity)
            nlv_actual = max_equity
            nlv_actual = await get_nlv(ib)
            # Actualizar barras + ATR
            if bars:
                new_data = normalize_bars_df(util.df(bars))
                if not new_data.empty:
                    df_bars = pd.concat([df_bars, new_data])
                    df_bars = df_bars[~df_bars.index.duplicated(keep='last')].sort_index()
                    df_bars['atr'] = calculate_atr(df_bars)
                else:
                    logger.warning("⚠️ bars actualizado pero vino vacío o sin columnas reconocibles.")
 
            current_pos = next((p.position for p in ib.positions() if p.contract.conId == contract.conId), 0.0)

            if current_pos != 0:
                actualizar_mae_mfe(df_bars)

            # ── CIERRE POR TIEMPO (2h por contrato) ──────────────────────────
            await cerrar_operaciones_por_tiempo(ib, contract, tick_value=TICK_VALUE, nlv=nlv_actual)
 
            # ── CIERRE INTRADAY (fin de sesión) ──────────────────────────────
            if current_pos != 0 and min_to_close <= CIERRE_VENTANA_FIN_MIN:
                action = 'SELL' if current_pos > 0 else 'BUY'
                ib.placeOrder(contract, MarketOrder(action, abs(current_pos), tif='DAY'))
                logger.info(f"🚨 {SYMBOL} FORCE CLOSE intraday")
                precio_ref = df_bars['close'].iloc[-1] if not df_bars.empty else 0.0
                for op in get_operaciones_abiertas():
                    MULTIPLIER = float(contract.multiplier) if contract.multiplier else 1.0
                    cerrar_operacion(op['trade_id'], precio_ref,
                                     motivo='CIERRE_FIN_SESION', tick_value=TICK_VALUE,
                                     equity_al_cierre=nlv_actual, multiplier=MULTIPLIER)
                active_sl_order = active_tp_order = None
 
            # ── PREDICCIÓN + PYRAMIDING + TRAILING ───────────────────────────
            intervalo_activo = PREDICCION_SIN_POSICION_MINUTOS if current_pos == 0 else PREDICCION_CADA_MINUTOS
            if (ahora - last_prediction_time).total_seconds() / 60 >= intervalo_activo:
                if not can_trade:
                    logger.critical("🚨 RISK OFF → CERRANDO POSICIONES")
                    await close_all_positions(ib, contract)
                    last_prediction_time = ahora
                    await asyncio.sleep(60)
                    continue
 
                if df_bars.empty or len(df_bars) < 100:
                    logger.warning(f"⚠️ Datos insuficientes en df_bars ({len(df_bars)} filas). Reintentando descarga...")
                    bars = await ib.reqHistoricalDataAsync(contract, '', DURACION_HISTORICO, TIMEFRAME,
                                                          WHAT_TO_SHOW, False, 1, keepUpToDate=True)
                    continue
 
                CONFIG_PATH = "../Data/features_config.json"
                specs = pd.read_json("../Data/futuros_specs.json")
                if SYMBOL == "GBP":
                    symbol = "bp"
                else:
                    symbol = SYMBOL.lower()
                if SYMBOL in ("IBEX", "IBEX35", "IB"):
                    spec = specs["IBEX"]
                else:
                    spec = specs[symbol[:2].upper()]
                TICK_SIZE_GC = spec['tick_size']
                df_features = prepare_features(
                    df_bars, json_config_path=CONFIG_PATH,
                    minutes=60, return_horizon_min=120, spec={'tick_size': TICK_SIZE_GC}
                )
                logger.info(f"Últimos 3 precios raw: {df_bars['close'].tail(3).values}")
                if df_features.empty:
                    logger.warning("⚠️ df_features quedó vacío tras el procesamiento. Esperando más datos...")
                    last_prediction_time = ahora
                    continue
 
                logger.info(f"Última fila features: {df_features.index[-1]} | Close: {df_features['close'].iloc[-1]}")
 
                signal, confidence = generar_prediccion(
                    df_features, models_bundle,
                    minutes=60,
                    json_config_path=CONFIG_PATH,
                    return_horizon_min=120,
                    spec={'tick_size': TICK_SIZE_GC},
                    lookback=48,
                    threshold_move=0.70 if SYMBOL in ('IBEX', 'IBEX35', 'IB') else 0.80,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                logger.info(f"[{ahora.strftime('%H:%M:%S')}] {SYMBOL} PRED: {signal} (conf={confidence:.3f}) | pos={current_pos}")
 
                last_prediction_time = ahora
 
                # HOLD → cerrar todo
                if signal == 'HOLD':
                    if current_pos != 0:
                        action = 'SELL' if current_pos > 0 else 'BUY'
                        ib.placeOrder(contract, MarketOrder(action, abs(current_pos), tif='DAY'))
                        logger.info("   → HOLD: cierre total")
                        precio_ref = df_bars['close'].iloc[-1] if not df_bars.empty else 0.0
                        for op in get_operaciones_abiertas():
                            MULTIPLIER = float(contract.multiplier) if contract.multiplier else 1.0
                            cerrar_operacion(op['trade_id'], precio_ref,
                                             motivo='CIERRE_HOLD', tick_value=TICK_VALUE, multiplier=MULTIPLIER)
                    consecutive_same = 0
                    last_signal = None
                    active_sl_order = active_tp_order = None
                    continue
 
                # Lógica de racha y reverse
                if (signal == 'BUY' and current_pos < 0) or (signal == 'SELL' and current_pos > 0):
                    if active_sl_order:
                        ib.cancelOrder(active_sl_order)
                    if active_tp_order:
                        ib.cancelOrder(active_tp_order)
                    action_flat = 'SELL' if current_pos > 0 else 'BUY'
                    if current_pos != 0:
                        ib.placeOrder(contract, MarketOrder(action_flat, abs(current_pos), tif='DAY'))
                        precio_ref = df_bars['close'].iloc[-1] if not df_bars.empty else 0.0
                        for op in get_operaciones_abiertas():
                            MULTIPLIER = float(contract.multiplier) if contract.multiplier else 1.0
                            cerrar_operacion(op['trade_id'], precio_ref,
                                             motivo='CIERRE_REVERSAL', tick_value=TICK_VALUE, multiplier=MULTIPLIER)
                    consecutive_same = 1
                    last_signal = signal
                elif signal == last_signal:
                    consecutive_same += 1
                else:
                    consecutive_same = 1
                    last_signal = signal
 
                threshold = 0.55 + (consecutive_same - 1) * 0.05
                total_contracts = abs(current_pos)
 
                # ── NUEVA ENTRADA O SCALE-IN ──────────────────────────────────
                if confidence >= threshold and total_contracts < MAX_CONTRATOS:
                    precio = df_bars['close'].iloc[-1]
 
                    # SL/TP dinámicos basados en ATR (punto 1)
                    atr_value = df_bars['atr'].iloc[-1]
                    # if pd.isna(atr_value) or atr_value <= 0:
                    #     logger.warning("⚠️ ATR no disponible, usando SL/TP fijos como fallback.")
                    #     atr_value = 0.0
                    #     sl_dist = SL_PUNTOS_FALLBACK
                    #     tp_dist = TP_PUNTOS_FALLBACK
                    if pd.isna(atr_value) or atr_value <= 0:
                        divisor = 10000 if SYMBOL == "GBP" else 1
                        sl_dist = SL_PUNTOS_FALLBACK / divisor
                        tp_dist = TP_PUNTOS_FALLBACK / divisor
                    else:
                        sl_dist = atr_value * SL_ATR_MULT
                        tp_dist = atr_value * TP_ATR_MULT

                    sl = _snap_price(precio - sl_dist if signal == 'BUY' else precio + sl_dist, SYMBOL)
                    tp = _snap_price(precio + tp_dist if signal == 'BUY' else precio - tp_dist, SYMBOL)

                    logger.info(f"   → SL/TP dinámicos | ATR={atr_value:.5f} | SL={sl} | TP={tp}")

                    bracket = ib.bracketOrder(
                        action=signal,
                        quantity=1,
                        limitPrice=_snap_price(precio, SYMBOL),
                        takeProfitPrice=tp,
                        stopLossPrice=sl
                    )
                    parent, tp_order, sl_order = bracket
                    parent.orderType = "MKT"
                    parent.transmit = True
                    tp_order.transmit = True
                    sl_order.transmit = True
                    parent.tif = "DAY"
                    tp_order.tif = "DAY"
                    sl_order.tif = "DAY"
 
                    ib.placeOrder(contract, parent)
                    ib.placeOrder(contract, tp_order)
                    ib.placeOrder(contract, sl_order)
 
                    active_sl_order = sl_order
                    active_tp_order = tp_order
 
                    # Registrar en JSON (punto 2)
                    registrar_entrada(
                        symbol=SYMBOL,
                        contract_local=contract.localSymbol,
                        signal=signal,
                        precio_entrada=precio,
                        sl=sl,
                        tp=tp,
                        atr=atr_value,
                        parent_order_id=parent.orderId,
                        sl_order_id=sl_order.orderId,
                        tp_order_id=tp_order.orderId,
                    )
 
                    logger.info(f"   → {'SCALE-IN' if total_contracts > 0 else 'ENTRADA'} {signal} "
                                f"{CANTIDAD_BASE} | total={total_contracts + CANTIDAD_BASE}")
 
                # ── TRAILING SL/TP CON ATR (baja confianza) ──────────────────
                elif confidence < threshold and current_pos != 0 and active_sl_order and active_tp_order:
                    precio = df_bars['close'].iloc[-1]
                    atr_value = df_bars['atr'].iloc[-1]
 
                    if signal == 'BUY':
                        new_sl = _snap_price(precio - (atr_value * ATR_TRAIL_MULTIPLIER), SYMBOL)
                        new_tp = _snap_price(precio + (atr_value * TP_ATR_MULT * 1.2), SYMBOL)
                    else:
                        new_sl = _snap_price(precio + (atr_value * ATR_TRAIL_MULTIPLIER), SYMBOL)
                        new_tp = _snap_price(precio - (atr_value * TP_ATR_MULT * 1.2), SYMBOL)
 
                    if active_sl_order:
                        active_sl_order.auxPrice = new_sl
                        ib.placeOrder(contract, active_sl_order)
                    if active_tp_order:
                        active_tp_order.lmtPrice = new_tp
                        active_tp_order.transmit = True
                        ib.placeOrder(contract, active_tp_order)
 
                    logger.info(f"   → TRAILING CON ATR | ATR={atr_value:.5f} | SL={new_sl} | TP={new_tp}")
 
            await asyncio.sleep(60)
 
    except Exception as e:
        logger.error(f"❌ Error {SYMBOL}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        ib.disconnect()
        logger.info(f"✅ {SYMBOL} desconectado")
 
if __name__ == "__main__":
    util.run(main())