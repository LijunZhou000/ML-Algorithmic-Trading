import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Literal, Tuple
import pandas as pd
import datetime
import pytz
import torch
import torch.nn as nn
import joblib  # ← para cargar el scaler
from ib_async import *

# ==================== LOGGING (preparado para Grafana + Alertmanager) ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==================== CLI ====================
parser = argparse.ArgumentParser(description="Bot intraday async PyTorch + ATR trailing + risk + scaler")
parser.add_argument('--symbol', type=str, required=True)
parser.add_argument('--expiration', type=str, default='')
parser.add_argument('--client-id', type=int, required=True)
parser.add_argument('--model-path', type=str, required=True, help='Ej: models/lstm_gc.pth')
args = parser.parse_args()

SYMBOL = args.symbol
EXPIRATION = args.expiration
CLIENT_ID = args.client_id
MODEL_PATH = args.model_path
PREDICCION_CADA_MINUTOS = 5

# ==================== CONFIG RIESGO + ATR ====================
MAX_DAILY_LOSS_USD = -1000.0
MAX_DRAWDOWN_PCT = 3.0
MARGIN_BUFFER_USD = 2000.0

ATR_PERIOD = 14
ATR_TRAIL_MULTIPLIER = 1.5

CANTIDAD_BASE = 1
MAX_CONTRATOS = 5
SL_PUNTOS = 20.0
TP_PUNTOS = 40.0
CIERRE_VENTANA_INICIO_MIN = 15
CIERRE_VENTANA_FIN_MIN = 5

TIMEFRAME = '1 min'
DURACION_HISTORICO = '3 D'

JSON_POS = Path(f'posiciones_{SYMBOL}.json')
JSON_BARS = Path(f'bars_{SYMBOL}.json')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== MODELO LSTM (exacta a la del retrain) ====================
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return self.sigmoid(out)

# ==================== CACHE BARRAS ====================
def load_bars_cache() -> pd.DataFrame:
    if not JSON_BARS.exists():
        return pd.DataFrame()
    with open(JSON_BARS) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

def save_bars_cache(df: pd.DataFrame):
    df_reset = df.reset_index()
    df_reset['datetime'] = df_reset['datetime'].astype(str)
    with open(JSON_BARS, 'w') as f:
        json.dump(df_reset.to_dict('records'), f, indent=2)

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ==================== PREDICCIÓN PyTorch + SCALER ====================
def generar_prediccion(df: pd.DataFrame, model: LSTMModel, scaler) -> Tuple[Literal['BUY', 'SELL', 'HOLD'], float]:
    if len(df) < 60:
        return 'HOLD', 0.0

    # === Features exactas con las que se entrenó (5 columnas) ===
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    seq = df[feature_cols].iloc[-60:].values.astype(float)

    # Aplicar scaler (el mismo que guardaste al entrenar)
    scaled = scaler.transform(seq)

    # Tensor para PyTorch
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # shape (1, 60, 5)

    model.eval()
    with torch.no_grad():
        pred = model(tensor).item()

    confidence = float(pred)
    if confidence > 0.60:
        return 'BUY', confidence
    elif confidence < 0.40:
        return 'SELL', confidence
    return 'HOLD', confidence

# ==================== CHECK MARGEN + DAILY LOSS + DRAWDOWN ====================
async def check_margin_and_risk(ib: IB, daily_start_nlv, max_equity) -> Tuple[bool, float, float]:
    summary = await ib.accountSummary()
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

    if daily_pnl < MAX_DAILY_LOSS_USD:
        logger.critical(f"🚨 PÉRDIDA DIARIA MÁXIMA ALCANZADA → CERRANDO TODO")
        return False, daily_start_nlv, max_equity
    if drawdown_pct > MAX_DRAWDOWN_PCT:
        logger.critical(f"🚨 MAX DRAWDOWN ALCANZADO → CERRANDO TODO")
        return False, daily_start_nlv, max_equity
    if excess_liquidity < MARGIN_BUFFER_USD:
        logger.warning(f"⚠️ Margen insuficiente → no se operará")
        return False, daily_start_nlv, max_equity

    return True, daily_start_nlv, max_equity

async def close_all_positions(ib: IB, contract: Contract):
    positions = ib.positions()
    pos = next((p for p in positions if p.contract.conId == contract.conId), None)
    if pos and pos.position != 0:
        action = 'SELL' if pos.position > 0 else 'BUY'
        await ib.placeOrder(contract, MarketOrder(action, abs(pos.position), tif='DAY'))

# ==================== MAIN ASYNC (TODO INTEGRADO) ====================
async def main():
    ib = IB()
    await ib.connect('127.0.0.1', 7497, clientId=CLIENT_ID, readonly=False)
    logger.info(f"✅ Conectado {SYMBOL} (clientId {CLIENT_ID})")

    async def ensure_connected():
        if not ib.isConnected():
            logger.warning("⚠️ Reconectando...")
            await ib.connect('127.0.0.1', 7497, clientId=CLIENT_ID, readonly=False)

    # Contrato
    contract = Future(symbol=SYMBOL, lastTradeDateOrContractMonth=EXPIRATION,
                      exchange='CME', currency='USD')
    contracts = await ib.qualifyContracts(contract)
    contract = contracts[0]
    logger.info(f"✅ Contrato: {contract.localSymbol} {contract.lastTradeDateOrContractMonth}")

    # Cierre de sesión
    details = (await ib.reqContractDetails(contract))[0]
    tz = pytz.timezone(details.timeZoneId)
    trading_hours = details.tradingHours
    now_ex = datetime.datetime.now(tz)
    today = now_ex.date()
    closes = []
    for seg in trading_hours.split(';'):
        if ':' not in seg: continue
        d_str, hours = seg.split(':', 1)
        if datetime.datetime.strptime(d_str, '%Y%m%d').date() == today and hours.upper() != 'CLOSED':
            for r in hours.split(','):
                _, c_str = r.split('-')
                c_time = datetime.datetime.strptime(c_str, '%H%M').time()
                closes.append(datetime.datetime.combine(today, c_time, tzinfo=tz))
    session_close = max(closes) if closes else now_ex + datetime.timedelta(hours=24)

    # Cargar modelo PyTorch + SCALER
    model = LSTMModel(input_size=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    scaler_path = MODEL_PATH.replace('.pth', '_scaler.joblib')
    scaler = joblib.load(scaler_path)
    logger.info(f"✅ Modelo PyTorch + Scaler cargados → {MODEL_PATH} | {scaler_path}")

    # Estado
    daily_start_nlv = None
    max_equity = None
    last_prediction_time = datetime.datetime.now() - datetime.timedelta(minutes=100)
    consecutive_same = 0
    last_signal: Literal['BUY', 'SELL', None] = None
    active_sl_order = None
    active_tp_order = None

    df_bars = load_bars_cache()
    if df_bars.empty:
        bars = await ib.reqHistoricalData(contract, '', DURACION_HISTORICO, TIMEFRAME,
                                          'TRADES', True, 1, keepUpToDate=True)
        df_bars = util.df(bars)
        df_bars['datetime'] = pd.to_datetime(df_bars['date'])
        df_bars.set_index('datetime', inplace=True)
        df_bars = df_bars[['open','high','low','close','volume']].astype(float)
        save_bars_cache(df_bars)

    logger.info(f"🚀 Bot {SYMBOL} PyTorch + Scaler iniciado")

    try:
        while True:
            await ensure_connected()
            ahora = datetime.datetime.now()
            now_ex = datetime.datetime.now(tz)
            min_to_close = (session_close - now_ex).total_seconds() / 60.0

            # Risk check
            can_trade, daily_start_nlv, max_equity = await check_margin_and_risk(ib, daily_start_nlv, max_equity)

            # Actualizar barras + ATR
            bars = await ib.reqHistoricalData(contract, '', '1 D', TIMEFRAME,
                                              'TRADES', True, 1, keepUpToDate=True)
            new_df = util.df(bars)
            new_df['datetime'] = pd.to_datetime(new_df['date'])
            new_df.set_index('datetime', inplace=True)
            new_df = new_df[['open','high','low','close','volume']].astype(float)
            df_bars = pd.concat([df_bars, new_df[~new_df.index.isin(df_bars.index)]])
            df_bars = df_bars[~df_bars.index.duplicated(keep='last')].sort_index()
            df_bars['atr'] = calculate_atr(df_bars)
            save_bars_cache(df_bars)

            current_pos = next((p.position for p in ib.positions() if p.contract.conId == contract.conId), 0.0)

            # CIERRE INTRADAY
            if current_pos != 0 and min_to_close <= CIERRE_VENTANA_FIN_MIN:
                action = 'SELL' if current_pos > 0 else 'BUY'
                await ib.placeOrder(contract, MarketOrder(action, abs(current_pos), tif='DAY'))
                logger.info(f"🚨 {SYMBOL} FORCE CLOSE intraday")
                active_sl_order = active_tp_order = None

            # PREDICCIÓN + PYRAMIDING + TRAILING
            if (ahora - last_prediction_time).total_seconds() / 60 >= PREDICCION_CADA_MINUTOS:
                if not can_trade:
                    last_prediction_time = ahora
                    await asyncio.sleep(60)
                    continue

                signal, confidence = generar_prediccion(df_bars, model, scaler)
                logger.info(f"[{ahora.strftime('%H:%M:%S')}] {SYMBOL} PRED: {signal} (conf={confidence:.3f}) | pos={current_pos}")

                last_prediction_time = ahora

                # HOLD → cerrar todo
                if signal == 'HOLD':
                    if current_pos != 0:
                        action = 'SELL' if current_pos > 0 else 'BUY'
                        await ib.placeOrder(contract, MarketOrder(action, abs(current_pos), tif='DAY'))
                        logger.info("   → HOLD: cierre total")
                    consecutive_same = 0
                    last_signal = None
                    active_sl_order = active_tp_order = None
                    continue

                # Lógica de racha y reverse
                if (signal == 'BUY' and current_pos < 0) or (signal == 'SELL' and current_pos > 0):
                    action_flat = 'SELL' if current_pos > 0 else 'BUY'
                    if current_pos != 0:
                        await ib.placeOrder(contract, MarketOrder(action_flat, abs(current_pos), tif='DAY'))
                    consecutive_same = 1
                    last_signal = signal
                elif signal == last_signal:
                    consecutive_same += 1
                else:
                    consecutive_same = 1
                    last_signal = signal

                threshold = 0.55 + (consecutive_same - 1) * 0.05
                total_contracts = abs(current_pos)

                # NUEVA ENTRADA O SCALE-IN
                if confidence >= threshold and total_contracts < MAX_CONTRATOS:
                    precio = df_bars['close'].iloc[-1]
                    sl = precio - SL_PUNTOS if signal == 'BUY' else precio + SL_PUNTOS
                    tp = precio + TP_PUNTOS if signal == 'BUY' else precio - TP_PUNTOS

                    parent, sl_order, tp_order = bracketOrder(
                        action=signal, quantity=CANTIDAD_BASE,
                        limitPrice=precio, stopLossPrice=sl, takeProfitPrice=tp,
                        orderType='MKT', tif='DAY'
                    )
                    await ib.placeOrder(contract, parent)
                    await ib.placeOrder(contract, sl_order)
                    await ib.placeOrder(contract, tp_order)

                    active_sl_order = sl_order
                    active_tp_order = tp_order
                    logger.info(f"   → {'SCALE-IN' if total_contracts > 0 else 'ENTRADA'} {signal} "
                                f"{CANTIDAD_BASE} | total={total_contracts + CANTIDAD_BASE}")

                # TRAILING SL/TP CON ATR (baja confianza)
                elif confidence < threshold and current_pos != 0 and active_sl_order and active_tp_order:
                    precio = df_bars['close'].iloc[-1]
                    atr_value = df_bars['atr'].iloc[-1]

                    if signal == 'BUY':
                        new_sl = precio - (atr_value * ATR_TRAIL_MULTIPLIER)
                        new_tp = precio + (TP_PUNTOS * 1.2)
                    else:
                        new_sl = precio + (atr_value * ATR_TRAIL_MULTIPLIER)
                        new_tp = precio - (TP_PUNTOS * 1.2)

                    if active_sl_order:
                        active_sl_order.auxPrice = new_sl
                        await ib.modifyOrder(active_sl_order)
                    if active_tp_order:
                        active_tp_order.auxPrice = new_tp
                        await ib.modifyOrder(active_tp_order)

                    logger.info(f"   → TRAILING CON ATR | ATR={atr_value:.2f} | SL={new_sl:.2f} | TP={new_tp:.2f}")

            await asyncio.sleep(60)

    except Exception as e:
        logger.error(f"❌ Error {SYMBOL}: {e}")
    finally:
        await ib.disconnect()
        logger.info(f"✅ {SYMBOL} desconectado")

if __name__ == "__main__":
    asyncio.run(main())
