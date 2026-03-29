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
from tradepy.load import import_dataset, clean, wavelet_denoising, resample_ohlcv, daily_ohlcv_cummulative
from tradepy.features import generate_features
import os
from ib_async import util

# ==================== LOGGING (preparado para Grafana + Alertmanager) ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==================== CLI ====================
parser = argparse.ArgumentParser(description="Bot intraday async PyTorch + ATR trailing + risk + scaler")
parser.add_argument('--symbol', type=str, required=True)
# parser.add_argument('--expiration', type=str, default='')
parser.add_argument('--client-id', type=int, required=True)
parser.add_argument('--model-path', type=str, required=True, help='Ej: models/lstm_gc.pth')
args = parser.parse_args()

SYMBOL = args.symbol
# EXPIRATION = args.expiration
CLIENT_ID = args.client_id
MODEL_PATH = args.model_path
PREDICCION_CADA_MINUTOS = 60

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
async def get_active_future_contract(ib, symbol):
    # OJO: exchange='NYMEX' y secType='FUT'
    template = Future(
        symbol=symbol,
        exchange='COMEX',
        currency='USD'
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
def generar_prediccion(df: pd.DataFrame, model: LSTMModel, scaler, minutes: int, json_config_path, return_horizon_min: int, spec: dict) -> Tuple[Literal['BUY', 'SELL', 'HOLD'], float]:
    if len(df) < 60:
        return 'HOLD', 0.0

    # === Features exactas con las que se entrenó (5 columnas) ===
    df = df.copy()
    df["close"] = wavelet_denoising(df['close'].values.copy())
    df_resampled = resample_ohlcv(df, period=f"{minutes}min")
    df_cumulative = daily_ohlcv_cummulative(df_resampled)
    with open(json_config_path) as f:
        config = json.load(f)

    config["global"]["sampling_minutes"] = minutes
    config["global"]["return_horizon_min"] = return_horizon_min
    config["global"]["tick_size"] = spec["tick_size"]
    
    df_final = generate_features(df_cumulative, config_json=config, dropna_strategy='any')
    exclude_cols_triple = [
        # Identificadores / temporales no cíclicos
        "datetime", "dtyyyymmdd", "trading_date", "ticker", "per", "openint",

        # OHLCV raw (el modelo no debería ver precios absolutos)
        "open", "high", "low", "close", "volume",

        # Targets
        "target_bin", f"target_logret_{minutes}", f"target_ret_{minutes}", f"target_ticks_{minutes}"
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols_triple]

    seq = df_final[feature_cols].iloc[-60:].values.astype(float)

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
            # logger.warning("⚠️ Reconectando...")
            await ib.connectAsync('127.0.0.1', 7497, clientId=CLIENT_ID, readonly=False)

    # # Contrato
    # contract = Future(symbol=SYMBOL, lastTradeDateOrContractMonth=EXPIRATION,
    #                   exchange='CME', currency='USD')
    # contracts = await ib.qualifyContracts(contract)
    # contract = contracts[0]
    # logger.info(f"✅ Contrato: {contract.localSymbol} {contract.lastTradeDateOrContractMonth}")
    logger.info(f"Buscando contrato con más volumen para {SYMBOL}...")
    contract = await get_active_future_contract(ib, SYMBOL)
    
    if not contract:
        logger.error("No se pudo determinar el contrato activo.")
        return

    logger.info(f"✅ Contrato seleccionado: {contract.localSymbol} (Vence: {contract.lastTradeDateOrContractMonth})")

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

    closes = []

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
    # Separamos la carga del scaler de la instancia del modelo para poder
    # inferir `input_dim` desde el scaler guardado (scaler.n_features_in_).
    def load_model_artifacts(folder_path, model_class=None, **model_params):
        """
        Carga el escalador y, opcionalmente, el modelo.
        - Si `model_class` es None, devuelve (None, scaler).
        - Si se proporciona `model_class` y `model_params`, instancia y carga el modelo.
        """
        # 1. Cargar el escalador
        scaler_path = os.path.join(folder_path, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✅ Escalador cargado.")
        else:
            raise FileNotFoundError("No se encontró el archivo del escalador.")

        # Si no pidieron el modelo, devolvemos solo el scaler
        if model_class is None:
            return None, scaler

        # 2. Instanciar la clase del modelo con los parámetros proporcionados
        model = model_class(**model_params)

        # 3. Cargar los pesos (.pth)
        model_path = os.path.join(folder_path, "model.pth")
        if os.path.exists(model_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()  # IMPORTANTE: Poner el modelo en modo evaluación
            print(f"✅ Modelo cargado y puesto en modo EVAL en {device}.")
        else:
            raise FileNotFoundError("No se encontró el archivo model.pth.")

        return model, scaler

    # --- EJEMPLO DE USO Y VERIFICACIÓN ---
    class GoldAttentionGRU_Triple(nn.Module):
        def __init__(self, input_dim, output_dim=3, hidden_dim=256, num_layers=2, dropout=0.3):
            super().__init__()
            # Usamos Bidirectional para que el modelo vea la estructura de la serie temporal en ambos sentidos
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout, bidirectional=True)
            
            # Mecanismo de Atención: mapea el estado oculto a una puntuación de importancia
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Normalización post-atención para estabilizar el entrenamiento triple
            self.bn = nn.BatchNorm1d(hidden_dim * 2)
            
            # Capas densas para clasificar en 3 categorías
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(64, output_dim) # output_dim = 3
            )
    
        def forward(self, x):
            # 1. Pasar por GRU: out shape [batch, seq_len, hidden_dim * 2]
            gru_out, _ = self.gru(x)
            
            # 2. Calcular pesos de atención para cada paso de la secuencia
            # energy shape: [batch, seq_len, 1]
            energy = self.attention(gru_out)
            weights = torch.softmax(energy, dim=1)
            
            # 3. Vector de contexto (Suma ponderada de todos los estados temporales)
            # context shape: [batch, hidden_dim * 2]
            context = torch.sum(weights * gru_out, dim=1)
            
            # 4. Clasificación final
            out = self.bn(context)
            logits = self.fc(out)
            
            return logits # Retorna logits (CrossEntropyLoss se encarga del Softmax internamente)
    # Primero cargamos solo el scaler para poder inferir el `input_dim`
    _model_tmp, scaler = load_model_artifacts(folder_path=MODEL_PATH, model_class=None)

    input_dim = getattr(scaler, 'n_features_in_', None)
    if input_dim is None:
        if hasattr(scaler, 'scale_'):
            input_dim = scaler.scale_.shape[0]
        elif hasattr(scaler, 'mean_'):
            input_dim = scaler.mean_.shape[0]
        else:
            raise RuntimeError('No se pudo inferir input_dim desde el scaler; pásalo manualmente.')

    params = {
        'input_dim': int(input_dim),
        'output_dim': 3,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3
    }

    model, scaler = load_model_artifacts(
        folder_path=MODEL_PATH,
        model_class=GoldAttentionGRU_Triple,
        **params
    )
    logger.info(f"✅ Modelo PyTorch + Scaler cargados → {MODEL_PATH}")

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
        bars = await ib.reqHistoricalDataAsync(contract, '', DURACION_HISTORICO, TIMEFRAME,
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
            bars = await ib.reqHistoricalDataAsync(contract, '', '1 D', TIMEFRAME,
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
                ib.placeOrder(contract, MarketOrder(action, abs(current_pos), tif='DAY'))
                logger.info(f"🚨 {SYMBOL} FORCE CLOSE intraday")
                active_sl_order = active_tp_order = None

            # PREDICCIÓN + PYRAMIDING + TRAILING
            if (ahora - last_prediction_time).total_seconds() / 60 >= PREDICCION_CADA_MINUTOS:
                if not can_trade:
                    last_prediction_time = ahora
                    await asyncio.sleep(60)
                    continue

                # Config para tu modelo XAU_GRU_bin_V2 (60min wavelet)
                CONFIG_PATH = "E:/Futuro/MLAlgoTrading/TradingBots/Features/features_config.json"  # ← Busca este archivo
                TICK_SIZE_GC = 0.10  # Gold futures COMEX
                
                signal, confidence = generar_prediccion(
                    df_bars, model, scaler,
                    minutes=60,
                    json_config_path=CONFIG_PATH,
                    return_horizon_min=60,
                    spec={'tick_size': TICK_SIZE_GC}
                )
                logger.info(f"[{ahora.strftime('%H:%M:%S')}] {SYMBOL} PRED: {signal} (conf={confidence:.3f}) | pos={current_pos}")

                last_prediction_time = ahora

                # HOLD → cerrar todo
                if signal == 'HOLD':
                    if current_pos != 0:
                        action = 'SELL' if current_pos > 0 else 'BUY'
                        ib.placeOrder(contract, MarketOrder(action, abs(current_pos), tif='DAY'))
                        logger.info("   → HOLD: cierre total")
                    consecutive_same = 0
                    last_signal = None
                    active_sl_order = active_tp_order = None
                    continue

                # Lógica de racha y reverse
                if (signal == 'BUY' and current_pos < 0) or (signal == 'SELL' and current_pos > 0):
                    action_flat = 'SELL' if current_pos > 0 else 'BUY'
                    if current_pos != 0:
                        ib.placeOrder(contract, MarketOrder(action_flat, abs(current_pos), tif='DAY'))
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
                    ib.placeOrder(contract, parent)
                    ib.placeOrder(contract, sl_order)
                    ib.placeOrder(contract, tp_order)

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
    util.run(main())
    