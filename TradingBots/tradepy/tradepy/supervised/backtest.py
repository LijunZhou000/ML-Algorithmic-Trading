from tradepy.data.loader import load_future
from tradepy.config.config import load_config, load_symbols, load_feature_config, load_exclude_config
from tradepy.data.loader import load_specs
from tradepy.data.cleaner import add_trading_date_by_gap
from tradepy.data.resampler import resample_ohlcv, daily_ohlcv_cummulative
from tradepy.features.generate import generate_features
from tradepy.features.quality_check import data_quality_report
from tradepy.supervised.load import prepare_all
from tradepy.paths import EXCLUDE_CONFIG, SYSTEM_CONFIG
from tradepy.supervised.target import target_triple_barrier_interday
from tradepy.supervised.pretrain import filter_features
from tradepy.supervised.models import GoldLSTM_L1_Move, GoldLSTM_L2_Dir, GoldGRU_L1_Move, GoldGRU_L2_Dir, BasicLSTM_L1_Move, BasicLSTM_L2_Dir, BasicGRU_L1_Move, BasicGRU_L2_Dir, BasicLSTM_L3_Regression, BasicGRU_L3_Regression
from tradepy.supervised.train import train_walk_forward_2level, grid_search_thresholds, create_lstm_dataset
from tradepy.supervised.posttrain import evaluate_model_classification, tune_threshold_wf, save_trading_model_2level, load_trading_model_3level
from tradepy.paths import MODELS_DIR, BT_DIR

import pandas as pd
import numpy as np

import torch

def load_bt_csv(ticker):
    data = pd.read_parquet(f"{BT_DIR}/{ticker}1min.parquet")
    data["openint"] = 0
    data["datetime"] = pd.to_datetime(data["time"], utc=True)
    # Cargar specs
    specs = load_specs()
    ticker = ticker.upper()
    if ticker not in specs:
        raise KeyError(f"No se encuentra spec para '{ticker}' en los JSON de info.")
    data = data.sort_values('datetime').drop_duplicates(subset='datetime').reset_index(drop=True)
    return data, specs[ticker]
    # return data
    
def prepare_bt(symbol, minutes, return_horizon_min, json_config_path="features_config.json"):
    df, spec = load_bt_csv(symbol)
    df = add_trading_date_by_gap(df)
    df_resampled = resample_ohlcv(df, period=f"{minutes}min")
    df_cumulative = daily_ohlcv_cummulative(df_resampled)
    config = load_feature_config()

    config["global"]["sampling_minutes"] = minutes
    config["global"]["return_horizon_min"] = return_horizon_min
    config["global"]["tick_size"] = spec["tick_size"]
    
    df_final = generate_features(df_cumulative, config)
    df_final["ticker"] = symbol
    return df_final, spec

def create_test_dataset(df, features, lookback=30):
    """
    Crea el dataset para LSTM usando NumPy strides (ultra rápido)
    y devuelve también los datetimes objetivo.
    
    Returns:
        X: np.ndarray de shape (num_samples, lookback, num_features)
        target_dt: pd.Series con los datetimes a predecir
    """
    # 1. Convertimos features a NumPy array
    feature_array = df[features].values
    
    # 2. Calculamos dimensiones
    num_samples = len(df) - lookback
    num_features = len(features)
    
    # 3. Crear ventanas con strides
    shape = (num_samples, lookback, num_features)
    strides = (feature_array.strides[0], feature_array.strides[0], feature_array.strides[1])
    
    X = np.lib.stride_tricks.as_strided(feature_array, shape=shape, strides=strides)
    
    # 4. Datetimes objetivo (los timestamps que vas a predecir)
    target_dt = df["datetime"].iloc[lookback:].reset_index(drop=True)
    
    return X, target_dt

def predict_in_batches(model, X_scaled, batch_size=512, device="cuda"):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = X_scaled[i:i+batch_size]
            batch_t = torch.tensor(batch, dtype=torch.float32).to(device)
            logits = model(batch_t)
            preds.append(logits.cpu())

    return torch.cat(preds, dim=0)

def run_backtest_complete(df, specs, initial_capital=1_000_000,
                          tp_mult=1.5, sl_mult=1.0):
    """
    df debe contener:
        - datetime
        - open, high, low, close
        - atr
        - pred_soft_final (0 short, 1 hold, 2 long)

    specs debe contener:
        - tick_size
        - tick_value
        - min_slippage_ticks
        - max_spread_ticks
        - initial_margin
        - maint_margin
    """

    df = df.sort_values("datetime").reset_index(drop=True)

    tick_size = specs["tick_size"]
    tick_value = specs["tick_value"]
    spread_ticks = specs["max_spread_ticks"]
    slip_ticks = specs["min_slippage_ticks"]

    initial_margin = specs["initial_margin"]
    maint_margin = specs["maint_margin"]

    # ============================================================
    # Estado del backtest
    # ============================================================
    equity = initial_capital
    position = None
    entry_price = None
    sl = None
    tp = None

    trades = []
    equity_curve = []

    def round_to_tick(price):
        return round(price / tick_size) * tick_size

    # ============================================================
    # Loop principal
    # ============================================================
    for i, row in df.iterrows():

        # ============================================================
        # 0. Margin Call: si equity < mantenimiento → cerrar posición
        # ============================================================
        if position is not None and equity < maint_margin:

            # cierre forzado al precio de mercado con slippage
            if position == "long":
                exit_price = row["open"] - slip_ticks * tick_size
            else:
                exit_price = row["open"] + slip_ticks * tick_size

            exit_price = round_to_tick(exit_price)

            if position == "long":
                pnl_points = exit_price - entry_price
            else:
                pnl_points = entry_price - exit_price

            pnl_usd = (pnl_points / tick_size) * tick_value
            equity += pnl_usd

            trades.append({
                "type": position,
                "entry": entry_price,
                "exit": exit_price,
                "pnl_points": pnl_points,
                "pnl_usd": pnl_usd,
                "datetime_exit": row["datetime"],
                "forced_exit": True
            })

            position = None

        # ============================================================
        # 1. Si hay posición abierta → comprobar SL/TP
        # ============================================================
        if position is not None:

            if position == "long":

                # SL-first
                if row["low"] <= sl:

                    exit_price = sl - slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)

                    pnl_points = exit_price - entry_price
                    pnl_usd = (pnl_points / tick_size) * tick_value

                    trades.append({
                        "type": "long",
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "datetime_exit": row["datetime"]
                    })

                    equity += pnl_usd
                    position = None

                # TP-second
                elif row["high"] >= tp:

                    exit_price = tp - slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)

                    pnl_points = exit_price - entry_price
                    pnl_usd = (pnl_points / tick_size) * tick_value

                    trades.append({
                        "type": "long",
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "datetime_exit": row["datetime"]
                    })

                    equity += pnl_usd
                    position = None

            elif position == "short":

                # SL-first
                if row["high"] >= sl:

                    exit_price = sl + slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)

                    pnl_points = entry_price - exit_price
                    pnl_usd = (pnl_points / tick_size) * tick_value

                    trades.append({
                        "type": "short",
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "datetime_exit": row["datetime"]
                    })

                    equity += pnl_usd
                    position = None

                # TP-second
                elif row["low"] <= tp:

                    exit_price = tp + slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)

                    pnl_points = entry_price - exit_price
                    pnl_usd = (pnl_points / tick_size) * tick_value

                    trades.append({
                        "type": "short",
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "datetime_exit": row["datetime"]
                    })

                    equity += pnl_usd
                    position = None

        # ============================================================
        # 2. Si NO hay posición abierta → comprobar señal nueva
        # ============================================================
        if position is None:

            # No abrir si no hay margen suficiente
            if equity < initial_margin:
                equity_curve.append(equity)
                continue

            signal = row["pred_soft_final"]

            if signal == 2:  # LONG

                raw_entry = row["open"] + spread_ticks * tick_size
                raw_entry += slip_ticks * tick_size
                entry_price = round_to_tick(raw_entry)

                sl = round_to_tick(entry_price - sl_mult * row["atr"])
                tp = round_to_tick(entry_price + tp_mult * row["atr"])

                position = "long"

            elif signal == 0:  # SHORT

                raw_entry = row["open"] - spread_ticks * tick_size
                raw_entry -= slip_ticks * tick_size
                entry_price = round_to_tick(raw_entry)

                sl = round_to_tick(entry_price + sl_mult * row["atr"])
                tp = round_to_tick(entry_price - tp_mult * row["atr"])

                position = "short"

        # ============================================================
        # 3. Guardar equity
        # ============================================================
        equity_curve.append(equity)

    return trades, equity_curve

def run_backtest_complete_scalein(df, specs, initial_capital=1_000_000,
                         tp_mult=1.5, sl_mult=1.0, max_positions=5):
    """
    Backtest con scale-in de hasta 5 posiciones (1 contrato por entrada).
    Cada posición tiene su propio SL y TP.
    """

    df = df.sort_values("datetime").reset_index(drop=True)

    tick_size = specs["tick_size"]
    tick_value = specs["tick_value"]
    spread_ticks = specs["max_spread_ticks"]
    slip_ticks = specs["min_slippage_ticks"]

    initial_margin = specs["initial_margin"]
    maint_margin = specs["maint_margin"]

    # ============================================================
    # Estado del backtest
    # ============================================================
    equity = initial_capital
    positions = []   # lista de dicts: {type, entry, sl, tp}
    trades = []
    equity_curve = []

    def round_to_tick(price):
        return round(price / tick_size) * tick_size

    # ============================================================
    # Loop principal
    # ============================================================
    for i, row in df.iterrows():

        # ============================================================
        # 0. Margin Call: equity < mantenimiento → cerrar TODO
        # ============================================================
        if positions and equity < maint_margin:

            for pos in positions:
                if pos["type"] == "long":
                    exit_price = row["open"] - slip_ticks * tick_size
                    pnl_points = exit_price - pos["entry"]
                else:
                    exit_price = row["open"] + slip_ticks * tick_size
                    pnl_points = pos["entry"] - exit_price

                exit_price = round_to_tick(exit_price)
                pnl_usd = (pnl_points / tick_size) * tick_value
                equity += pnl_usd

                trades.append({
                    "type": pos["type"],
                    "entry": pos["entry"],
                    "exit": exit_price,
                    "pnl_points": pnl_points,
                    "pnl_usd": pnl_usd,
                    "datetime_exit": row["datetime"],
                    "forced_exit": True
                })

            positions = []

        # ============================================================
        # 1. Gestionar SL/TP de cada posición abierta
        # ============================================================
        new_positions = []
        for pos in positions:

            closed = False

            if pos["type"] == "long":

                # SL-first
                if row["low"] <= pos["sl"]:
                    exit_price = pos["sl"] - slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)
                    pnl_points = exit_price - pos["entry"]
                    closed = True

                # TP-second
                elif row["high"] >= pos["tp"]:
                    exit_price = pos["tp"] - slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)
                    pnl_points = exit_price - pos["entry"]
                    closed = True

            else:  # SHORT

                # SL-first
                if row["high"] >= pos["sl"]:
                    exit_price = pos["sl"] + slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)
                    pnl_points = pos["entry"] - exit_price
                    closed = True

                # TP-second
                elif row["low"] <= pos["tp"]:
                    exit_price = pos["tp"] + slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)
                    pnl_points = pos["entry"] - exit_price
                    closed = True

            if closed:
                pnl_usd = (pnl_points / tick_size) * tick_value
                equity += pnl_usd

                trades.append({
                    "type": pos["type"],
                    "entry": pos["entry"],
                    "exit": exit_price,
                    "pnl_points": pnl_points,
                    "pnl_usd": pnl_usd,
                    "datetime_exit": row["datetime"]
                })

            else:
                new_positions.append(pos)

        positions = new_positions

        # ============================================================
        # 2. Entradas nuevas (scale-in)
        # ============================================================
        signal = row["pred_soft_final"]

        # No abrir si no hay margen
        if equity >= initial_margin:

            # LONG
            if signal == 2:

                # Si no hay posiciones o todas son long → scale-in
                if len(positions) < max_positions and \
                   (not positions or positions[0]["type"] == "long"):

                    raw_entry = row["open"] + spread_ticks * tick_size
                    raw_entry += slip_ticks * tick_size
                    entry_price = round_to_tick(raw_entry)

                    sl = round_to_tick(entry_price - sl_mult * row["atr"])
                    tp = round_to_tick(entry_price + tp_mult * row["atr"])

                    positions.append({
                        "type": "long",
                        "entry": entry_price,
                        "sl": sl,
                        "tp": tp
                    })

            # SHORT
            elif signal == 0:

                if len(positions) < max_positions and \
                   (not positions or positions[0]["type"] == "short"):

                    raw_entry = row["open"] - spread_ticks * tick_size
                    raw_entry -= slip_ticks * tick_size
                    entry_price = round_to_tick(raw_entry)

                    sl = round_to_tick(entry_price + sl_mult * row["atr"])
                    tp = round_to_tick(entry_price - tp_mult * row["atr"])

                    positions.append({
                        "type": "short",
                        "entry": entry_price,
                        "sl": sl,
                        "tp": tp
                    })

        # ============================================================
        # 3. Guardar equity
        # ============================================================
        equity_curve.append(equity)

    return trades, equity_curve

def run_backtest_sizing_only(df, specs, initial_capital=1_000_000,
                             tp_mult=1.5, sl_mult=1.0,
                             risk_per_trade=1.0, max_size_units=5):
    """
    Backtest SIN scale-in pero CON sizing dinámico basado en ATR.
    - Solo se abre UNA posición por señal.
    - El tamaño se calcula como risk_per_trade / ATR.
    - El tamaño máximo permitido es max_size_units.
    - Usa initial_margin y maint_margin.
    - Cierre forzado si equity < maint_margin.
    """

    df = df.sort_values("datetime").reset_index(drop=True)

    tick_size = specs["tick_size"]
    tick_value = specs["tick_value"]
    initial_margin = specs["initial_margin"]
    maint_margin = specs["maint_margin"]

    open_position = None
    closed_trades = []
    equity_curve = []

    equity = initial_capital

    for i, row in df.iterrows():

        # ============================================================
        # 0. Margin Call
        # ============================================================
        if open_position is not None and equity < maint_margin:

            pos = open_position
            exit_price = row["open"]

            pnl_points = (exit_price - pos["entry"]) if pos["type"] == "long" else (pos["entry"] - exit_price)
            pnl_usd = (pnl_points / tick_size) * tick_value * pos["size"]

            equity += pnl_usd

            closed_trades.append({
                "type": pos["type"],
                "entry": pos["entry"],
                "exit": exit_price,
                "pnl_points": pnl_points,
                "pnl_usd": pnl_usd,
                "size": pos["size"],
                "datetime_exit": row["datetime"],
                "forced_exit": True
            })

            open_position = None

        # ============================================================
        # 1. Actualizar posición abierta
        # ============================================================
        if open_position is not None:

            pos = open_position
            closed = False

            if pos["type"] == "long":

                if row["low"] <= pos["sl"]:  # SL-first
                    exit_price = pos["sl"]
                    closed = True

                elif row["high"] >= pos["tp"]:  # TP-second
                    exit_price = pos["tp"]
                    closed = True

            else:  # SHORT

                if row["high"] >= pos["sl"]:  # SL-first
                    exit_price = pos["sl"]
                    closed = True

                elif row["low"] <= pos["tp"]:  # TP-second
                    exit_price = pos["tp"]
                    closed = True

            if closed:
                pnl_points = (exit_price - pos["entry"]) if pos["type"] == "long" else (pos["entry"] - exit_price)
                pnl_usd = (pnl_points / tick_size) * tick_value * pos["size"]

                equity += pnl_usd

                closed_trades.append({
                    "type": pos["type"],
                    "entry": pos["entry"],
                    "exit": exit_price,
                    "pnl_points": pnl_points,
                    "pnl_usd": pnl_usd,
                    "size": pos["size"],
                    "datetime_exit": row["datetime"]
                })

                open_position = None

        # ============================================================
        # 2. Abrir nueva posición
        # ============================================================
        if open_position is None:

            if equity < initial_margin:
                equity_curve.append(equity)
                continue

            signal = row["pred_soft_final"]

            if signal in (0, 2):

                entry = row["open"]
                atr = row["atr"]

                size = min(max_size_units, risk_per_trade / atr)

                if signal == 2:  # LONG
                    sl = entry - sl_mult * atr
                    tp = entry + tp_mult * atr
                    pos_type = "long"

                else:  # SHORT
                    sl = entry + sl_mult * atr
                    tp = entry - tp_mult * atr
                    pos_type = "short"

                open_position = {
                    "type": pos_type,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "size": size,
                    "datetime_entry": row["datetime"]
                }

        # ============================================================
        # 3. Guardar equity
        # ============================================================
        equity_curve.append(equity)

    return closed_trades, equity_curve

def run_backtest_scalein_sizing(df, specs,
                                initial_capital=1_000_000,
                                tp_mult=1.5, sl_mult=1.0,
                                max_scalein=25,
                                max_size_per_entry=5,
                                risk_per_unit_usd=100):
    """
    df:
        datetime, open, high, low, close, atr, pred_soft_final (0 short, 1 hold, 2 long)

    specs:
        tick_size, tick_value, min_slippage_ticks, max_spread_ticks,
        initial_margin, maint_margin
    """

    df = df.sort_values("datetime").reset_index(drop=True)

    tick_size = specs["tick_size"]
    tick_value = specs["tick_value"]
    spread_ticks = specs["max_spread_ticks"]
    slip_ticks = specs["min_slippage_ticks"]

    initial_margin = specs["initial_margin"]
    maint_margin = specs["maint_margin"]

    # ============================================================
    # Estado del backtest
    # ============================================================
    equity = initial_capital
    positions = []   # lista de posiciones independientes
    trades = []
    equity_curve = []

    def round_to_tick(price):
        return round(price / tick_size) * tick_size

    # ============================================================
    # Loop principal
    # ============================================================
    for i, row in df.iterrows():

        # ============================================================
        # 0. Margin Call → cerrar TODAS las posiciones
        # ============================================================
        if positions and equity < maint_margin:

            for pos in positions:
                if pos["type"] == "long":
                    exit_price = row["open"] - slip_ticks * tick_size
                else:
                    exit_price = row["open"] + slip_ticks * tick_size

                exit_price = round_to_tick(exit_price)

                if pos["type"] == "long":
                    pnl_points = exit_price - pos["entry"]
                else:
                    pnl_points = pos["entry"] - exit_price

                pnl_usd = (pnl_points / tick_size) * tick_value * pos["size"]
                equity += pnl_usd

                trades.append({
                    "type": pos["type"],
                    "entry": pos["entry"],
                    "exit": exit_price,
                    "pnl_points": pnl_points,
                    "pnl_usd": pnl_usd,
                    "size": pos["size"],
                    "datetime_exit": row["datetime"],
                    "forced_exit": True
                })

            positions = []

        # ============================================================
        # 1. Evaluar SL/TP de TODAS las posiciones abiertas
        # ============================================================
        still_open = []
        for pos in positions:

            closed = False

            if pos["type"] == "long":

                # SL-first
                if row["low"] <= pos["sl"]:
                    exit_price = pos["sl"] - slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)

                    pnl_points = exit_price - pos["entry"]
                    pnl_usd = (pnl_points / tick_size) * tick_value * pos["size"]
                    equity += pnl_usd

                    trades.append({
                        "type": "long",
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "size": pos["size"],
                        "datetime_exit": row["datetime"]
                    })

                    closed = True

                # TP-second
                elif row["high"] >= pos["tp"]:
                    exit_price = pos["tp"] - slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)

                    pnl_points = exit_price - pos["entry"]
                    pnl_usd = (pnl_points / tick_size) * tick_value * pos["size"]
                    equity += pnl_usd

                    trades.append({
                        "type": "long",
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "size": pos["size"],
                        "datetime_exit": row["datetime"]
                    })

                    closed = True

            else:  # SHORT

                # SL-first
                if row["high"] >= pos["sl"]:
                    exit_price = pos["sl"] + slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)

                    pnl_points = pos["entry"] - exit_price
                    pnl_usd = (pnl_points / tick_size) * tick_value * pos["size"]
                    equity += pnl_usd

                    trades.append({
                        "type": "short",
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "size": pos["size"],
                        "datetime_exit": row["datetime"]
                    })

                    closed = True

                # TP-second
                elif row["low"] <= pos["tp"]:
                    exit_price = pos["tp"] + slip_ticks * tick_size
                    exit_price = round_to_tick(exit_price)

                    pnl_points = pos["entry"] - exit_price
                    pnl_usd = (pnl_points / tick_size) * tick_value * pos["size"]
                    equity += pnl_usd

                    trades.append({
                        "type": "short",
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "size": pos["size"],
                        "datetime_exit": row["datetime"]
                    })

                    closed = True

            if not closed:
                still_open.append(pos)

        positions = still_open

        # ============================================================
        # 2. Si hay señal y se puede abrir → scale-in + sizing
        # ============================================================
        if len(positions) < max_scalein and equity >= initial_margin:

            signal = row["pred_soft_final"]

            if signal in (0, 2):

                # -----------------------------
                # Dynamic sizing (máx 5 unidades)
                # -----------------------------
                atr = row["atr"]
                if atr <= 0:
                    size = 1
                else:
                    size = int(min(max_size_per_entry,
                                   max(1, risk_per_unit_usd / (atr * tick_value))))
                # -----------------------------

                if signal == 2:  # LONG
                    raw_entry = row["open"] + spread_ticks * tick_size
                    raw_entry += slip_ticks * tick_size
                    entry_price = round_to_tick(raw_entry)

                    sl = round_to_tick(entry_price - sl_mult * atr)
                    tp = round_to_tick(entry_price + tp_mult * atr)

                    pos_type = "long"

                elif signal == 0:  # SHORT
                    raw_entry = row["open"] - spread_ticks * tick_size
                    raw_entry -= slip_ticks * tick_size
                    entry_price = round_to_tick(raw_entry)

                    sl = round_to_tick(entry_price + sl_mult * atr)
                    tp = round_to_tick(entry_price - tp_mult * atr)

                    pos_type = "short"

                positions.append({
                    "type": pos_type,
                    "entry": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "size": size
                })

        # ============================================================
        # 3. Guardar equity
        # ============================================================
        equity_curve.append(equity)

    return trades, equity_curve

def build_overlapping_blocks(series, block_size):
    n = len(series)

    if block_size > n:
        block_size = n  # máximo posible

    if block_size <= 0:
        raise ValueError("block_size inválido")

    blocks = []
    for start in range(n - block_size + 1):
        blocks.append(series[start:start + block_size])

    return blocks, block_size

def monte_carlo_blocks(blocks, target_length, n_sims=1000):
    if len(blocks) == 0:
        raise ValueError("No hay bloques disponibles.")

    block_size = len(blocks[0])
    sims = []

    for _ in range(n_sims):
        pnl = []
        while len(pnl) < target_length:
            block = blocks[np.random.randint(0, len(blocks))]
            pnl.extend(block)
        sims.append(np.array(pnl[:target_length]))

    return np.array(sims)