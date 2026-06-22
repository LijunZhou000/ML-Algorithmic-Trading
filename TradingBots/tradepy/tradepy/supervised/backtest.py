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

import matplotlib.pyplot as plt

import os
import json

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

                atr_usd = (atr / tick_size) * tick_value          # ATR en dólares
                size = int(min(max_size_units, max(1, risk_per_trade / atr_usd)))

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

def plot_equities(curvas, asset, start, strat_name, show=True, save=False, save_path="classification_report.png"):
    fig, ax = plt.subplots(figsize=(12, 5))

    labels = ["Normal", "Scale in", "Sizing", "Scale in + Sizing"]

    # Estilos de línea tipo "dash pattern" como en el ejemplo
    estilos = [
        (0, (3, 6)),   # estilo 1
        (3, (3, 6)),   # estilo 2
        (6, (3, 6)),   # estilo 3
        (0, (1, 1))    # estilo 4 (punteado fino)
    ]

    colores = ["r", "g", "b", "k"]  # rojo, verde, azul, negro

    for curva, label, estilo, color in zip(curvas, labels, estilos, colores):
        linea, = ax.plot(
            curva,
            color=color,
            linestyle=estilo,
            linewidth=2,
            label=label
        )

        # Tag al final de la curva
        ax.text(
            x=len(curva)-1,
            y=curva[-1],
            s=label,
            fontsize=9,
            va='center',
            ha='left',
            color=color
        )

    ax.set_title(f"Comparación de curvas {asset}, desde {start}, estrategia {strat_name}")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save:
        fig.savefig(save_path, dpi=300)
        print(f"💾 Figura guardada en: {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)

def max_drawdown(equity_row):
    peak = np.maximum.accumulate(equity_row)
    dd = (equity_row - peak) / peak
    return dd.min()

def max_losing_streak(pnl_row):
    streak = max_streak = 0
    for p in pnl_row:
        streak = streak + 1 if p < 0 else 0
        max_streak = max(max_streak, streak)
    return max_streak

def plot_conjunto(weekly_pnl, mc, mdd_per_sim, asset, start, strat_name, show=True, save=False, save_path="classification_report.png"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    plt.style.use('seaborn-v0_8-whitegrid')

    # ── Datos ──────────────────────────────────────────────────────────────────────
    real_equity = weekly_pnl.cumsum()
    mc_equity   = mc.cumsum(axis=1)
    x = range(mc_equity.shape[1])

    # Percentiles temporales (para el abanico)
    p5  = np.percentile(mc_equity, 5,  axis=0)
    p25 = np.percentile(mc_equity, 25, axis=0)
    p50 = np.percentile(mc_equity, 50, axis=0)
    p75 = np.percentile(mc_equity, 75, axis=0)
    p95 = np.percentile(mc_equity, 95, axis=0)

    # Percentiles del resultado final
    final_equity = mc_equity[:, -1]
    fe_p5,  fe_p25  = np.percentile(final_equity, [5,  25])
    fe_p50, fe_p75  = np.percentile(final_equity, [50, 75])
    fe_p95          = np.percentile(final_equity, 95)

    # Percentiles del MDD
    mdd_pct = mdd_per_sim * 100
    mdd_p5,  mdd_p25  = np.percentile(mdd_pct, [5,  25])
    mdd_p50, mdd_p75  = np.percentile(mdd_pct, [50, 75])
    mdd_p95           = np.percentile(mdd_pct, 95)

    # ── Layout 2×2: fila superior ocupa las dos columnas ──────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax_mc   = fig.add_subplot(gs[0, :])   # fila 0, todas las columnas
    ax_fe   = fig.add_subplot(gs[1, 0])   # fila 1, columna 0
    ax_mdd  = fig.add_subplot(gs[1, 1])   # fila 1, columna 1

    # ── Panel superior: Equity curves ─────────────────────────────────────────────
    for i in range(300):
        ax_mc.plot(x, mc_equity[i], color='steelblue', alpha=0.04, lw=0.5)

    ax_mc.fill_between(x, p5,  p95,  alpha=0.10, color='steelblue', label='p5-p95')
    ax_mc.fill_between(x, p25, p75,  alpha=0.20, color='steelblue', label='p25-p75')
    ax_mc.plot(x, p50, color='steelblue', lw=1.2, ls='--', alpha=0.7, label='Mediana simulada')

    # halo blanco + curva real encima
    ax_mc.plot(x, real_equity, color='white',   lw=5,   zorder=4, alpha=0.6)
    ax_mc.plot(x, real_equity, color='#e63946', lw=2.5, zorder=5, label='Real')

    # punto y anotación al final
    final_val = real_equity.iloc[-1] if hasattr(real_equity, 'iloc') else real_equity[-1]
    ax_mc.scatter(x[-1], final_val, color='#e63946', zorder=6, s=60)
    ax_mc.annotate(f'${final_val:,.0f}',
                   xy=(x[-1], final_val),
                   xytext=(8, 0), textcoords='offset points',
                   va='center', color='#e63946', fontsize=9, fontweight='bold')

    ax_mc.axhline(0, color='black', ls=':', lw=1, alpha=0.5)
    ax_mc.set_title(f'Monte Carlo — Curva equity diaria ({asset}, n={len(mc_equity)}, desde {start}, estrategia {strat_name})', fontsize=13)
    ax_mc.set_xlabel('Semanas')
    ax_mc.set_ylabel('PnL acumulado (USD)')
    ax_mc.legend(fontsize=9)

    # ── Panel inferior izquierdo: Distribución resultado final ────────────────────
    ax_fe.hist(final_equity, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
    ax_fe.axvline(fe_p5,  color='salmon',          ls='--', lw=1,   label=f'p5  = ${fe_p5:,.0f}')
    ax_fe.axvline(fe_p25, color='orange',          ls='--', lw=1,   label=f'p25 = ${fe_p25:,.0f}')
    ax_fe.axvline(fe_p50, color='steelblue',       ls='-',  lw=1.5, label=f'p50 = ${fe_p50:,.0f}')
    ax_fe.axvline(fe_p75, color='mediumseagreen',  ls='--', lw=1,   label=f'p75 = ${fe_p75:,.0f}')
    ax_fe.axvline(fe_p95, color='seagreen',        ls='--', lw=1,   label=f'p95 = ${fe_p95:,.0f}')
    ax_fe.axvline(final_val, color='#e63946',      ls='-',  lw=2,   label=f'Real = ${final_val:,.0f}')
    ax_fe.set_title('Distribución resultado final')
    ax_fe.set_xlabel('PnL acumulado (USD)')
    ax_fe.set_ylabel('Frecuencia')
    ax_fe.legend(fontsize=8)

    # ── Panel inferior derecho: Distribución Max Drawdown ─────────────────────────
    ax_mdd.hist(mdd_pct, bins=60, color='steelblue', edgecolor='none', alpha=0.8)
    ax_mdd.axvline(mdd_p5,  color='salmon',         ls='--', lw=1,   label=f'p5  = {mdd_p5:.1f}%')
    ax_mdd.axvline(mdd_p25, color='orange',         ls='--', lw=1,   label=f'p25 = {mdd_p25:.1f}%')
    ax_mdd.axvline(mdd_p50, color='steelblue',      ls='-',  lw=1.5, label=f'p50 = {mdd_p50:.1f}%')
    ax_mdd.axvline(mdd_p75, color='mediumseagreen', ls='--', lw=1,   label=f'p75 = {mdd_p75:.1f}%')
    ax_mdd.axvline(mdd_p95, color='seagreen',       ls='--', lw=1,   label=f'p95 = {mdd_p95:.1f}%')
    ax_mdd.set_title('Distribución Max Drawdown')
    ax_mdd.set_xlabel('%')
    ax_mdd.set_ylabel('Frecuencia')
    ax_mdd.legend(fontsize=8)

    # plt.savefig('monte_carlo.pdf', dpi=150, bbox_inches='tight')
    if save:
        fig.savefig(save_path, dpi=300)
        print(f"💾 Figura guardada en: {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)

# ── helper ────────────────────────────────────────────────────────────────────

def _calmar_metrics(equity_series) -> dict:
    equity = pd.Series(equity_series, dtype=float).dropna().reset_index(drop=True)

    if len(equity) < 2:
        return {
            "calmar": -np.inf,
            "sharpe": -np.inf,
            "max_dd": np.inf,
            "cagr": -np.inf,
        }

    start_eq = float(equity.iloc[0])
    end_eq = float(equity.iloc[-1])

    if start_eq <= 0:
        return {
            "calmar": -np.inf,
            "sharpe": -np.inf,
            "max_dd": np.inf,
            "cagr": -np.inf,
        }

    # Estrategia quebrada
    if end_eq <= 0:
        return {
            "calmar": -np.inf,
            "sharpe": -np.inf,
            "max_dd": 1.0,
            "cagr": -1.0,
        }

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    months = max(len(equity) / 21.0, 1e-9)

    ratio = end_eq / start_eq
    cagr = ratio ** (12.0 / months) - 1.0

    roll_max = equity.cummax()
    drawdown = (roll_max - equity) / roll_max.replace(0, np.nan)
    max_dd = float(drawdown.max())

    vol_ann = float(returns.std() * np.sqrt(252))

    if not np.isfinite(vol_ann) or vol_ann <= 0:
        sharpe = 0.0 if cagr >= 0 else -np.inf
    else:
        returns = equity.pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)

    calmar = cagr / (max_dd + 1e-9)

    metrics = {
        "calmar": float(calmar),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "cagr": float(cagr),
    }

    # Evitar NaN o inf inesperados
    for k, v in metrics.items():
        if not np.isfinite(v):
            metrics[k] = -np.inf if k != "max_dd" else np.inf

    return metrics

# ── función principal ─────────────────────────────────────────────────────────

def backtest_grid_search(models_dir, asset, minutes, return_horizon_min, start, end, show, save):

    # ── carga de modelos ──────────────────────────────────────────────────────
    (
        model_l1, model_l2, model_l3,
        sc_features, sc_l3,
        features, params, thresholds, df_results
    ) = load_trading_model_3level(
        BasicLSTM_L1_Move, BasicLSTM_L2_Dir, BasicLSTM_L3_Regression,
        path=f"{models_dir}/{asset.lower()}/trading_model_lstm_3level",
        device="cuda"
    )
    (
        model_l1_g, model_l2_g, model_l3_g,
        sc_features_g, sc_l3_g,
        features_g, params_g, thresholds_g, df_results_g
    ) = load_trading_model_3level(
        BasicGRU_L1_Move, BasicGRU_L2_Dir, BasicGRU_L3_Regression,
        path=f"{models_dir}/{asset.lower()}/trading_model_gru_3level",
        device="cuda"
    )

    # ── preparar datos ────────────────────────────────────────────────────────
    df_bt, spec_bt = prepare_bt(asset, minutes, return_horizon_min,
                                json_config_path="../Data/features_config.json")
    mask = (df_bt["datetime"] >= start) & (df_bt["datetime"] < end)
    df_filtered = df_bt.loc[mask].copy()
    df_filtered["hour"] = df_filtered["datetime"].dt.hour

    # ── inferencia LSTM ───────────────────────────────────────────────────────
    test, test_dt = create_test_dataset(df_filtered, features, lookback=30)
    test_flat   = test.reshape(-1, test.shape[-1])
    test_scaled = sc_features.transform(test_flat).reshape(test.shape)

    probs_l1 = torch.softmax(predict_in_batches(model_l1, test_scaled), dim=1).numpy()
    df_probs  = pd.DataFrame(probs_l1, columns=["prob_hold", "prob_move"])

    probs_l2 = torch.softmax(predict_in_batches(model_l2, test_scaled), dim=1).numpy()
    df_probs[["prob_short", "prob_long"]] = pd.DataFrame(probs_l2, columns=["prob_short", "prob_long"])

    preds_l3 = predict_in_batches(model_l3, test_scaled).numpy().reshape(-1)
    df_probs["pred_l3"] = preds_l3

    df_probs["pred_l1"] = (df_probs["prob_move"] > thresholds["threshold_move"]).astype(int)
    df_probs["pred_l2"] = (df_probs["prob_long"] > thresholds["threshold_dir"]).astype(int)
    df_probs["datetime"] = pd.to_datetime(test_dt.values, utc=True)
    df_probs["pred"] = np.where(
        df_probs["pred_l1"] == 0, 1,
        np.where(df_probs["pred_l2"] == 0, 0, 2)
    )

    # ── inferencia GRU ────────────────────────────────────────────────────────
    test_g, test_dt_g = create_test_dataset(df_filtered, features_g, lookback=30)
    test_flat_g   = test_g.reshape(-1, test_g.shape[-1])
    test_scaled_g = sc_features_g.transform(test_flat_g).reshape(test_g.shape)

    probs_l1_g = torch.softmax(predict_in_batches(model_l1_g, test_scaled_g), dim=1).numpy()
    df_probs_g  = pd.DataFrame(probs_l1_g, columns=["prob_hold", "prob_move"])

    probs_l2_g = torch.softmax(predict_in_batches(model_l2_g, test_scaled_g), dim=1).numpy()
    df_probs_g[["prob_short", "prob_long"]] = pd.DataFrame(probs_l2_g, columns=["prob_short", "prob_long"])

    preds_l3_g = predict_in_batches(model_l3_g, test_scaled_g).numpy().reshape(-1)
    df_probs_g["pred_l3"] = preds_l3_g

    df_probs_g["pred_l1"] = (df_probs_g["prob_move"] > thresholds_g["threshold_move"]).astype(int)
    df_probs_g["pred_l2"] = (df_probs_g["prob_long"] > thresholds_g["threshold_dir"]).astype(int)
    df_probs_g["datetime"] = pd.to_datetime(test_dt_g.values, utc=True)
    df_probs_g["pred"] = np.where(
        df_probs_g["pred_l1"] == 0, 1,
        np.where(df_probs_g["pred_l2"] == 0, 0, 2)
    )

    # ── soft-voting LSTM + GRU ────────────────────────────────────────────────
    df_merged_total = df_probs.merge(
        df_probs_g, on="datetime", how="left", suffixes=("_lstm", "_gru")
    )

    threshold_total = {k: min(thresholds[k], thresholds_g[k]) for k in thresholds}
    eps = 1e-9

    abs_l3_lstm = df_merged_total["pred_l3_lstm"].abs()
    abs_l3_gru  = df_merged_total["pred_l3_gru"].abs()
    w_lstm = abs_l3_lstm / (abs_l3_lstm + abs_l3_gru + eps)
    w_gru  = abs_l3_gru  / (abs_l3_lstm + abs_l3_gru + eps)

    df_merged_total["soft_move_score"] = (
        w_lstm * df_merged_total["prob_move_lstm"] +
        w_gru  * df_merged_total["prob_move_gru"]
    )
    df_merged_total["pred_soft_l1"] = (
        df_merged_total["soft_move_score"] > threshold_total["threshold_move"]
    ).astype(int)

    df_merged_total["sign_l3_lstm"] = np.sign(
        df_merged_total[df_merged_total.pred_l1_lstm == 1]["pred_l3_lstm"]
    )
    df_merged_total["sign_l3_gru"] = np.sign(
        df_merged_total[df_merged_total.pred_l1_lstm == 1]["pred_l3_gru"]
    )

    # diagnóstico de desacuerdo entre modelos
    df_moves = df_merged_total[df_merged_total.pred_soft_l1 == 1].copy()
    df_moves["sign_l3_lstm"] = np.sign(df_moves["pred_l3_lstm"])
    df_moves["sign_l3_gru"]  = np.sign(df_moves["pred_l3_gru"])
    total_moves       = len(df_moves)
    num_l3_disagree   = (df_moves["sign_l3_lstm"] != df_moves["sign_l3_gru"]).sum()
    num_l2_disagree   = (df_moves["pred_l2_lstm"] != df_moves["pred_l2_gru"]).sum()
    summary = pd.DataFrame({
        "metric":               ["L3_sign_disagree", "L2_direction_disagree"],
        "count":                [num_l3_disagree, num_l2_disagree],
        "total_moves_softvote": [total_moves, total_moves],
        "ratio":                [num_l3_disagree / max(total_moves, 1),
                                 num_l2_disagree / max(total_moves, 1)],
    })

    df_merged_total["soft_long_score"] = (
        w_lstm * df_merged_total["prob_long_lstm"] +
        w_gru  * df_merged_total["prob_long_gru"]
    )
    df_merged_total["soft_short_score"] = (
        w_lstm * df_merged_total["prob_short_lstm"] +
        w_gru  * df_merged_total["prob_short_gru"]
    )
    df_merged_total["pred_soft_l2"] = (
        df_merged_total["soft_long_score"] > threshold_total["threshold_dir"]
    ).astype(int)

    df_merged_total["l3_total"] = (
        w_lstm * df_merged_total["pred_l3_lstm"] +
        w_gru  * df_merged_total["pred_l3_gru"]
    )
    df_merged_total["sign_l3_total"] = np.sign(df_merged_total["l3_total"])
    df_merged_total["pred_soft_l2_clean"] = np.where(
        df_merged_total["sign_l3_total"] == 0,
        np.nan,
        df_merged_total["pred_soft_l2"]
    )
    df_merged_total["pred_soft_final"] = np.where(
        df_merged_total["pred_soft_l1"] == 0, 1,
        np.where(
            df_merged_total["pred_soft_l2_clean"].isna(), 1,
            np.where(df_merged_total["pred_soft_l2_clean"] == 0, 0, 2)
        )
    )

    # ── definición de estrategias de señal ───────────────────────────────────
    threshold_l3 = df_merged_total["l3_total"].abs().quantile(0.7)
    df_ops         = df_merged_total[df_merged_total["pred_soft_final"] != 1]
    df_long        = df_merged_total[df_merged_total["pred_soft_final"] == 2]
    df_long_strong = df_merged_total[
        (df_merged_total["pred_soft_final"] == 2) &
        (df_merged_total["l3_total"].abs() > threshold_l3)
    ]
    df_ops_strong  = df_merged_total[
        (df_merged_total["pred_soft_final"] != 1) &
        (df_merged_total["l3_total"].abs() > threshold_l3)
    ]

    strat_p1_to_try = {
        "df_ops":         df_ops,
        "df_long":        df_long,
        "df_long_strong": df_long_strong,
        "df_ops_strong":  df_ops_strong,
    }

    backtest_fns = {
        "base":            run_backtest_complete,
        "scalein":         run_backtest_complete_scalein,
        "sizing":          run_backtest_sizing_only,
        "scalein_sizing":  run_backtest_scalein_sizing,
    }

    # ── grid search con selección por Calmar ──────────────────────────────────
    specs           = load_specs()
    initial_capital = specs[asset]["initial_margin"] * 10
    all_results     = []   # lista plana de todas las combinaciones

    for strat_name, signal_df in strat_p1_to_try.items():
        df_for_bt = (
            df_filtered[["open", "close", "high", "low", "atr", "datetime"]]
            .merge(signal_df, how="outer", on="datetime")
        )

        equities = {}   # para plot_equities de esta señal
        trades_map = {}

        for bt_name, bt_fn in backtest_fns.items():
            trades, equity = bt_fn(df_for_bt, specs[asset],
                                   initial_capital=initial_capital)
            equities[bt_name]   = equity
            trades_map[bt_name] = trades

            metrics = _calmar_metrics(pd.Series(equity))
            all_results.append({
                "strat":    strat_name,
                "backtest": bt_name,
                "trades":   trades,
                "equity":   equity,
                **metrics,
            })

        # plot de las 4 variantes para esta señal (igual que antes)
        df_eq = pd.DataFrame(equities).astype(float)
        plot_equities(
            df_eq.values.T, asset, start, strat_name, show, save,
            f"{models_dir}/{asset.lower()}/equity_{strat_name}.png"
        )

    # ── elegir el mejor por Calmar (desempate: Sharpe) ────────────────────────
    best = max(all_results, key=lambda r: (r["calmar"], r["sharpe"]))

    best_equity = pd.Series(best["equity"]).dropna().reset_index(drop=True)
    best_trades = pd.DataFrame(best["trades"])

    print(
        f"\n[{asset}] mejor: {best['strat']} + {best['backtest']}"
        f"  |  Calmar={best['calmar']:.3f}"
        f"  |  Sharpe={best['sharpe']:.3f}"
        f"  |  MDD={best['max_dd']:.1%}"
        f"  |  CAGR={best['cagr']:.1%}"
        f"  |  PnL=${best_equity.iloc[-1] - initial_capital:,.0f}"
    )

    # ── Monte Carlo sobre el mejor ────────────────────────────────────────────
    best_trades["datetime_exit"] = pd.to_datetime(best_trades["datetime_exit"])
    best_trades = best_trades.sort_values("datetime_exit").reset_index(drop=True)
    best_trades["date"] = best_trades["datetime_exit"].dt.date
    daily      = best_trades.groupby("date")["pnl_usd"].sum().reset_index()
    weekly_pnl = daily["pnl_usd"].values

    blocks, block_size = build_overlapping_blocks(weekly_pnl, block_size=1)
    print("block_size usado:", block_size, " | n_blocks:", len(blocks))

    mc             = monte_carlo_blocks(blocks, len(weekly_pnl), n_sims=5000)
    pnl_cumsum     = np.cumsum(mc, axis=1)
    equity_curves  = 1_000_000 + pnl_cumsum
    final_equity   = equity_curves[:, -1]
    mdd_per_sim    = np.array([max_drawdown(eq) for eq in equity_curves])
    streaks        = np.array([max_losing_streak(row) for row in mc])

    print(f"Median MDD      : {np.median(mdd_per_sim):.1%}")
    print(f"Worst MDD (p5)  : {np.percentile(mdd_per_sim, 5):.1%}")
    print(f"Racha perd. med.: {np.median(streaks):.0f} trades")
    print(f"Racha perd. p95 : {np.percentile(streaks, 95):.0f} trades")

    plot_conjunto(
        weekly_pnl, mc, mdd_per_sim, asset, start, best['strat'], show, save,
        f"{models_dir}/{asset.lower()}/conjunto_best.png"
    )

    # ── persistencia ─────────────────────────────────────────────────────────
    asset_dir = f"{models_dir}/{asset.lower()}"
    os.makedirs(asset_dir, exist_ok=True)

    best_equity.to_frame("equity").to_parquet(
        f"{asset_dir}/equity_best.parquet", index=False
    )
    best_trades.to_parquet(
        f"{asset_dir}/trades_best.parquet", index=False
    )

    result_entry = {
        asset: {
            "strat":        best["strat"],
            "backtest":     best["backtest"],
            "calmar":       round(float(best["calmar"]), 4),
            "sharpe":       round(float(best["sharpe"]), 4),
            "max_dd":       round(float(best["max_dd"]), 4),
            "cagr":         round(float(best["cagr"]),   4),
            "equity_final": round(float(best_equity.iloc[-1]),   4),
            "pnl_final": round(float(best_equity.iloc[-1] - initial_capital),   4),
            "equity_path":  f"{asset_dir}/equity_best.parquet",
            "trades_path":  f"{asset_dir}/trades_best.parquet",
        }
    }

    json_path = f"{models_dir}/best_per_asset.json"
    # carga el JSON existente si ya hay resultados de otros assets
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}
    existing.update(result_entry)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"JSON actualizado: {json_path}")

    return result_entry