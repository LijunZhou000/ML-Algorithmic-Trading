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

def run_backtest_complete(df, specs, tp_mult=1.5, sl_mult=1.0):
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
    """

    df = df.sort_values("datetime").reset_index(drop=True)

    tick_size = specs["tick_size"]
    tick_value = specs["tick_value"]
    spread_ticks = specs["max_spread_ticks"]
    slip_ticks = specs["min_slippage_ticks"]

    equity = 0.0
    position = None
    entry_price = None
    sl = None
    tp = None

    trades = []
    equity_curve = []

    def round_to_tick(price):
        return round(price / tick_size) * tick_size

    for i, row in df.iterrows():

        # ============================================================
        # 1. Si hay posición abierta → comprobar SL/TP
        # ============================================================
        if position is not None:

            if position == "long":

                # SL-first
                if row["low"] <= sl:

                    # aplicar slippage en salida
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

            signal = row["pred_soft_final"]

            if signal == 2:  # LONG

                # entrada al ask + slippage
                raw_entry = row["open"] + spread_ticks * tick_size
                raw_entry += slip_ticks * tick_size
                entry_price = round_to_tick(raw_entry)

                sl = round_to_tick(entry_price - sl_mult * row["atr"])
                tp = round_to_tick(entry_price + tp_mult * row["atr"])

                position = "long"

            elif signal == 0:  # SHORT

                # entrada al bid - slippage
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