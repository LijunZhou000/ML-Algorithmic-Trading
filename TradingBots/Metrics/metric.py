import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def evaluate_double_lstm(
    movement_model: torch.nn.Module,
    direction_model: torch.nn.Module,
    scaler,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = 60,
    movement_threshold: float = 0.0005,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Evalúa ambos modelos (Movement + Direction) sobre un test set.
    Devuelve métricas de clasificación + métricas combinadas de la estrategia.
    """
    movement_model.eval()
    direction_model.eval()

    # Preparar features + targets reales
    test_scaled = scaler.transform(test_df[feature_cols])
    test_scaled_df = pd.DataFrame(test_scaled, columns=feature_cols, index=test_df.index)

    # Crear secuencias
    X_test, y_move_test = [], []
    y_dir_test = []
    for i in range(len(test_scaled_df) - sequence_length):
        seq = test_scaled_df.iloc[i:i+sequence_length].values
        X_test.append(seq)
        
        # Targets reales
        ret_next = test_df['close'].iloc[i+sequence_length] / test_df['close'].iloc[i+sequence_length-1] - 1
        move_real = 1 if abs(ret_next) > movement_threshold else 0
        dir_real = 1 if ret_next > 0 else 0
        
        y_move_test.append(move_real)
        if move_real == 1:
            y_dir_test.append(dir_real)
        else:
            y_dir_test.append(-1)  # placeholder para ignorar en métricas de direction

    X_test = torch.tensor(np.array(X_test, dtype=np.float32)).to(device)

    # === Inferencia Movement ===
    with torch.no_grad():
        pred_move = movement_model(X_test).cpu().numpy().flatten()
        pred_move_class = (pred_move > 0.5).astype(int)

    # === Inferencia Direction (solo cuando Movement predice 1) ===
    move_mask = pred_move_class == 1
    if move_mask.sum() > 0:
        X_dir_test = X_test[move_mask]
        with torch.no_grad():
            pred_dir = direction_model(X_dir_test).cpu().numpy().flatten()
            pred_dir_class = (pred_dir > 0.5).astype(int)
    else:
        pred_dir_class = np.array([])

    # === Métricas Classification ===
    metrics = {}

    # Movement
    metrics['movement'] = {
        'accuracy': accuracy_score(y_move_test, pred_move_class),
        'precision': precision_score(y_move_test, pred_move_class, zero_division=0),
        'recall': recall_score(y_move_test, pred_move_class, zero_division=0),
        'f1': f1_score(y_move_test, pred_move_class, zero_division=0),
        'confusion': confusion_matrix(y_move_test, pred_move_class).tolist()
    }

    # Direction (solo sobre las muestras donde movement real = 1)
    real_move_idx = [i for i, v in enumerate(y_move_test) if v == 1]
    if len(real_move_idx) > 0 and len(pred_dir_class) > 0:
        y_dir_real = [y_dir_test[i] for i in real_move_idx]
        metrics['direction'] = {
            'accuracy': accuracy_score(y_dir_real, pred_dir_class),
            'precision': precision_score(y_dir_real, pred_dir_class, zero_division=0),
            'recall': recall_score(y_dir_real, pred_dir_class, zero_division=0),
            'f1': f1_score(y_dir_real, pred_dir_class, zero_division=0),
        }
    else:
        metrics['direction'] = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    logger.info(f"Movement → Acc: {metrics['movement']['accuracy']:.3f} | F1: {metrics['movement']['f1']:.3f}")
    logger.info(f"Direction → Acc: {metrics['direction']['accuracy']:.3f} | F1: {metrics['direction']['f1']:.3f}")

    return metrics, pred_move_class, pred_dir_class


def run_vectorized_backtest(
    df: pd.DataFrame,
    movement_model: torch.nn.Module,
    direction_model: torch.nn.Module,
    scaler,
    feature_cols: list[str],
    initial_capital: float = 100_000,
    atr_period: int = 14,
    atr_trail_mult: float = 1.5,
    sl_puntos: float = 20.0,
    tp_puntos: float = 40.0,
    position_size_contracts: int = 1,
    sequence_length: int = 60,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Backtest vectorizado completo con la estrategia DOUBLE LSTM.
    Devuelve equity curve + todas las métricas de trading típicas.
    """
    df = df.copy()
    df['atr'] = calculate_atr(df, atr_period)

    # Preparar secuencias para inferencia
    scaled = scaler.transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)

    X = []
    for i in range(len(scaled_df) - sequence_length):
        X.append(scaled_df.iloc[i:i+sequence_length].values)
    X = torch.tensor(np.array(X, dtype=np.float32)).to(device)

    # Inferencia
    movement_model.eval()
    direction_model.eval()
    with torch.no_grad():
        pred_move = movement_model(X).cpu().numpy().flatten()
        pred_move_class = (pred_move > 0.5).astype(int)

        move_mask = pred_move_class == 1
        if move_mask.sum() > 0:
            pred_dir = direction_model(X[move_mask]).cpu().numpy().flatten()
            pred_dir_class = (pred_dir > 0.5).astype(int)
        else:
            pred_dir_class = np.array([])

    # Construir señal final
    signals = np.zeros(len(df) - sequence_length)
    signals[pred_move_class == 0] = 0          # HOLD
    signals[pred_move_class == 1] = 1          # MOVEMENT
    dir_idx = 0
    for i in range(len(signals)):
        if pred_move_class[i] == 1 and dir_idx < len(pred_dir_class):
            signals[i] = 1 if pred_dir_class[dir_idx] == 1 else -1
            dir_idx += 1

    # Backtest vectorizado
    equity = [initial_capital]
    position = 0
    entry_price = 0
    trades = []
    equity_curve = np.zeros(len(df))

    for i in range(sequence_length, len(df)):
        sig = signals[i - sequence_length]
        price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]

        # Cerrar posición si existe y hay señal contraria o HOLD
        if position != 0:
            if sig == 0 or (sig > 0 and position < 0) or (sig < 0 and position > 0):
                pnl = (price - entry_price) * position * 50   # 50$ por punto en ES/GC/CL (ajusta si usas 6B)
                equity[-1] += pnl
                trades.append(pnl)
                position = 0

        # Nueva entrada
        if position == 0 and sig != 0:
            position = sig * position_size_contracts
            entry_price = price
            sl = price - sl_puntos if sig > 0 else price + sl_puntos
            tp = price + tp_puntos if sig > 0 else price - tp_puntos

        # Trailing con ATR (solo si está en posición)
        if position != 0:
            if position > 0:
                new_sl = price - (atr * atr_trail_mult)
                if new_sl > sl:
                    sl = new_sl
            else:
                new_sl = price + (atr * atr_trail_mult)
                if new_sl < sl:
                    sl = new_sl

            # Check SL/TP
            if (position > 0 and price <= sl) or (position < 0 and price >= sl):
                pnl = (price - entry_price) * position * 50
                equity[-1] += pnl
                trades.append(pnl)
                position = 0
            elif (position > 0 and price >= tp) or (position < 0 and price <= tp):
                pnl = (price - entry_price) * position * 50
                equity[-1] += pnl
                trades.append(pnl)
                position = 0

        equity_curve[i] = equity[-1]

    # Métricas de trading
    returns = np.diff(equity_curve[equity_curve > 0]) / equity_curve[:-1][equity_curve[:-1] > 0]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60) if len(returns) > 1 else 0  # anualizado intraday
    max_dd = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve).max()
    win_rate = len([t for t in trades if t > 0]) / len(trades) if trades else 0
    profit_factor = sum(t for t in trades if t > 0) / abs(sum(t for t in trades if t < 0)) if any(t < 0 for t in trades) else float('inf')

    metrics = {
        'sharpe': float(sharpe),
        'max_drawdown_pct': float(max_dd * 100),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'total_trades': len(trades),
        'final_equity': float(equity[-1]),
        'total_return_pct': float((equity[-1] / initial_capital - 1) * 100),
        'equity_curve': equity_curve.tolist()
    }

    logger.info(f"BACKTEST → Sharpe: {metrics['sharpe']:.2f} | MaxDD: {metrics['max_drawdown_pct']:.2f}% | "
                f"WinRate: {metrics['win_rate']:.1f}% | PF: {metrics['profit_factor']:.2f} | Trades: {metrics['total_trades']}")

    return metrics, equity_curve


def calculate_atr(df: pd.DataFrame, period: int = 14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()
