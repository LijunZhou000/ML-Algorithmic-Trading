import numpy as np
import pandas as pd

def target_triple_barrier_interday(
    df,
    spec,
    return_horizon_days=5,       # en días en vez de minutos
    atr_col="atr",
    tp_atr_mult=2.0,
    sl_atr_mult=1.2,
    profit_factor=2.5,
    use_swing_barriers=True,     # NUEVO: anclar barreras a swings
    swing_barrier_weight=0.5,    # peso entre ATR puro y nivel de swing
):
    df = df.copy()

    tick_size = float(spec["tick_size"])
    tick_value = float(spec["tick_value"])

    # En interday, max_steps = días (si tienes datos diarios, 1 row = 1 día)
    max_steps = max(1, return_horizon_days)

    fee_side = float(spec.get("approx_total_fee_per_side", 2.5))
    fees_rt_ticks = (fee_side * 2.0) / tick_value
    slip_ticks = float(spec.get("min_slippage_ticks", 1.0))
    spread_ticks = float(spec.get("min_spread_ticks", 1.0))
    total_cost_ticks = fees_rt_ticks + slip_ticks + spread_ticks

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df[atr_col].values

    # Swing levels (ya los tienes en tus features)
    has_swings = use_swing_barriers and all(
        c in df.columns for c in ["last_swing_high_price", "last_swing_low_price"]
    )
    if has_swings:
        swing_highs = df["last_swing_high_price"].values
        swing_lows = df["last_swing_low_price"].values

    labels = np.full(len(df), np.nan)

    for i in range(len(df) - max_steps):

        entry_price = close[i]
        atr_ticks = atr[i] / tick_size

        # Barreras base por ATR
        tp_ticks_atr = tp_atr_mult * atr_ticks
        sl_ticks_atr = sl_atr_mult * atr_ticks

        min_profit_ticks = total_cost_ticks * profit_factor
        tp_ticks_atr = max(tp_ticks_atr, min_profit_ticks)

        # NUEVO: ajuste por swings
        if has_swings and not np.isnan(swing_highs[i]) and not np.isnan(swing_lows[i]):
            dist_to_swing_high = (swing_highs[i] - entry_price) / tick_size
            dist_to_swing_low = (entry_price - swing_lows[i]) / tick_size

            # Mezcla ponderada entre ATR y distancia al swing
            # Si el swing está más cerca que el ATR, la barrera se contrae (más realista)
            tp_ticks = (
                swing_barrier_weight * max(dist_to_swing_high, min_profit_ticks)
                + (1 - swing_barrier_weight) * tp_ticks_atr
            )
            sl_ticks = (
                swing_barrier_weight * max(dist_to_swing_low, 1.0)
                + (1 - swing_barrier_weight) * sl_ticks_atr
            )
        else:
            tp_ticks = tp_ticks_atr
            sl_ticks = sl_ticks_atr

        upper_barrier = entry_price + tp_ticks * tick_size
        lower_barrier = entry_price - sl_ticks * tick_size

        label = 0  # HOLD por defecto

        for j in range(1, max_steps + 1):
            hit_tp = high[i + j] >= upper_barrier
            hit_sl = low[i + j] <= lower_barrier

            if hit_tp and hit_sl:
                label = -1  # conflicto intrabarra → peor caso
                break
            elif hit_tp:
                label = 1
                break
            elif hit_sl:
                label = -1
                break

        labels[i] = label

    df[f"target_tb_{return_horizon_days}d"] = pd.Series(labels, index=df.index).map({
        -1: 0,
        0: 1,
        1: 2,
    })

    return df.dropna(subset=[f"target_tb_{return_horizon_days}d"])