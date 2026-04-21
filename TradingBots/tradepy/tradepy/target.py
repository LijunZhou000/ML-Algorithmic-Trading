import numpy as np
import pandas as pd

def create_targets(df, return_horizon_min, sampling_minutes, tick_size=None, ternary=False, threshold_buy=0.0, threshold_sell=None):
    """
    Parameters
    ----------
    threshold : float
        En ticks si se pasa tick_size, en log-return si no.
        Siempre positivo; se aplica simétricamente.
    """
    df = df.copy()
    shift = max(1, int(return_horizon_min / max(1, int(sampling_minutes))))
    suffix = f"{return_horizon_min}m"

    future_price = df["close"].shift(-shift)

    # --- Retornos continuos (siempre útiles como features) ---
    df[f"target_logret_{suffix}"] = np.log(future_price / df["close"])
    df[f"target_ret_{suffix}"]    = future_price / df["close"] - 1

    # --- Retorno "crudo" para clasificar ---
    if tick_size is not None:
        raw = (future_price - df["close"]) / float(tick_size)
        df[f"target_ticks_{suffix}"] = raw
    else:
        raw = df[f"target_logret_{suffix}"]

    thresh_buy = float(abs(threshold_buy))
    thresh_sell = float(abs(threshold_sell)) if threshold_sell is not None else thresh_buy

    # --- Binario: 1=BUY claro, 0=todo lo demás ---
    # (si quieres SELL explícito, mejor usa el ternario)
    df["target_bin"] = (raw > thresh_buy).astype("Int8")   # NaN-safe

    # --- Ternario: 0=SELL, 1=NEUTRAL, 2=BUY ---
    if ternary:
        df["target_class"] = np.select(
            condlist=[raw > thresh_buy, raw < -thresh_sell],
            choicelist=[2, 0],
            default=1
        ).astype(float)
        df.loc[raw.isna(), "target_class"] = np.nan

    return df.iloc[:-shift].copy()

def create_targets_atr(
    df,
    return_horizon_min,
    sampling_minutes,
    tick_size=None,
    ternary=False,
    atr_multiplier=1.5,
    instrument_info=None,
    profit_target_multiplier=1.0,
):
    """
    Crea targets basados en ATR pero expresados en ticks cuando se pasa `tick_size`.

    - Si `instrument_info` está presente puede usar campos como
      `tick_value`, `approx_total_fee_per_side`, `min_slippage_ticks` y
      `min_spread_ticks` para estimar costes reales en ticks.
    - `profit_target_multiplier` permite escalar el umbral objetivo (p.ej. 1.1 para +10%).
    """

    df = df.copy()
    shift = max(1, int(return_horizon_min / max(1, int(sampling_minutes))))
    suffix = f"{return_horizon_min}m"

    future_price = df["close"].shift(-shift)
    df[f"target_logret_{suffix}"] = np.log(future_price / df["close"])
    df[f"target_ret_{suffix}"] = future_price / df["close"] - 1
    price_diff = future_price - df["close"]

    # Umbral dinámico basado en ATR (en unidades de precio)
    dynamic_threshold = df["atr"] * atr_multiplier

    # Si no se pasa tick_size mantenemos la lógica anterior en precio
    if tick_size is None:
        # Binario basado en ATR en precio
        df["target_bin"] = (price_diff > dynamic_threshold).astype("Int8")

        if ternary:
            df["target_class"] = np.select(
                condlist=[price_diff > dynamic_threshold, price_diff < -dynamic_threshold],
                choicelist=[2, 0],
                default=1,
            ).astype(float)
            df.loc[future_price.isna(), "target_class"] = np.nan

        df.loc[future_price.isna(), "target_bin"] = np.nan
        return df.iloc[:-shift].copy()

    # --- Cuando hay tick_size: convertir todo a ticks y descontar costes ---
    raw_ticks = price_diff / float(tick_size)

    # Instrument info defaults
    tick_value = None
    fee_per_side = 0.0
    min_slippage_ticks = 0.0
    spread_ticks = 0.0

    if instrument_info:
        tick_value = instrument_info.get("tick_value")
        fee_per_side = float(instrument_info.get("approx_total_fee_per_side", 0.0))
        min_slippage_ticks = float(instrument_info.get("min_slippage_ticks", 0.0))
        # some specs may provide spread estimate
        spread_ticks = float(
            instrument_info.get(
                "min_spread_ticks", instrument_info.get("spread_ticks", min_slippage_ticks or 0.0)
            )
        )

    # Si no se conoce tick_value, asumimos 1.0 (solo afecta conversión fee->ticks)
    if tick_value is None or tick_value == 0:
        tick_value = 1.0

    # Costes en ticks (round-trip fees + spread + slippage)
    fees_roundtrip_ticks = (fee_per_side * 2.0) / float(tick_value) if fee_per_side else 0.0
    slippage_ticks = min_slippage_ticks
    total_cost_ticks = fees_roundtrip_ticks + spread_ticks + slippage_ticks

    # Umbral dinámico en ticks
    dynamic_threshold_ticks = dynamic_threshold / float(tick_size)

    # Umbral efectivo: exigimos superar ATR en ticks + costes, y opcionalmente escalado por
    # `profit_target_multiplier` (p.ej. 1.1 para requerir 10% más de beneficio)
    effective_threshold_ticks = dynamic_threshold_ticks * float(profit_target_multiplier) + total_cost_ticks

    # Guardar columnas explicativas
    df[f"target_ticks_{suffix}"] = raw_ticks
    df[f"target_ticks_net_{suffix}"] = raw_ticks - np.sign(raw_ticks) * total_cost_ticks
    df["fees_roundtrip_ticks"] = fees_roundtrip_ticks
    df["spread_ticks"] = spread_ticks
    df["slippage_ticks"] = slippage_ticks
    df["total_cost_ticks"] = total_cost_ticks
    df["dynamic_threshold_ticks"] = dynamic_threshold_ticks
    df["effective_threshold_ticks"] = effective_threshold_ticks

    # Clasificación usando ticks netos vs umbral efectivo
    df["target_bin"] = (raw_ticks > effective_threshold_ticks).astype("Int8")

    if ternary:
        df["target_class"] = np.select(
            condlist=[raw_ticks > effective_threshold_ticks, raw_ticks < -effective_threshold_ticks],
            choicelist=[2, 0],
            default=1,
        ).astype(float)

        df.loc[future_price.isna(), "target_class"] = np.nan

    df.loc[future_price.isna(), "target_bin"] = np.nan

    return df.iloc[:-shift].copy()

def apply_liquidity_filter(df, target_cols=['target_class', 'target_bin']):
    df = df.copy()
    
    # Condición de liquidez
    is_liquid = (df['is_ny'] == 1) | (df['is_london'] == 1)
    
    for col in target_cols:
        if col in df.columns:
            if col == 'target_class':
                # En ternario, 1.0 es Neutral
                df.loc[~is_liquid, col] = 1.0
            else:
                # En binario, 0 es Neutral/No operamos
                df.loc[~is_liquid, col] = 0
    
    return df

def audit_single_df(df, feature_cols):
    """
    Audita y limpia un único DataFrame:
    - Reemplaza infinitos por NaN
    - Dropea filas con NaNs en las columnas de features
    - Devuelve el DataFrame limpio
    """
    df = df.copy()

    # 1. Detectar infinitos
    inf_count = np.isinf(df[feature_cols]).sum().sum()
    if inf_count > 0:
        print(f"⚠️ Encontrados {inf_count} infinitos → reemplazando por NaN")
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 2. Detectar NaNs
    nan_count = df[feature_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"⚠️ Encontrados {nan_count} NaNs → dropna en features")
        df = df.dropna(subset=feature_cols)
        print(f"   → Filas restantes: {len(df)}")
    else:
        print("💎 Sin NaNs en features")

    return df

def evaluate_atr_multipliers(
    df,
    return_horizon_min,
    sampling_minutes,
    tick_size,
    atr_min=0.5,
    atr_max=1.5,
    atr_step=0.25,
):
    """
    Evalúa diferentes multiplicadores ATR y devuelve un DataFrame con los conteos
    de target_class antes y después del filtro de liquidez.
    """

    results = []

    multipliers = np.arange(atr_min, atr_max + atr_step, atr_step)

    for m in multipliers:
        # 1. Crear targets con ATR
        df_atr = create_targets_atr(
            df.copy(),
            return_horizon_min=return_horizon_min,
            sampling_minutes=sampling_minutes,
            tick_size=tick_size,
            ternary=True,
            atr_multiplier=m
        )

        # 2. Aplicar filtro de liquidez
        df_liquid = apply_liquidity_filter(df_atr)

        # 3. Conteos antes y después
        counts_before = df_atr["target_class"].value_counts(dropna=False)
        counts_after = df_liquid["target_class"].value_counts(dropna=False)

        # 4. Normalizados
        norm_before = df_atr["target_class"].value_counts(normalize=True, dropna=False)
        norm_after = df_liquid["target_class"].value_counts(normalize=True, dropna=False)

        # 5. Consolidar resultados
        for cls in sorted(set(df_atr["target_class"].unique())):
            results.append({
                "atr_multiplier": m,
                "target_class": cls,
                "count_before": counts_before.get(cls, 0),
                "count_after": counts_after.get(cls, 0),
                "norm_before": norm_before.get(cls, 0.0),
                "norm_after": norm_after.get(cls, 0.0),
            })

    return pd.DataFrame(results)

def target_realistic(df, spec, return_horizon_min, sampling_minutes):
    df = df.copy()
    tick_size = float(spec["tick_size"])
    tick_value = float(spec["tick_value"])
    
    # 1. Horizonte
    shift = max(1, int(return_horizon_min / max(1, int(sampling_minutes))))
    
    # 2. Diferencia de precio en ticks
    future_price = df["close"].shift(-shift)
    raw_ticks = (future_price - df["close"]) / tick_size

    # 3. Costes Reales (Cálculo basado en tu JSON)
    fee_side = float(spec.get("approx_total_fee_per_side", 2.5))
    fees_rt_ticks = (fee_side * 2.0) / tick_value # 0.5 ticks
    slip_ticks = float(spec.get("min_slippage_ticks", 1.0)) # 1.0 tick
    spread_ticks = float(spec.get("min_spread_ticks", 1.0)) # 1.0 tick
    
    total_cost_ticks = fees_rt_ticks + slip_ticks + spread_ticks # Total: 2.5 ticks
    
    # 4. AJUSTE CRÍTICO: El Multiplicador de "Ruido"
    # Para el Oro en 120m, un movimiento de 2.5 ticks es insignificante.
    # Queremos que el objetivo sea al menos 3 o 4 veces el coste para que valga la pena.
    profit_factor = 4.0 
    threshold = total_cost_ticks * profit_factor # 2.5 * 4 = 10 ticks ($1.0 en precio)

    # 5. Clasificación Ternaria
    conditions = [
        (raw_ticks > threshold),  # BUY
        (raw_ticks < -threshold)  # SELL
    ]
    # Usamos 0 para HOLD, 1 para BUY, 2 para SELL (común para Softmax)
    # O mantén [-1, 0, 1] si prefieres, yo usaré [1, 0, -1] aquí:
    df[f"target_class_{return_horizon_min}m"] = np.select(conditions, [1, -1], default=0)
    # Mapear a 0, 1, 2 para Softmax (opcional):
    df[f"target_class_{return_horizon_min}m"] = df[f"target_class_{return_horizon_min}m"].map({1: 1, 0: 0, -1: 2})
    
    return df.dropna()

def target_triple_barrier_realistic(
    df,
    spec,
    return_horizon_min=120,
    sampling_minutes=60,
    atr_col="atr",
    tp_atr_mult=1.8,
    sl_atr_mult=1.2,
    profit_factor=2.5
):
    df = df.copy()

    tick_size = float(spec["tick_size"])
    tick_value = float(spec["tick_value"])

    max_steps = max(1, int(return_horizon_min / sampling_minutes))

    fee_side = float(spec.get("approx_total_fee_per_side", 2.5))
    fees_rt_ticks = (fee_side * 2.0) / tick_value
    slip_ticks = float(spec.get("min_slippage_ticks", 1.0))
    spread_ticks = float(spec.get("min_spread_ticks", 1.0))

    total_cost_ticks = fees_rt_ticks + slip_ticks + spread_ticks

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df[atr_col].values

    # CORRECCIÓN 1: Usamos np.nan en lugar de ceros.
    # Así, las últimas velas que no pueden mirar al futuro se eliminarán 
    # correctamente con el dropna() final, evitando datos falsos.
    labels = np.full(len(df), np.nan)

    for i in range(len(df) - max_steps):

        entry_price = close[i]
        atr_ticks = atr[i] / tick_size

        tp_ticks = tp_atr_mult * atr_ticks
        sl_ticks = sl_atr_mult * atr_ticks

        min_profit_ticks = total_cost_ticks * profit_factor
        tp_ticks = max(tp_ticks, min_profit_ticks)

        upper_barrier = entry_price + tp_ticks * tick_size
        lower_barrier = entry_price - sl_ticks * tick_size

        label = 0  # HOLD por defecto

        for j in range(1, max_steps + 1):
            
            # Evaluamos ambas condiciones en la misma vela
            hit_tp = high[i + j] >= upper_barrier
            hit_sl = low[i + j] <= lower_barrier

            # CORRECCIÓN 2: Lógica de conflicto intrabarra
            if hit_tp and hit_sl:
                # Si toca ambos en la misma vela, asumimos el peor escenario (SL)
                label = -1
                break
            elif hit_tp:
                label = 1
                break
            elif hit_sl:
                label = -1
                break

        labels[i] = label

    df[f"target_tb_{return_horizon_min}m"] = pd.Series(labels, index=df.index).map({
        -1: 0,  # SELL (o Stop Loss)
        0: 1,   # HOLD (Expiración de tiempo)
        1: 2    # BUY (Take Profit)
    })

    # Ahora sí, dropna() eliminará las últimas 'max_steps' filas llenas de NaN
    return df.dropna()