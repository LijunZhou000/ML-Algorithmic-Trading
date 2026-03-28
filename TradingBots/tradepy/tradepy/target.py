import numpy as np

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

def create_targets_atr(df, return_horizon_min, sampling_minutes, tick_size=None, ternary=False, atr_multiplier=1.5):
    df = df.copy()
    shift = max(1, int(return_horizon_min / max(1, int(sampling_minutes))))
    suffix = f"{return_horizon_min}m"

    future_price = df["close"].shift(-shift)
    df[f"target_logret_{suffix}"] = np.log(future_price / df["close"])
    price_diff = future_price - df["close"]

    # Umbral dinámico basado en ATR
    dynamic_threshold = df["atr"] * atr_multiplier

    # --- Lógica Binaria (Siempre se calcula) ---
    # 1 si sube más que el ATR, 0 si no (incluye caídas y neutralidad)
    df["target_bin"] = (price_diff > dynamic_threshold).astype("Int8")

    # --- Lógica Ternaria ---
    if ternary:
        df["target_class"] = np.select(
            condlist=[
                price_diff > dynamic_threshold, 
                price_diff < -dynamic_threshold
            ],
            choicelist=[2, 0],
            default=1
        ).astype(float)
        
        # Limpiar NaNs finales
        df.loc[future_price.isna(), "target_class"] = np.nan
    
    # Limpiar NaNs finales para binario también
    df.loc[future_price.isna(), "target_bin"] = np.nan

    if tick_size:
        df[f"target_ticks_{suffix}"] = price_diff / float(tick_size)

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