"""
swing_features.py
=================
Features inter-día basadas en swing highs / swing lows.
 
Diseño sin look-ahead
----------------------
Un swing high en la barra i requiere N velas a la derecha para confirmarse.
Por tanto, en la barra i+N es cuando "sabemos" que fue un swing.
Implementamos esto con .shift(-n) para encontrar el máximo local y luego
reasignamos el valor a la barra original mediante forward-fill, de modo que
en cada barra t sólo se conoce el swing más reciente ya confirmado (cuyo
índice real es <= t - n).
 
Dependencias del decorador
---------------------------
  group  = 'swing'
  requires = [] para swing_pivots (sólo usa columnas OHLCV base)
  requires = ['swing_high_price', 'swing_low_price']   para el resto
"""
import talib as ta
import numpy as np
import pandas as pd
import json
import inspect
from tradepy.features.registry import feature


# ──────────────────────────────────────────────────────────────────────────────
# 1. DETECCIÓN DE PIVOTES  (grupo base, sin dependencias)
# ──────────────────────────────────────────────────────────────────────────────
 
@feature(group='swing', requires=[])
def swing_pivots(df: pd.DataFrame, n: int = 2) -> dict:
    """
    Detecta swing highs y swing lows locales con ventana ±n velas.
 
    Retraso real
    ------------
    El pivote en la barra i se confirma en la barra i+n (cuando se dispone
    de las n velas derechas). Para evitar look-ahead:
      - Identificamos el índice posicional del pivote.
      - Lo "publicamos" en la posición i+n desplazando el array.
      - Forward-fill → en cada barra t el último swing conocido es el último
        publicado hasta t (es decir, ocurrió en t-n o antes).
 
    Columnas producidas
    -------------------
    swing_high          : 1.0 en la barra confirmada del swing high, else NaN
    swing_low           : 1.0 en la barra confirmada del swing low,  else NaN
    swing_high_price    : precio del swing high en la barra confirmada, else NaN
    swing_low_price     : precio del swing low  en la barra confirmada, else NaN
    swing_high_strength : (high_pivot - min(flancos n izq + n der)) normalizado por close
    swing_low_strength  : (max(flancos) - low_pivot) normalizado por close
    last_swing_high_price : precio del swing high más reciente ya confirmado (ffill)
    last_swing_low_price  : precio del swing low  más reciente ya confirmado (ffill)
    last_swing_high_bar   : índice posicional del último swing high confirmado
    last_swing_low_bar    : índice posicional del último swing low  confirmado
    """
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    nrows = len(df)
 
    # ── detección de pivotes locales ──────────────────────────────────────────
    # swing high en i: high[i] == max(high[i-n : i+n+1])
    # swing low  en i: low[i]  == min(low[i-n  : i+n+1])
    swing_h_price = np.full(nrows, np.nan)
    swing_l_price = np.full(nrows, np.nan)
    swing_h_strength = np.full(nrows, np.nan)
    swing_l_strength = np.full(nrows, np.nan)
 
    for i in range(n, nrows - n):
        window_h = high[i - n: i + n + 1]
        window_l = low[i  - n: i + n + 1]
 
        if high[i] == window_h.max():
            swing_h_price[i] = high[i]
            # fuerza: cuánto sobresale sobre el mínimo de los flancos
            flanks_min = min(window_l[0], window_l[-1])
            swing_h_strength[i] = (high[i] - flanks_min) / close[i] if close[i] != 0 else np.nan
 
        if low[i] == window_l.min():
            swing_l_price[i] = low[i]
            flanks_max = max(window_h[0], window_h[-1])
            swing_l_strength[i] = (flanks_max - low[i]) / close[i] if close[i] != 0 else np.nan
 
    # ── retraso real: publicar en barra i+n ───────────────────────────────────
    # shift(n) hacia adelante → la barra i se publica en i+n
    sh_price_delayed  = np.full(nrows, np.nan)
    sl_price_delayed  = np.full(nrows, np.nan)
    sh_strength_del   = np.full(nrows, np.nan)
    sl_strength_del   = np.full(nrows, np.nan)
 
    sh_price_delayed[n:]  = swing_h_price[:-n]
    sl_price_delayed[n:]  = swing_l_price[:-n]
    sh_strength_del[n:]   = swing_h_strength[:-n]
    sl_strength_del[n:]   = swing_l_strength[:-n]
 
    # ── forward-fill: en cada barra t, último swing conocido ──────────────────
    def _ffill(arr: np.ndarray):
        s = pd.Series(arr)
        return s.ffill().values
 
    def _ffill_bar(arr: np.ndarray):
        """Guarda el índice posicional del último swing confirmado."""
        bar_idx = np.where(~np.isnan(arr), np.arange(len(arr)), np.nan)
        return pd.Series(bar_idx).ffill().values
 
    last_sh_price = _ffill(sh_price_delayed)
    last_sl_price = _ffill(sl_price_delayed)
    last_sh_bar   = _ffill_bar(sh_price_delayed)
    last_sl_bar   = _ffill_bar(sl_price_delayed)

    # forward-fill de la "fuerza" del swing para uso continuo
    last_sh_strength = _ffill(sh_strength_del)
    last_sl_strength = _ffill(sl_strength_del)
 
    idx = df.index
    return {
        # señal en la barra confirmada
        'swing_high':           pd.Series(np.where(~np.isnan(sh_price_delayed), 1.0, np.nan), index=idx),
        'swing_low':            pd.Series(np.where(~np.isnan(sl_price_delayed), 1.0, np.nan), index=idx),
        'swing_high_price':     pd.Series(sh_price_delayed,  index=idx),
        'swing_low_price':      pd.Series(sl_price_delayed,  index=idx),
        'swing_high_strength':  pd.Series(sh_strength_del,   index=idx),
        'swing_low_strength':   pd.Series(sl_strength_del,   index=idx),
        # valor continuo (ffill) para features derivadas
        'last_swing_high_price':  pd.Series(last_sh_price,     index=idx),
        'last_swing_low_price':   pd.Series(last_sl_price,     index=idx),
        'last_swing_high_strength': pd.Series(last_sh_strength, index=idx),
        'last_swing_low_strength':  pd.Series(last_sl_strength, index=idx),
        'last_swing_high_bar':    pd.Series(last_sh_bar,       index=idx),
        'last_swing_low_bar':     pd.Series(last_sl_bar,       index=idx),
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 2. DISTANCIAS AL ÚLTIMO SWING
# ──────────────────────────────────────────────────────────────────────────────
 
@feature(group='swing', requires=['last_swing_high_price', 'last_swing_low_price'])
def swing_distances(df: pd.DataFrame) -> dict:
    """
    Distancias del precio actual al último swing high/low confirmado.
 
    Columnas producidas
    -------------------
    dist_to_last_swing_high   : (last_swing_high_price - close) / close
    dist_to_last_swing_low    : (close - last_swing_low_price)  / close
    swing_range               : last_swing_high_price - last_swing_low_price
    swing_range_pct           : swing_range / close
    bars_since_swing_high     : barras transcurridas desde el último swing high confirmado
    bars_since_swing_low      : barras desde el último swing low  confirmado
    close_in_swing_range      : posición de close dentro del rango [0..1] (0=en el low, 1=en el high)
    """
    close = df['close']
    sh    = df['last_swing_high_price']
    sl    = df['last_swing_low_price']
 
    bar_pos = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
 
    swing_range = sh - sl
    swing_range_safe = swing_range.replace(0, np.nan)
 
    return {
        'dist_to_last_swing_high': (sh - close) / close.replace(0, np.nan),
        'dist_to_last_swing_low':  (close - sl)  / close.replace(0, np.nan),
        'swing_range':             swing_range,
        'swing_range_pct':         swing_range / close.replace(0, np.nan),
        'bars_since_swing_high':   bar_pos - df['last_swing_high_bar'],
        'bars_since_swing_low':    bar_pos - df['last_swing_low_bar'],
        'close_in_swing_range':    (close - sl) / swing_range_safe,
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 3. ESTRUCTURA DE MERCADO  (HH / HL / LH / LL)
# ──────────────────────────────────────────────────────────────────────────────
 
@feature(group='swing', requires=['swing_high_price', 'swing_low_price'])
def swing_structure(df: pd.DataFrame, trend_window: int = 4) -> dict:
    """
    Clasifica cada swing respecto al anterior del mismo tipo y deriva
    métricas de tendencia estructural.
 
    Columnas producidas
    -------------------
    swing_high_type   : +1 = Higher High, -1 = Lower High,  0 = igual / sin datos
    swing_low_type    : +1 = Higher Low,  -1 = Lower Low,   0 = igual / sin datos
    market_structure  : +1 = uptrend (HH+HL), -1 = downtrend (LH+LL), 0 = mixto
    swing_trend_score : suma rodante de los últimos `trend_window` swing types (+/-1)
                        como proxy de momentum estructural
    """
    nrows   = len(df)
    sh_arr  = df['swing_high_price'].values   # NaN salvo en confirmación
    sl_arr  = df['swing_low_price'].values
 
    sh_type = np.zeros(nrows, dtype=float)
    sl_type = np.zeros(nrows, dtype=float)
 
    prev_sh = np.nan
    prev_sl = np.nan
 
    for i in range(nrows):
        if not np.isnan(sh_arr[i]):
            if not np.isnan(prev_sh):
                sh_type[i] = 1.0 if sh_arr[i] > prev_sh else (-1.0 if sh_arr[i] < prev_sh else 0.0)
            prev_sh = sh_arr[i]
 
        if not np.isnan(sl_arr[i]):
            if not np.isnan(prev_sl):
                sl_type[i] = 1.0 if sl_arr[i] > prev_sl else (-1.0 if sl_arr[i] < prev_sl else 0.0)
            prev_sl = sl_arr[i]
 
    # market_structure: +1 si el swing high más reciente es HH y el low más reciente es HL
    # Para ello, forward-fill los tipos
    sh_type_s = pd.Series(np.where(sh_type != 0, sh_type, np.nan), index=df.index).ffill().fillna(0)
    sl_type_s = pd.Series(np.where(sl_type != 0, sl_type, np.nan), index=df.index).ffill().fillna(0)
 
    market_structure = np.where(
        (sh_type_s == 1) & (sl_type_s == 1),  1.0,   # uptrend
        np.where(
            (sh_type_s == -1) & (sl_type_s == -1), -1.0,   # downtrend
            0.0                                             # mixto
        )
    )
 
    # trend_score: cuenta los últimos trend_window eventos de swing (HH/HL positivo, LH/LL negativo)
    combined = pd.Series(sh_type + sl_type, index=df.index)
    trend_score = combined.rolling(trend_window, min_periods=1).sum()
 
    idx = df.index
    # devolver tipos forward-filled (útiles como features continuas)
    return {
        'swing_high_type':   sh_type_s,
        'swing_low_type':    sl_type_s,
        'market_structure':  pd.Series(market_structure,  index=idx),
        'swing_trend_score': trend_score,
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 4. RETROCESOS DE FIBONACCI NATURALES
# ──────────────────────────────────────────────────────────────────────────────
 
@feature(group='swing', requires=['last_swing_high_price', 'last_swing_low_price', 'swing_range'])
def swing_retracement(df: pd.DataFrame) -> dict:
    """
    Calcula el retroceso actual como fracción del último swing range y
    la distancia a los niveles de Fibonacci clásicos (0.236, 0.382, 0.5, 0.618, 0.786).
 
    Columnas producidas
    -------------------
    retracement_ratio    : (close - last_swing_low) / swing_range  ∈ [0, 1]
                           0 = en el swing low, 1 = en el swing high
    fib_dist_236 … 786   : (retracement_ratio - fib_level), positivo = por encima del nivel
    nearest_fib_level    : nivel Fibonacci más cercano al retroceso actual
    dist_to_nearest_fib  : distancia absoluta al nivel más cercano
    swing_extension      : cuánto supera el precio al último swing high (en % de swing_range)
                           positivo = extensión alcista
    """
    close  = df['close']
    sh     = df['last_swing_high_price']
    sl     = df['last_swing_low_price']
    rng    = df['swing_range'].replace(0, np.nan)
 
    retracement = (close - sl) / rng   # 0 = low, 1 = high
 
    fibs = [0.236, 0.382, 0.500, 0.618, 0.786]
 
    out = {'retracement_ratio': retracement}
 
    fib_distances = {}
    for f in fibs:
        key = f'fib_dist_{str(f).replace("0.", "").replace(".", "")}'
        fib_distances[key] = retracement - f
        out[key] = retracement - f
 
    # nivel más cercano y distancia
    fib_arr = np.array(fibs)
    ret_vals = retracement.values
    nearest = np.full(len(ret_vals), np.nan)
    dist_nearest = np.full(len(ret_vals), np.nan)
    for i, r in enumerate(ret_vals):
        if not np.isnan(r):
            diffs = np.abs(fib_arr - r)
            idx_min = np.argmin(diffs)
            nearest[i] = fib_arr[idx_min]
            dist_nearest[i] = r - fib_arr[idx_min]
 
    out['nearest_fib_level']    = pd.Series(nearest,      index=df.index)
    out['dist_to_nearest_fib']  = pd.Series(dist_nearest, index=df.index)
 
    # extensión: precio por encima del último swing high (normalizado por el rango)
    out['swing_extension'] = (close - sh) / rng   # positivo si superó el swing high
 
    return out
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 5. VELOCIDAD Y TIEMPO ENTRE SWINGS
# ──────────────────────────────────────────────────────────────────────────────
 
@feature(group='swing', requires=['swing_high_price', 'swing_low_price',
                                   'last_swing_high_bar', 'last_swing_low_bar',
                                   'swing_range'])
def swing_velocity(df: pd.DataFrame) -> dict:
    """
    Mide la cadencia temporal de los swings y la velocidad del precio
    entre ellos.
 
    Columnas producidas
    -------------------
    bars_between_swing_highs  : barras entre el penúltimo y el último swing high
    bars_between_swing_lows   : barras entre el penúltimo y el último swing low
    swing_high_velocity       : swing_range / bars_between_swing_highs  (precio/barra)
    swing_low_velocity        : swing_range / bars_between_swing_lows
    bars_between_swings       : barras entre el último swing high y el último swing low
                                (independientemente del orden)
    swing_cycle_ratio         : bars_since_swing_high / bars_between_swings
                                → posición dentro del ciclo [0..1]
    """
    sh_price = df['swing_high_price'].values
    sl_price = df['swing_low_price'].values
    nrows    = len(df)
    idx      = df.index
 
    # ── barras entre swings consecutivos del mismo tipo ───────────────────────
    btwn_sh = np.full(nrows, np.nan)
    btwn_sl = np.full(nrows, np.nan)
    sh_vel  = np.full(nrows, np.nan)
    sl_vel  = np.full(nrows, np.nan)
 
    prev_sh_bar = np.nan
    prev_sl_bar = np.nan
 
    for i in range(nrows):
        if not np.isnan(sh_price[i]):
            if not np.isnan(prev_sh_bar):
                gap = i - prev_sh_bar
                btwn_sh[i] = gap
                rng_val = df['swing_range'].iloc[i]
                sh_vel[i] = rng_val / gap if gap > 0 else np.nan
            prev_sh_bar = i
 
        if not np.isnan(sl_price[i]):
            if not np.isnan(prev_sl_bar):
                gap = i - prev_sl_bar
                btwn_sl[i] = gap
                rng_val = df['swing_range'].iloc[i]
                sl_vel[i] = rng_val / gap if gap > 0 else np.nan
            prev_sl_bar = i
 
    # forward-fill para que cada barra lleve el valor del último ciclo conocido
    btwn_sh_ff = pd.Series(btwn_sh, index=idx).ffill()
    btwn_sl_ff = pd.Series(btwn_sl, index=idx).ffill()
    sh_vel_ff  = pd.Series(sh_vel,  index=idx).ffill()
    sl_vel_ff  = pd.Series(sl_vel,  index=idx).ffill()
 
    # ── barras entre último SH y último SL (sin importar orden) ──────────────
    last_sh_bar = df['last_swing_high_bar']
    last_sl_bar = df['last_swing_low_bar']
    bars_between = (last_sh_bar - last_sl_bar).abs()
 
    bar_pos = pd.Series(np.arange(nrows, dtype=float), index=idx)
    bars_since_sh = bar_pos - last_sh_bar
    cycle_ratio = bars_since_sh / bars_between.replace(0, np.nan)

    # ffill para disminuir NaNs donde tenga sentido (mantener comportamiento causal)
    bars_between_ff = bars_between.ffill()
    cycle_ratio_ff = cycle_ratio.ffill()
 
    return {
        'bars_between_swing_highs': btwn_sh_ff,
        'bars_between_swing_lows':  btwn_sl_ff,
        'swing_high_velocity':      sh_vel_ff,
        'swing_low_velocity':       sl_vel_ff,
        'bars_between_swings':      bars_between_ff,
        'swing_cycle_ratio':        cycle_ratio_ff,
    }