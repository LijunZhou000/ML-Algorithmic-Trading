import numpy as np
def create_targets(df, return_horizon_min, sampling_minutes=1, tick_size=None, ternary=False, pos_threshold=0.0):
    """Create target columns for a given horizon.

    Behavior:
    - If `tick_size` is provided the target is computed in ticks and
        classification thresholds (`pos_threshold`) are interpreted in ticks.
    - If no `tick_size` is provided the function falls back to log-returns/pct.

    Parameters
    - `return_horizon_min`: horizon in minutes (e.g., 30, 60)
    - `sampling_minutes`: minutes per sample (1 for 1-min, 30 for 30-min)
    - `tick_size`: tick size (e.g. 0.1). If provided `target_ticks_*m` is created
    - `ternary`: if True create `target_class` with 0=SELL,1=NEUTRAL,2=BUY
    - `pos_threshold`: threshold for deciding neutral vs buy/sell. Interpreted as
        ticks when `tick_size` is provided, otherwise as log-return.
    """
    df = df.copy()
    # compute shift in rows
    shift = max(1, int(return_horizon_min / max(1, int(sampling_minutes))))

    # log return and pct return for horizon
    col_log = f'target_logret_{return_horizon_min}m'
    col_pct = f'target_ret_{return_horizon_min}m'
    df[col_log] = np.log(df['close'].shift(-shift) / df['close'])
    df[col_pct] = df['close'].shift(-shift) / df['close'] - 1

    # ticks if tick_size provided
    if tick_size is not None:
        col_ticks = f'target_ticks_{return_horizon_min}m'
        df[col_ticks] = (df['close'].shift(-shift) - df['close']) / float(tick_size)

    # Prefer ticks when tick_size is provided
    if tick_size is not None:
        # create ticks column already above; classify by ticks
        # Interpret `pos_threshold` flexibly: if user passed a small value (<1)
        # treat it as a relative price return and convert to ticks per-row.
        if pos_threshold is None:
            pos_threshold = 0.0

        if abs(pos_threshold) < 1.0 and pos_threshold != 0.0:
            # per-row threshold in ticks = (return_threshold * price) / tick_size
            thresh_ticks = (pos_threshold * df['close']).abs() / float(tick_size)
        else:
            # pos_threshold already expressed in ticks (or is 0)
            thresh_ticks = float(abs(pos_threshold))

        # Binary target: require movement in ticks beyond threshold (if threshold>0)
        if (hasattr(thresh_ticks, 'shape')):
            df['target_bin'] = (df[col_ticks].abs() > thresh_ticks).astype(int) * (df[col_ticks] > 0).astype(int)
            # This produces 1 for positive moves above threshold, 0 otherwise
        else:
            if thresh_ticks > 0:
                df['target_bin'] = (df[col_ticks] > thresh_ticks).astype(int)
            else:
                df['target_bin'] = (df[col_ticks] > 0).astype(int)

        if ternary:
            # Vectorized ternary classification using per-row thresholds when available
            if hasattr(thresh_ticks, 'shape'):
                df['target_class'] = np.where(df[col_ticks] > thresh_ticks, 2,
                                              np.where(df[col_ticks] < -thresh_ticks, 0, 1))
                # preserve NaN where appropriate
                df.loc[df[col_ticks].isna(), 'target_class'] = np.nan
            else:
                t = float(thresh_ticks)
                def _clas_ticks(x):
                    if np.isnan(x):
                        return np.nan
                    if x > t:
                        return 2
                    if x < -t:
                        return 0
                    return 1

                df['target_class'] = df[col_ticks].apply(_clas_ticks)
    else:
        # fallback to log-return based targets
        df['target_bin'] = (df[col_log] > 0).astype(int)
        if ternary:
            def _clas(val):
                if np.isnan(val):
                    return np.nan
                if val > pos_threshold:
                    return 2
                if val < -pos_threshold:
                    return 0
                return 1

            df['target_class'] = df[col_log].apply(_clas)
    df = df.iloc[:-shift]
    return df