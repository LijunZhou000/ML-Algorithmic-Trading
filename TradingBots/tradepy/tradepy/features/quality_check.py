import talib as ta
import numpy as np
import pandas as pd
import json
import inspect

def data_quality_report(
    df,
    future_horizon=1,
    verbose=True,
    top_n=20,
    clean_infs=False,
    compute_corr=False,
    corr_method='pearson',
    corr_max_cols=100,
    cols=None,
    by_day=False,
):
    """Run a set of data-quality checks on `df` and return a report dict.

    New options:
    - `clean_infs`: replace inf/-inf with NaN and report counts
    - `compute_corr`: compute correlation with future returns (limited by `corr_max_cols`)
    - `cols`: optional list of columns to focus the analysis on
    - `by_day`: if True and `trading_date` present, compute NaN fraction by day
    Returns a dict `report` and includes a `column_profile` DataFrame under that key.
    """

    df = df.copy()
    report = {}
    n_rows, n_cols = df.shape
    report['rows'] = int(n_rows)
    report['cols'] = int(n_cols)

    # Work on a subset if requested to avoid heavy work
    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        df_sub = df[cols].copy()
    else:
        df_sub = df.copy()

    # NaN fractions
    na_frac = df_sub.isna().mean().sort_values(ascending=False)
    report['na_fraction'] = na_frac.to_dict()
    report['top_na'] = na_frac.head(top_n).to_dict()

    # Constant / zero-variance columns
    nunique = df_sub.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    report['constant_columns'] = constant_cols

    # Infinite values (detect TRUE infinities only). Optionally replace them with NaN.
    numeric_cols = df_sub.select_dtypes(include=[np.number]).columns.tolist()
    inf_cols = []
    inf_counts = {}
    for c in numeric_cols:
        try:
            arr = df_sub[c].values
            inf_mask = np.isinf(arr)
            if inf_mask.any():
                inf_cols.append(c)
                inf_counts[c] = int(inf_mask.sum())
                if clean_infs:
                    df_sub.loc[inf_mask, c] = np.nan
                    # also reflect on main df
                    df.loc[inf_mask, c] = np.nan
        except Exception:
            continue
    report['infinite_columns'] = inf_cols
    report['infinite_counts'] = inf_counts

    # Fraction of zeros
    if numeric_cols:
        zero_frac = (df_sub[numeric_cols] == 0).mean().sort_values(ascending=False)
        report['top_zero_fraction'] = zero_frac.head(top_n).to_dict()
    else:
        report['top_zero_fraction'] = {}

    # Object/category uniques
    obj_uniques = df_sub.select_dtypes(include=['object', 'category']).apply(lambda s: int(s.nunique(dropna=False))).to_dict()
    report['object_unique_counts'] = obj_uniques

    # Basic describe summary (reduced)
    try:
        desc = df_sub.describe().T[['min', '25%', '50%', '75%', 'max']]
        report['describe'] = desc.to_dict(orient='index')
    except Exception:
        report['describe'] = {}

    # Correlation with future returns (if close exists)
    top_corr = {}
    if compute_corr and 'close' in df_sub.columns and numeric_cols:
        # limit numeric columns considered to avoid heavy computation
        cand_cols = [c for c in numeric_cols if c != 'close']
        if len(cand_cols) > corr_max_cols:
            variances = df_sub[cand_cols].var().sort_values(ascending=False)
            cand_cols = variances.head(corr_max_cols).index.tolist()

        try:
            future = df_sub['close'].pct_change(-int(future_horizon))
            corrs = {}
            for c in cand_cols:
                try:
                    valid = (~df_sub[c].isna()) & (~future.isna())
                    if valid.sum() < 3:
                        continue
                    if corr_method == 'pearson':
                        val = float(np.corrcoef(df_sub.loc[valid, c], future.loc[valid])[0,1])
                    else:
                        val = df_sub.loc[valid, c].corr(future.loc[valid], method=corr_method)
                    corrs[c] = float(val) if not pd.isna(val) else None
                except Exception:
                    continue
            if corrs:
                top_corr = dict(sorted(corrs.items(), key=lambda kv: abs(kv[1]) if kv[1] is not None else 0, reverse=True)[:top_n])
        except Exception:
            top_corr = {}
    report['top_corr_with_future'] = top_corr

    # Daily-like columns NaN fraction
    daily_cols = [c for c in df_sub.columns if '_daily' in c]
    daily_na = {c: float(df_sub[c].isna().mean()) for c in daily_cols}
    report['daily_columns_na_fraction'] = daily_na

    # Optional: NaN fraction by trading day (if by_day requested)
    if by_day and 'trading_date' in df.columns:
        try:
            per_day = df.groupby('trading_date').apply(lambda d: d.isna().mean()).T
            report['nan_fraction_by_day'] = per_day.mean(axis=0).to_dict()
        except Exception:
            report['nan_fraction_by_day'] = {}

    # Suggestions
    suggestions = []
    if len(na_frac) and na_frac.max() > 0.5:
        suggestions.append('Many columns have >50% NaNs; consider dropping or re-engineering them.')
    if constant_cols:
        suggestions.append(f'Drop or fix constant columns: {constant_cols[:10]}')
    if inf_cols:
        suggestions.append(f'Columns contain true infinities (use clean_infs=True to replace): {inf_cols[:20]}')
    if any(c.startswith('slope_') for c in df_sub.columns):
        suggestions.append('`rolling_slope` may be slow; consider vectorized replacement or numba.')
    suggestions.append('To avoid fragmentation, build new columns in a dict and `pd.concat` once per group.')
    report['suggestions'] = suggestions

    # Build a compact column-level DataFrame for inspection
    try:
        col_stats = pd.DataFrame({
            'na_frac': na_frac,
            'inf_count': pd.Series(inf_counts),
        }).fillna(0)
        if numeric_cols:
            zf = pd.Series(report['top_zero_fraction'])
            col_stats['zero_frac'] = zf.reindex(col_stats.index).fillna(0)
        nunique = df_sub.nunique(dropna=False)
        col_stats['nunique'] = nunique.reindex(col_stats.index)
        report['column_profile'] = col_stats
    except Exception:
        report['column_profile'] = None

    if verbose:
        print('Data Quality Report')
        print(f' - rows: {n_rows}, cols: {n_cols}')
        print(f" - top {top_n} NaN fractions:\n{pd.Series(report['top_na'])}")
        if report['infinite_columns']:
            print(' - infinite columns:', report['infinite_counts'])
        if report['top_corr_with_future']:
            print(' - top corr with future:', report['top_corr_with_future'])
        print('\nSuggestions:')
        for s in suggestions:
            print(' - ' + s)

    return report