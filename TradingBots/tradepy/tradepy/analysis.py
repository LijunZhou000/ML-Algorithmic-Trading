import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def kama(df, window=10, fast=2, slow=30):
    close = df['close'].values.astype(float)

    # Efficiency Ratio (ER) - cálculo seguro evitando divisiones por cero/NaN
    change = np.abs(close - np.roll(close, window))
    volatility = np.abs(np.diff(close))
    volatility = np.concatenate(([np.nan], volatility)).astype(float)
    volatility = pd.Series(volatility).rolling(window).sum().values

    # Evitar evaluación del branch 'change / volatility' cuando volatility==0 o NaN
    er = np.zeros_like(volatility, dtype=float)
    mask = (volatility != 0) & (~np.isnan(volatility))
    er[mask] = change[mask] / volatility[mask]
    er[~mask] = 0.0

    # Smoothing Constant (SC)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)

    # KAMA calculation (recursive) - asegurar dtype float y manejo de valores iniciales
    kama = np.full(close.shape, np.nan, dtype=float)
    if len(close) > 0:
        kama[0] = close[0]
    for i in range(1, len(close)):
        # sc[i] ya está saneado (no NaN/inf)
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    df['kama'] = kama
    return df

def plot_master_market_analysis(df, asset_name="Future", tick_size=0.1,
                                price_plots=True, tick_plots=True, max_autocorr=10000,
                                first_plot_candles=60*24*365, # int (num velas en la base) o 'all'
                                show=True, save=False):
    df_diag = df.copy()
    df_diag.columns = df_diag.columns.str.strip().str.lower()

    # 1. Preparación de Datetime
    if 'datetime' in df_diag.columns:
        df_diag['datetime'] = pd.to_datetime(df_diag['datetime'])
        df_diag = df_diag.set_index('datetime')
    else:
        df_diag.index = pd.to_datetime(df_diag.index)

    # 2. Cálculo de Columnas Auxiliares y Microestructura
    df_diag['hour'] = df_diag.index.hour
    df_diag['day_name'] = df_diag.index.day_name()
    df_diag['volatility'] = df_diag['high'] - df_diag['low']
    df_diag['ticks_range'] = df_diag['volatility'] / tick_size

    # --- CÁLCULOS AVANZADOS (VWAP & EFICIENCIA) ---
    df_diag['tp'] = (df_diag['high'] + df_diag['low'] + df_diag['close']) / 3
    df_diag['pv'] = df_diag['tp'] * df_diag['volume']

    date_group = df_diag.groupby(df_diag.index.date)
    df_diag['vwap'] = date_group['pv'].cumsum() / date_group['volume'].cumsum()

    vwap_std = date_group['tp'].transform(lambda x: x.expanding().std())
    df_diag['vwap_zscore'] = (df_diag['close'] - df_diag['vwap']) / (vwap_std + 0.001)

    lookback_eff = 30
    net_chg = df_diag['close'].diff(lookback_eff).abs()
    path_chg = df_diag['close'].diff().abs().rolling(window=lookback_eff).sum()
    df_diag['efficiency_ratio'] = net_chg / (path_chg + 0.001)

    df_diag['vol_per_tick'] = df_diag['volume'] / (df_diag['ticks_range'] + 0.1)
    df_diag['intensity'] = df_diag['vol_per_tick']
    df_diag['speed'] = df_diag['ticks_range']

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    price_rows = 6 if price_plots else 0
    tick_rows = 6 if tick_plots else 0
    total_rows = price_rows + tick_rows

    fig = plt.figure(figsize=(18, 4 * total_rows))
    gs = fig.add_gridspec(total_rows, 2)
    plt.suptitle(f"MASTER DASHBOARD PROFESIONAL: {asset_name} Future", fontsize=22, y=0.99, fontweight='bold')

    row = 0

    df_diag["kama"] = kama(df_diag)['kama']
    
    # A. Relative Volume (RVOL) - Volumen vs su propia media a esa misma hora
    hourly_avg_vol = df_diag.groupby('hour')['volume'].transform('mean')
    df_diag['rvol'] = df_diag['volume'] / (hourly_avg_vol + 0.001)

    # B. Log-Return Variance (Volatilidad Realizada corta)
    df_diag['log_ret'] = np.log(df_diag['close'] / df_diag['close'].shift(1))
    df_diag['realized_vol'] = df_diag['log_ret'].rolling(window=10).std()
    
    # C. Aceleración de Ticks (Cambio en la velocidad)
    df_diag['speed'] = df_diag['ticks_range']
    df_diag['tick_acceleration'] = df_diag['speed'].diff()

    # --- SECCIÓN DE PRECIO ---
    if price_plots:
        # Fila 1: Precio Diario (selección de cuántas velas mostrar)
        ax0 = fig.add_subplot(gs[row, :])
        df_daily = df_diag['close'].resample('D').last().dropna()

        # Interpretación de first_plot_candles
        if isinstance(first_plot_candles, str) and first_plot_candles == 'all':
            df_daily_to_plot = df_daily
        else:
            try:
                n_candles = int(first_plot_candles)
            except Exception:
                n_candles = 60*24*365

            # estimar duración de cada vela
            idx_diff = df_diag.index.to_series().diff().median()
            median_minutes = idx_diff.total_seconds() / 60.0 if not pd.isna(idx_diff) else 1.0
            candles_per_day = max(1.0, (24 * 60) / median_minutes)
            days = int(np.ceil(n_candles / candles_per_day))
            df_daily_to_plot = df_daily.tail(days)

        # --- PLOT PRECIO DIARIO ---
        ax0.plot(df_daily_to_plot.index, df_daily_to_plot.values,
                 color='#1a508b', linewidth=1.5, label='Cierre Diario')

        # --- KAMA SEMANAL (sin ffill, solo puntos reales) ---
        kama_weekly = df_diag['kama'].resample('W').last().dropna()

        # Filtrar KAMA al mismo rango temporal que df_daily_to_plot
        kama_weekly_visible = kama_weekly.loc[
            (kama_weekly.index >= df_daily_to_plot.index.min()) &
            (kama_weekly.index <= df_daily_to_plot.index.max())
        ]

        # Plot: línea entre puntos semanales + marcadores
        ax0.plot(kama_weekly_visible.index, kama_weekly_visible.values,
                 color='orange', linewidth=1.5, marker='o',
                 label='KAMA semanal')

        ax0.set_title("Evolución del Precio (Cierre Diario) y KAMA")
        ax0.legend()
        row += 1


        # Fila 2: Volumen y Frecuencia
        ax1 = fig.add_subplot(gs[row, 0])
        stats_h = df_diag.groupby('hour').agg({'volume': 'sum', 'hour': 'count'}).rename(columns={'hour': 'count'})
        sns.barplot(x=stats_h.index, y=stats_h['count'], color='skyblue', ax=ax1, alpha=0.6)
        ax1_t = ax1.twinx()
        sns.lineplot(x=stats_h.index, y=stats_h['volume'], color='navy', marker='o', ax=ax1_t)
        ax1.set_title("Nº Velas vs Volumen por Hora")

        ax2 = fig.add_subplot(gs[row, 1])
        stats_d = df_diag.groupby('day_name').agg({'volume': 'sum', 'day_name': 'count'}).reindex(days_order).dropna()
        sns.barplot(x=stats_d.index, y=stats_d['day_name'], color='lightgreen', ax=ax2, alpha=0.6)
        ax2_t = ax2.twinx()
        sns.lineplot(x=stats_d.index, y=stats_d['volume'], color='darkgreen', marker='o', ax=ax2_t)
        ax2.set_title("Integridad de Datos por Día")
        row += 1

        # Fila 3: Wyckoff (Esfuerzo/Resultado)
        ax3 = fig.add_subplot(gs[row, :])
        h_stats = df_diag.groupby('hour').agg({'volume': 'mean', 'volatility': 'mean'})
        sns.barplot(x=h_stats.index, y=h_stats['volume'], color='skyblue', alpha=0.5, ax=ax3)
        ax3_t = ax3.twinx()
        sns.lineplot(x=h_stats.index, y=h_stats['volatility'], color='red', marker='o', ax=ax3_t)
        ax3.set_title("Esfuerzo (Volumen) vs Resultado (Volatilidad)")
        row += 1

        # Fila 4: Heatmap
        ax4 = fig.add_subplot(gs[row, :])
        hmap = df_diag.groupby(['day_name', 'hour'])['volatility'].mean().unstack().reindex(days_order).fillna(0)
        sns.heatmap(hmap, cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Ticks'})
        ax4.set_title("Heatmap: Volatilidad Horaria")
        row += 1

        # Fila 5: Autocorr y Gaps
        ax_ac = fig.add_subplot(gs[row, 0])
        returns = np.log(df_diag['close'] / df_diag['close'].shift(1)).dropna()
        pd.plotting.autocorrelation_plot(returns.tail(max_autocorr), ax=ax_ac)
        ax_ac.set_ylim(-0.05, 0.05)
        ax_ac.set_title("Autocorrelación (Memoria)")

        ax_gap = fig.add_subplot(gs[row, 1])
        df_gap = df_diag.resample('D').agg({'open': 'first', 'close': 'last'}).dropna()
        df_gap['gap'] = df_gap['open'] - df_gap['close'].shift(1)
        sns.histplot(df_gap['gap'].dropna(), kde=True, color='purple', ax=ax_gap)
        ax_gap.set_title("Distribución de Gaps")
        row += 1
        
        # Fila 2: RVOL y Realized Vol
        # Construir subset de minuto que cubra el mismo rango que df_daily_to_plot
        start_dt = df_daily_to_plot.index.min()
        end_dt = df_daily_to_plot.index.max() + pd.Timedelta(days=1)
        df_min_to_plot = df_diag.loc[start_dt:end_dt]

        ax_rvol = fig.add_subplot(gs[row, 0])
        sns.lineplot(data=df_min_to_plot, x=df_min_to_plot.index, y='rvol', ax=ax_rvol, color='green')
        ax_rvol.axhline(1, color='red', linestyle='--')
        ax_rvol.set_title("Relative Volume (RVOL > 1 es inusual)")

        ax_rvol2 = fig.add_subplot(gs[row, 1])
        sns.lineplot(data=df_min_to_plot, x=df_min_to_plot.index, y='realized_vol', ax=ax_rvol2, color='darkred')
        ax_rvol2.set_title("Realized Volatility (Log-Return Std)")
        row += 1

    # --- SECCIÓN DE TICKS & MICROESTRUCTURA ---
    if tick_plots:
        # Fila 6: Distribución e Intensidad
        ax_t1 = fig.add_subplot(gs[row, 0])
        sns.histplot(df_diag['ticks_range'].dropna(), bins=50, kde=True, color='orange', ax=ax_t1)
        ax_t1.set_title("Distribución de Ticks por Vela")

        ax_t2 = fig.add_subplot(gs[row, 1])
        h_vol_tick = df_diag.groupby('hour')['vol_per_tick'].mean()
        sns.lineplot(x=h_vol_tick.index, y=h_vol_tick.values, marker='d', color='purple', ax=ax_t2)
        ax_t2.set_title("Contratos por Tick (Institucional)")
        row += 1

        # Fila 7: Heatmap Ticks
        ax_t3 = fig.add_subplot(gs[row, :])
        hmap_t = df_diag.groupby(['day_name', 'hour'])['ticks_range'].mean().unstack().reindex(days_order).fillna(0)
        sns.heatmap(hmap_t, cmap='YlOrRd', ax=ax_t3)
        ax_t3.set_title("Heatmap: Rango Medio de Ticks")
        row += 1

        # Fila 8: Densidad e Integridad
        ax_t4 = fig.add_subplot(gs[row, 0])
        sns.kdeplot(df_diag['intensity'].dropna(), fill=True, color='purple', ax=ax_t4)
        ax_t4.set_title("Densidad de Intensidad (Contratos/Tick)")

        ax_t5 = fig.add_subplot(gs[row, 1])
        sns.ecdfplot(df_diag['ticks_range'].dropna().clip(upper=df_diag['ticks_range'].quantile(0.99)), ax=ax_t5, color='brown')
        ax_t5.set_title("ECDF: Rango de Ticks (99% Perc)")
        row += 1

        # Fila 9: Velocidad y Eficiencia
        ax_speed = fig.add_subplot(gs[row, 0])
        df_diag.groupby('hour')['speed'].mean().plot(kind='bar', ax=ax_speed, color='orange', alpha=0.7)
        ax_speed.set_title("Velocidad Media (Ticks/Min)")

        ax_eff = fig.add_subplot(gs[row, 1])
        df_diag.groupby('hour')['efficiency_ratio'].mean().plot(ax=ax_eff, color='gold', marker='s')
        ax_eff.set_title("Eficiencia del Movimiento (Trend vs Noise)")
        row += 1

        # Fila 10: Análisis de VWAP
        ax_vwap = fig.add_subplot(gs[row, :])
        sns.histplot(df_diag['vwap_zscore'].dropna(), kde=True, ax=ax_vwap, color='teal')
        ax_vwap.set_title("Z-Score del VWAP (Desviación del Precio Justo)")
        row += 1
        
        ax_acc = fig.add_subplot(gs[row, :])
        df_diag.groupby('hour')['tick_acceleration'].mean().plot(kind='bar', ax=ax_acc, color='salmon')
        ax_acc.set_title("Aceleración Media de Ticks por Hora")
        row += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save:
        plt.savefig(f"../Data/{asset_name}_master_analysis.png", dpi=300)
        print(f"📸 Gráficos guardados como: {asset_name}_master_analysis.png")
    if show:
        plt.show()
    else:
        plt.close()

    
    # return df_diag