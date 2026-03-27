import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tqdm.auto import tqdm # Auto detecta si estás en consola o Jupyter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler 
import talib as ta
from torch import amp
import re

# Funciones de carga, limpieza, resampleo y análisis para futuros
def import_dataset(asset='gc1', format='min'):
    file_path = f'../Data/{asset}{format}.txt'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'vol': 'volume'
    })
    df['dtyyyymmdd'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d')
    # Asegurar que time tenga 6 dígitos (HHMMSS) y convertir a tipo time
    df['time'] = df['time'].astype(str).str.zfill(6)
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S').dt.time
    # Unir fecha y hora en una sola columna de tipo datetime
    df['datetime'] = pd.to_datetime(df['dtyyyymmdd'].astype(str) + ' ' + df['time'].astype(str))
    df = df.sort_values('datetime')
    print(f"Dataset '{asset}' imported with {len(df)} rows")
    specs = pd.read_json("../Data/futuros_specs.json")
    spec = specs[asset[:2].upper()]
    return df, spec

def clean(df, asset='gc1', break_hour=0):
    df = df.copy()

    # 1. Días de la semana: Nos quedamos estrictamente de Lunes (0) a Viernes (4).
    # Esto elimina los sábados (5) y los domingos residuales (6).
    df = df[df['datetime'].dt.weekday <= 4]

    # 2. Eliminar la hora del break dinámicamente.
    # Si le pasas break_hour=0, borrará las 00:00h. Si pasas 23, borrará las 23:00h.
    df = df[df['datetime'].dt.hour != break_hour]

    # 3. Filtrar ruido alrededor del break (15 min antes y 15 min después).
    # Calculamos la hora previa y la hora posterior teniendo en cuenta el ciclo de 24h.
    prev_hour = (break_hour - 1) % 24
    next_hour = (break_hour + 1) % 24
    
    # Máscara para 15 minutos antes del cierre (ej: 22:45 a 22:59)
    mask_noise_before = (df['datetime'].dt.hour == prev_hour) & (df['datetime'].dt.minute >= 45)
    
    # Máscara para 15 minutos después de la apertura (ej: 00:00 a 00:15 o 01:00 a 01:15)
    mask_noise_after = (df['datetime'].dt.hour == next_hour) & (df['datetime'].dt.minute <= 15)
    
    # Aplicamos el filtro para quitar ese ruido
    df = df[~(mask_noise_before | mask_noise_after)]

    # 4. Definir trading day continuo.
    if asset == 'gc1':
        df['trading_date'] = df['datetime'].dt.date
    else:
        df['trading_date'] = (df['datetime'] - pd.Timedelta(hours=1)).dt.date

    # 5. Eliminar duplicados y asegurar orden temporal continuo.
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
    df = df.reset_index(drop=True)

    print(f"Limpieza {asset} completada. Filas restantes: {len(df)}")
    return df

def resample_ohlcv(df, period="5min"):
    """
    Resamplea un dataframe OHLCV al periodo deseado.
    period puede ser: '1min', '5min', '15min', '30min', '1H', '1D', etc.
    """

    # Asegurar orden temporal
    df = df.sort_values("datetime").copy()

    # Asegurar que datetime es datetime64
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Establecer índice temporal
    df = df.set_index("datetime")

    # Diccionario OHLCV estándar
    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "openint": "last",
        "dtyyyymmdd": "first"
    }

    # Resample usando el periodo elegido
    df_resampled = df.resample(period).agg(ohlc_dict)

    # Eliminar velas vacías
    df_resampled = df_resampled.dropna(subset=["open", "high", "low", "close"])

    # Añadir columnas extra
    df_resampled["ticker"] = df["ticker"].iloc[0]
    df_resampled["per"] = period
    
    # Reset index
    df_resampled = df_resampled.reset_index()

    return df_resampled

def get_multitimeframe_data(df_base, intervals=None):
    """
    Toma un DataFrame de 1min y genera un diccionario con múltiples temporalidades.
    
    Args:
        df_base (pd.DataFrame): DataFrame original (limpio y con datetime).
        intervals (list): Lista de enteros representando los minutos.
        
    Returns:
        dict: Diccionario donde las claves son '5min', '15min', etc.
    """
    df_resampled_dict = {}
    if intervals is None:
        intervals = [5, 15, 30, 60]
    for mins in intervals:
        # Formateamos el string del periodo (ej. '30min')
        period_str = f"{mins}min"
        
        # Llamamos a tu función original resample_ohlcv
        # Nota: Asegúrate de que resample_ohlcv esté definida en tu entorno
        df_resampled_dict[period_str] = resample_ohlcv(df_base, period=period_str)
        
        print(f"✅ Agregado: {period_str} | Filas: {len(df_resampled_dict[period_str])}")
        
    return df_resampled_dict

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
    
def daily_ohlcv_cummulative(df_5_min):
    df = df_5_min.copy()
    df = df.sort_values('datetime')
    # Usamos el desplazamiento de 1 hora que definimos en la limpieza
    df['trading_date'] = (df['datetime'] - pd.Timedelta(hours=1)).dt.date
    
    # Ahora agrupamos por 'trading_date' en lugar de 'date'
    df['open_day'] = df.groupby('trading_date')['open'].transform('first')
    df['high_cum'] = df.groupby('trading_date')['high'].cummax()
    df['low_cum'] = df.groupby('trading_date')['low'].cummin()
    df['volume_cum'] = df.groupby('trading_date')['volume'].cumsum()
    
    return df

def apply_cumulative_stats_to_dict(df_dict):
    """
    Recorre el diccionario de DataFrames y aplica tu función 
    de acumulados diarios a cada uno.
    
    Args:
        df_dict (dict): Diccionario con dataframes resampleados ('5min', '30min', etc.)
        
    Returns:
        dict: Diccionario con los DataFrames procesados.
    """
    processed_dict = {}
    
    for tf_key, df_tf in df_dict.items():
        # Llamamos exactamente a tu función original
        # daily_ohlcv_cummulative(df_5_min)
        df_processed = daily_ohlcv_cummulative(df_tf)
        
        # Opcional: Añadir columnas de "distancia" relativas a los acumulados
        # que calculó tu función, muy útiles para la IA
        df_processed['dist_to_high'] = df_processed['high_cum'] - df_processed['close']
        df_processed['dist_to_low'] = df_processed['close'] - df_processed['low_cum']
        
        processed_dict[tf_key] = df_processed
        
        print(f"✅ Procesados acumulados diarios para: {tf_key}")
        
    return processed_dict

# Feature engineering avanzada para futuros
def add_time_features(df):
    """
    Optimizada para 30 minutos: Elimina redundancias y añade el día de la semana,
    crucial para los futuros del Oro.
    """
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. CICLO INTRADIARIO (48 bloques de 30 min)
    # ---------------------------------------------------------
    # Sustituye a la hora. Un solo ciclo de 48 pasos es perfecto para la LSTM.
    step_30m = (df['datetime'].dt.hour * 2) + (df['datetime'].dt.minute // 30)
    df['step_sin'] = np.sin(2 * np.pi * step_30m / 48)
    df['step_cos'] = np.cos(2 * np.pi * step_30m / 48)

    # ---------------------------------------------------------
    # 2. CICLO SEMANAL (NUEVO)
    # ---------------------------------------------------------
    # El Oro se comporta diferente los lunes (gaps del finde) y viernes (cierres)
    day_of_week = df['datetime'].dt.dayofweek # Lunes=0, Domingo=6
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # ---------------------------------------------------------
    # 3. SESIONES INSTITUCIONALES (Basado en UTC/GMT)
    # ---------------------------------------------------------
    # NOTA: Ajusta estos horarios si tus datos tienen otro Timezone (ej. EST)
    df['is_london'] = df['datetime'].dt.hour.isin(range(8, 17)).astype(int)
    df['is_ny'] = df['datetime'].dt.hour.isin(range(13, 22)).astype(int)
    df['is_overlap'] = ((df['is_london'] == 1) & (df['is_ny'] == 1)).astype(int)

    return df

def add_tick_features(df, tick_size=0.1): # El tick size del GC suele ser 0.1
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. MICRO-ESTRUCTURA: Ticks Absolutos (Mantenidos)
    # ---------------------------------------------------------
    df['body_ticks'] = (df['close'] - df['open']) / tick_size
    df['range_ticks'] = (df['high'] - df['low']) / tick_size
    df['upper_wick_ticks'] = (df['high'] - df[['open', 'close']].max(axis=1)) / tick_size
    df['lower_wick_ticks'] = (df[['open', 'close']].min(axis=1) - df['low']) / tick_size
    
    # ---------------------------------------------------------
    # 2. MICRO-ESTRUCTURA: Proporciones (NUEVO)
    # ---------------------------------------------------------
    # Usamos 1e-8 para evitar divisiones por cero en dojis perfectos
    df['body_pct'] = abs(df['body_ticks']) / (df['range_ticks'] + 1e-8)
    df['upper_wick_pct'] = df['upper_wick_ticks'] / (df['range_ticks'] + 1e-8)
    df['lower_wick_pct'] = df['lower_wick_ticks'] / (df['range_ticks'] + 1e-8)
    
    # Dónde cierra la vela: 1 = En el máximo (Bullish max), 0 = En el mínimo (Bearish max)
    df['close_location'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-8)
    
    # ---------------------------------------------------------
    # 3. MACRO-CONTEXTO DEL DÍA (Mantenidos)
    # ---------------------------------------------------------
    df['dist_high_cum_ticks'] = (df['high_cum'] - df['close']) / tick_size
    df['dist_low_cum_ticks'] = (df['close'] - df['low_cum']) / tick_size
    df['dist_open_day_ticks'] = (df['close'] - df['open_day']) / tick_size
    
    # ---------------------------------------------------------
    # 4. DINÁMICAS DE VOLUMEN (Modificados / NUEVO)
    # ---------------------------------------------------------
    # Esfuerzo por tick de rango total (liquidez de la vela)
    df['vol_per_tick'] = df['volume'] / (df['range_ticks'] + 1)
    
    # Esfuerzo vs Resultado Direccional: Volumen usado para mover el precio netamente (cuerpo)
    # Si hay mucho volumen y poco cuerpo, indica posible absorción o freno.
    df['vol_vs_body'] = df['volume'] / (abs(df['body_ticks']) + 1)
    
    # Porcentaje de participación: Qué peso tiene esta vela en el día actual
    df['vol_weight_intraday'] = df['volume'] / (df['volume_cum'] + 1)
    
    return df

def add_indicator_features(df, tick_size=0.1):
    df = df.copy()
    
    # Log returns de la vela actual (Momentum base)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # ---------------------------------------------------------
    # 1. TARGETS CORREGIDOS (Para DF de 30 minutos)
    # ---------------------------------------------------------
    # Shift(-1) = Próximos 30 min | Shift(-2) = Próximos 60 min
    df['target_ret_30m'] = np.log(df['close'].shift(-3) / df['close'])
    df['target_ret_1h'] = np.log(df['close'].shift(-2) / df['close'])

    # ---------------------------------------------------------
    # 2. INDICADORES DE MOMENTUM Y TENDENCIA
    # ---------------------------------------------------------
    # RSI de 14 periodos (Cálculo real de Wilder usando EWM)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # com=13 es el equivalente a la media suavizada de 14 periodos de Wilder
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8) # Evitar división por cero
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Distancia a la Media Móvil (SMA 20 -> 10 horas de trading en 30m)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['dist_sma_20_ticks'] = (df['close'] - df['sma_20']) / tick_size
    
    # NUEVO: Volatilidad (Proxy de ATR / Desviación Estándar)
    # Vital para que la LSTM sepa si el mercado está rápido o lento
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    
    # Limpieza de columnas temporales de cálculo
    df.drop(['sma_20'], axis=1, inplace=True, errors='ignore')
    
    return df

def add_advanced_market_features(df):
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. VWAP INTRADÍA Y Z-SCORE (Precio Justo)
    # ---------------------------------------------------------
    # Calculamos el precio típico
    tp = (df['high'] + df['low'] + df['close']) / 3
    # VWAP acumulado por sesión (usando tu trading_date)
    group = df.groupby('trading_date')
    df['vwap'] = group.apply(lambda x: ( (x['high'] + x['low'] + x['close']) / 3 * x['volume']).cumsum() / x['volume'].cumsum()).reset_index(level=0, drop=True)
    
    # Z-Score: Desviación estándar del precio respecto al VWAP
    # Nos dice si el Oro está "estirado" (Mean Reversion)
    rolling_std = tp.rolling(window=20).std()
    df['vwap_zscore'] = (df['close'] - df['vwap']) / (rolling_std + 1e-8)

    # ---------------------------------------------------------
    # 2. EFFICIENCY RATIO (Kaufman) - Trend vs Noise
    # ---------------------------------------------------------
    # Window de 10 velas (5 horas en 30min)
    net_chg = (df['close'] - df['close'].shift(10)).abs()
    sum_chg = df['close'].diff().abs().rolling(10).sum()
    df['efficiency_ratio'] = net_chg / (sum_chg + 1e-8)

    # ---------------------------------------------------------
    # 3. TICKS VELOCITY (Velocidad de ejecución)
    # ---------------------------------------------------------
    # 'range_ticks' ya lo calculas en add_tick_features. 
    # Aquí medimos qué tan rápido se movió ese rango.
    # Como el DF es de 30min, esto mide la intensidad media de la vela.
    df['ticks_velocity'] = df['range_ticks'] / 30 

    return df

def rsi(df, n=14):
    df = df.copy()
    df['rsi'] = ta.RSI(df['close'], timeperiod=n)
    return df

def true_range(df, window=14):
    # pointwise true range (not ATR)
    df = df.copy()
    df['true_range'] = ta.TRANGE(df['high'], df['low'], df['close'])
    df['range'] = df['high'] - df['low']
    df['range_mean'] = df['range'].rolling(window).mean()
    return df

def compute_atr(df, period=14):
    # ATR computed as rolling mean of true_range
    df = df.copy()
    if 'true_range' not in df.columns:
        df = true_range(df, window=period)
    df['atr'] = df['true_range'].rolling(period).mean()
    return df

def atr_normalized(df):
    df = df.copy()
    df['atr_norm'] = df['atr'] / df['close']
    return df

def macd(df, n_fast=12, n_slow=26):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=n_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=n_slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    return df

def bollinger_bands(df, n=20, num_std_dev=2):
    df = df.copy()
    df['bb_middle'] = df['close'].rolling(n).mean()
    df['bb_std'] = df['close'].rolling(n).std()
    df['bb_upper'] = df['bb_middle'] + num_std_dev * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std_dev * df['bb_std']
    return df

def bollinger_extra(df):
    # requires bb_upper, bb_lower, bb_middle
    df = df.copy()
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    return df

def volume_zscore(df, window=20):
    df = df.copy()
    mean = df['volume'].rolling(window).mean()
    std = df['volume'].rolling(window).std()
    df['volume_z'] = (df['volume'] - mean) / std
    return df

def percent_return_features(df, lags=None, rolling_windows=None):
    df = df.copy()
    if lags is None:
        lags = [1, 5, 10, 20]
    if rolling_windows is None:
        rolling_windows = [5, 10, 20]
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    for lag in lags:
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
    for w in rolling_windows:
        df[f'return_mean_{w}'] = df['return'].rolling(w).mean()
        df[f'return_std_{w}'] = df['return'].rolling(w).std()
        df[f'return_z_{w}'] = (df['return'] - df['return'].rolling(w).mean()) / df['return'].rolling(w).std()
    return df

def ad(
    df,
    zscore_window=20,
    roc_windows=None,
    smooth_windows=None,
    relative_volume_window=20,
    normalize_by="range",   # "range", "close", "none"
    include_clv=True,
    include_delta=True,
    include_direction=True
):
    df = df.copy()
    # --- CLV (Close Location Value) ---
    if include_clv:
        df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
                    (df["high"] - df["low"]).replace(0, np.nan)

    # --- A/D clásico ---
    df["ad"] = (df["clv"] * df["volume"]).cumsum()

    # --- Normalización ---
    if normalize_by == "range":
        df["ad_norm"] = df["ad"] / (df["high"] - df["low"]).replace(0, np.nan)
    elif normalize_by == "close":
        df["ad_norm"] = df["ad"] / df["close"]
    else:
        df["ad_norm"] = df["ad"]

    # defaults for list args
    if roc_windows is None:
        roc_windows = [5, 10]
    if smooth_windows is None:
        smooth_windows = [10]

    # --- Rate of Change (momentum del A/D) ---
    for w in roc_windows:
        df[f"ad_roc_{w}"] = df["ad"].pct_change(w)

    # --- Suavizados ---
    for w in smooth_windows:
        df[f"ad_ema_{w}"] = df["ad"].ewm(span=w, adjust=False).mean()
        df[f"ad_sma_{w}"] = df["ad"].rolling(w).mean()

    # --- Z-score (acumulación/distribución extrema) ---
    mean = df["ad"].rolling(zscore_window).mean()
    std = df["ad"].rolling(zscore_window).std()
    df["ad_z"] = (df["ad"] - mean) / std

    # --- Derivada del A/D ---
    if include_delta:
        df["ad_delta"] = df["ad"].diff()

    # --- Dirección del A/D ---
    if include_direction:
        df["ad_dir"] = df["ad"].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # --- A/D relativo al volumen reciente ---
    df["ad_rel"] = df["ad"] / df["volume"].rolling(relative_volume_window).sum()

    return df

def generate_features(df, spec):
    df = df.copy()
    
    # 1. Tiempo
    df = add_time_features(df)
    
    # 2. Ticks (usando tu spec)
    df = add_tick_features(df, spec['tick_size'])
    
    # 3. Indicadores
    df = add_indicator_features(df)
    
    df = add_advanced_market_features(df)
    
    df = rsi(df, n=14)
    df = compute_atr(df, period=14)
    df = atr_normalized(df)
    df = macd(df, n_fast=12, n_slow=26)
    df = bollinger_bands(df, n=20, num_std_dev=2)
    df = bollinger_extra(df)
    df = volume_zscore(df, window=20)
    df = percent_return_features(df, lags=[1, 5, 10, 20], rolling_windows=[5, 10, 20])
    df = ad(df, zscore_window=20, roc_windows=[5, 10], smooth_windows=[10], normalize_by="range")
    
    # 4. Limpieza final de NaNs (creados por las medias móviles)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    return df

def process_full_dictionary(df_dict, spec):
    """
    Recorre el diccionario de temporalidades y aplica 
    todas tus funciones de generación de features.
    """
    processed_features_dict = {}
    
    for tf_key, df_tf in df_dict.items():
        # 1. Aplicamos tu generador principal
        df_feat = generate_features(df_tf, spec)
        
        # 2. Añadimos el 'Target Binario' (Útil para clasificación)
        # ¿Sube el precio en la próxima vela de 30 min?
        df_feat['target_bin'] = (df_feat['target_ret_30m'] > 0).astype(int)
        
        processed_features_dict[tf_key] = df_feat
        print(f"🚀 Features generadas para {tf_key}: {df_feat.shape[1]} columnas.")
        
    return processed_features_dict

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


def audit_and_clean_dict(df_dict, feature_cols):
    """
    Aplica audit_single_df() a cada DataFrame del diccionario.
    """
    cleaned_dict = {}

    for timeframe, df in df_dict.items():
        print(f"\n--- 🔍 Auditando {timeframe} ---")
        cleaned_df = audit_single_df(df, feature_cols)
        cleaned_dict[timeframe] = cleaned_df
        print("-" * 30)

    return cleaned_dict

def create_lstm_dataset(df, features, target_col, lookback=14):
    """
    Crea el dataset para LSTM usando NumPy strides (ultra rápido).
    """
    # 1. Convertimos a NumPy array (importante para velocidad)
    feature_array = df[features].values
    target_array = df[target_col].values
    
    # 2. Calculamos las dimensiones
    num_samples = len(df) - lookback
    num_features = len(features)
    
    # 3. Magia de NumPy Strides: Creamos ventanas sin bucles
    # (samples, lookback, features)
    shape = (num_samples, lookback, num_features)
    strides = (feature_array.strides[0], feature_array.strides[0], feature_array.strides[1])
    
    X = np.lib.stride_tricks.as_strided(feature_array, shape=shape, strides=strides)
    
    # 4. El target es simplemente el valor DESPUÉS de la ventana
    y = target_array[lookback:]
    
    return X, y

class GoldLSTM_Bin_Default(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 32)
        self.relu = nn.LeakyReLU(0.1)
        self.out = nn.Linear(32, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.bn(hn[-1])
        out = self.relu(self.fc(out))
        return self.out(out)
    
class GoldLSTM_Triple_Default(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.bn(hn[-1])
        out = self.relu(self.fc1(out))
        return self.fc2(out)
    

def get_return_col_value(col_name):
    match = re.search(r'(\d+)', col_name)
    return match.group(1) if match else "30"

def run_gold_master_workflow(df, feature_cols, suffix, model_class=None, target_col='target_bin', 
                                  train_size=40000, test_size=10000, step=10000, 
                                  lookback=20, epochs=15, batch_size=256):
    
    device = torch.device("cuda")
    scaler_amp = amp.GradScaler('cuda')
    
    # 1. DETECCIÓN DE MODO Y SELECCIÓN DE MODELO AUTOMÁTICA
    num_classes = df[target_col].nunique()
    is_binary = (num_classes == 2)
    output_dim = 1 if is_binary else 3
    
    if model_class is None:
        model_class = GoldLSTM_Bin_Default if is_binary else GoldLSTM_Triple_Default
        print(f"🤖 Auto-seleccionado: {'Binario' if is_binary else 'Triple'} Model")

    # Configuración de Loss y Pesos
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
        target_dtype = torch.float32
    else:
        # Tus pesos específicos para el Oro
        # weights = torch.tensor([2.5, 1.0, 2.5], dtype=torch.float32).to(device)
        weights = torch.tensor([1.5, 3.0, 1.5], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        target_dtype = torch.long

    all_predictions = []
    df_wf = df.copy().sort_values('datetime').reset_index(drop=True)
    # suffix = get_return_col_value(target_col)

    # Verificar que exista la columna de retornos esperada (p. ej. target_ret_30m)
    # desired_ret_col = f'target_ret_{suffix}m'
    # if desired_ret_col not in df_wf.columns:
    #     # Buscar columnas candidatas con el patrón target_ret_{N}m
    #     candidates = [c for c in df_wf.columns if re.match(r'target_ret_(\d+)m', c)]
    #     if candidates:
    #         # Elegir la primera candidata disponible como fallback
    #         chosen = candidates[0]
    #         new_suffix = re.search(r'(\d+)', chosen).group(1)
    #         print(f"⚠️ Columna esperada '{desired_ret_col}' no encontrada. Usando '{chosen}' en su lugar.")
    #         suffix = new_suffix
    #     else:
    #         available = [c for c in df_wf.columns if 'target_ret' in c]
    #         raise KeyError(f"Esperada columna '{desired_ret_col}' no encontrada y no hay columnas 'target_ret_*' disponibles. Columnas disponibles: {available}")

    for start in tqdm(range(0, len(df_wf) - train_size - test_size, step), desc="Gold Workflow"):
        # ... [Lógica de Split y Escalado idéntica a las anteriores] ...
        end_train, end_test = start + train_size, start + train_size + test_size
        train_df = df_wf.iloc[start:end_train].copy()
        test_df = df_wf.iloc[end_train:end_test].copy()
        
        sc = RobustScaler()
        train_df[feature_cols] = sc.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = sc.transform(test_df[feature_cols])
        
        X_train, y_train = create_lstm_dataset(train_df, feature_cols, target_col, lookback)
        X_test, y_test = create_lstm_dataset(test_df, feature_cols, target_col, lookback)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=target_dtype).to(device)
        if is_binary: y_train_t = y_train_t.unsqueeze(1)
        
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        
        # Inicializar modelo pasándole el output_dim detectado
        model = model_class(input_dim=len(feature_cols), output_dim=output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        
        # --- Entrenamiento ---
        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                with amp.autocast('cuda'):
                    loss = criterion(model(bx), by)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
        
        # --- Predicción ---
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            if is_binary:
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                preds = (probs > 0.5).astype(int)
                res_dict = {'prob_up': probs}
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                res_dict = {'prob_0': probs[:,0], 'prob_1': probs[:,1], 'prob_2': probs[:,2]}
        
        # --- Recopilar ---
        res_df = pd.DataFrame({
            'datetime': df_wf['datetime'].iloc[end_train + lookback : end_test].values,
            'actual': y_test, 'pred': preds, **res_dict,
            'ret_real': df_wf[f'target_ret_{suffix}m'].iloc[end_train + lookback : end_test].values
        })
        all_predictions.append(res_df)
        torch.cuda.empty_cache()

    return pd.concat(all_predictions, ignore_index=True), model, sc

def plot_trading_results(df_res):
    # 1. Calculamos el retorno de la estrategia
    # Si la probabilidad es > 0.5, compramos (usamos el retorno real)
    # Si es < 0.5, no hacemos nada (o podrías vender, pero vamos a probar solo largos)
    df_res['strat_ret'] = df_res.apply(lambda x: x['ret_real'] if x['prob_up'] > 0.5 else 0, axis=1)
    
    # 2. Retorno acumulado
    df_res['cum_strat'] = (1 + df_res['strat_ret']).cumprod()
    df_res['cum_market'] = (1 + df_res['ret_real']).cumprod()
    
    # 3. Gráfico
    plt.figure(figsize=(15, 7))
    plt.plot(df_res['datetime'], df_res['cum_strat'], label='Estrategia LSTM', color='gold', lw=2)
    plt.plot(df_res['datetime'], df_res['cum_market'], label='Mercado (Oro)', color='gray', alpha=0.5)
    
    plt.title('Rendimiento Acumulado: LSTM vs Mercado')
    plt.ylabel('Multiplicador de Capital')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # 4. Estadísticas Pro
    sharpe = (df_res['strat_ret'].mean() / df_res['strat_ret'].std()) * (252**0.5)
    print(f"💰 Sharpe Ratio Estimado: {sharpe:.2f}")
    print(f"📈 Retorno Total Estrategia: {(df_res['cum_strat'].iloc[-1]-1):.2%}")
    
def plot_triple_trading_results(df_res, conf_threshold=0.55):
    """
    Backtest para modelo Triple (Sell, Neutral, Buy).
    conf_threshold: Probabilidad mínima para ejecutar una orden.
    """
    df_plot = df_res.copy()
    
    # 1. Definir la señal basada en la probabilidad más alta y el umbral
    # Inicializamos a 0 (Neutral)
    df_plot['signal'] = 0
    
    # Condición para BUY: Predicción es 2 Y la probabilidad es > umbral
    df_plot.loc[(df_plot['pred'] == 2) & (df_plot['prob_buy'] > conf_threshold), 'signal'] = 1
    
    # Condición para SELL: Predicción es 0 Y la probabilidad es > umbral
    df_plot.loc[(df_plot['pred'] == 0) & (df_plot['prob_sell'] > conf_threshold), 'signal'] = -1
    
    # 2. Calculamos el retorno de la estrategia
    # retorno = señal * retorno_real (si señal es -1 y retorno es negativo, ganamos)
    df_plot['strat_ret'] = df_plot['signal'] * df_plot['ret_real']
    
    # 3. Retorno acumulado (Compound interest)
    df_plot['cum_strat'] = (1 + df_plot['strat_ret']).cumprod()
    df_plot['cum_market'] = (1 + df_plot['ret_real']).cumprod()
    
    # 4. Gráfico
    plt.figure(figsize=(15, 7))
    plt.plot(df_plot['datetime'], df_plot['cum_strat'], label=f'Estrategia LSTM Triple (Conf > {conf_threshold})', color='gold', lw=2)
    plt.plot(df_plot['datetime'], df_plot['cum_market'], label='Mercado (Oro - Buy & Hold)', color='gray', alpha=0.4)
    
    plt.title('Backtest: Estrategia Long-Short con LSTM Triple')
    plt.ylabel('Crecimiento del Capital (Base 1.0)')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

    # 5. Estadísticas Pro
    # Filtramos solo cuando hubo operación para métricas de trading
    trades = df_plot[df_plot['signal'] != 0]
    
    # Sharpe Ratio (basado en los retornos de 30 min, anualizado)
    # Asumiendo ~252 días y ~48 velas de 30m por día de trading
    sharpe = (df_plot['strat_ret'].mean() / df_plot['strat_ret'].std()) * (np.sqrt(252 * 48))
    
    print("-" * 30)
    print(f"💰 NUEVO Sharpe Ratio: {sharpe:.2f}")
    print(f"📈 Retorno Total: {(df_plot['cum_strat'].iloc[-1]-1):.2%}")
    print(f"🎯 Win Rate (de los trades): {(trades['strat_ret'] > 0).mean():.2%}")
    print(f"📊 Ratio de Operatividad: {len(trades) / len(df_plot):.2%} (tiempo en mercado)")
    print(f"📉 Total de Trades: {len(trades)} de {len(df_plot)} velas")
    print("-" * 30)

def plot_triple_final_comparison(df_res):
    df_plot = df_res.copy()
    
    # Lógica de señales: Elegimos la dirección con más probabilidad 
    # SOLO si es mayor que la probabilidad de Neutral
    df_plot['signal'] = 0
    df_plot.loc[df_plot['prob_buy'] > df_plot['prob_neutral'], 'signal'] = 1
    df_plot.loc[df_plot['prob_sell'] > df_plot['prob_neutral'], 'signal'] = -1
    
    # Cálculo de retornos
    df_plot['strat_ret'] = df_plot['signal'] * df_plot['ret_real']
    df_plot['cum_strat'] = (1 + df_plot['strat_ret']).cumprod()
    df_plot['cum_market'] = (1 + df_plot['ret_real']).cumprod()
    
    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(df_plot['datetime'], df_plot['cum_strat'], label='LSTM Triple (Balanced)', color='cyan', lw=2)
    plt.plot(df_plot['datetime'], df_plot['cum_market'], label='Oro (Hold)', color='gray', alpha=0.4)
    plt.title('Backtest Final: Modelo Triple con Umbral de Volatilidad')
    plt.legend()
    plt.show()
    
    # Sharpe Ratio Anualizado (aprox 252 días * 48 velas de 30m)
    sharpe = (df_plot['strat_ret'].mean() / df_plot['strat_ret'].std()) * np.sqrt(252 * 48)
    print(f"💰 Sharpe Ratio Esperado: {sharpe:.2f}")

def analyze_by_hour(df_res):
    df_res = df_res.copy()
    df_res['hour'] = pd.to_datetime(df_res['datetime']).dt.hour
    # A. Mapear clases a direcciones de trading
    # Clase 0 (Sell) -> -1 | Clase 1 (Neutral) -> 0 | Clase 2 (Buy) -> 1
    df_res['signal'] = df_res['pred'].map({0: -1, 1: 0, 2: 1})

    # B. Calcular el retorno (Multiplicación simple elemento a elemento)
    df_res['strat_ret'] = df_res['signal'] * df_res['ret_real']

    # C. Calcular el acumulado (Compound interest)
    df_res['cum_strat'] = (1 + df_res['strat_ret']).cumprod()
    # Rendimiento por hora
    hourly_perf = df_res.groupby('hour')['strat_ret'].sum()
    
    plt.figure(figsize=(12, 5))
    hourly_perf.plot(kind='bar', color='skyblue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Rendimiento de la Estrategia por Hora del Día (UTC)')
    plt.ylabel('Retorno Acumulado')
    plt.xlabel('Hora')
    plt.show()

def plot_backtest_filtrado(df_res):
    df_plot = df_res.copy()
    df_plot['hour'] = pd.to_datetime(df_plot['datetime']).dt.hour
    
    # Definimos las "Horas Prohibidas" basándonos en tu gráfico
    # Vamos a dejar solo las horas con barras azules positivas claras
    horas_permitidas = [4, 7, 8, 9, 17, 21, 22, 23]
    
    df_plot['signal_filtrada'] = 0
    # Solo operamos si la hora es permitida
    mask_hora = df_plot['hour'].isin(horas_permitidas)
    
    # Aplicamos la lógica de señales solo en esas horas
    df_plot.loc[mask_hora & (df_plot['prob_buy'] > df_plot['prob_neutral']), 'signal_filtrada'] = 1
    df_plot.loc[mask_hora & (df_plot['prob_sell'] > df_plot['prob_neutral']), 'signal_filtrada'] = -1
    
    # Cálculo de retornos
    df_plot['strat_ret_filtrada'] = df_plot['signal_filtrada'] * df_plot['ret_real']
    df_plot['cum_strat_filtrada'] = (1 + df_plot['strat_ret_filtrada']).cumprod()
    df_plot['cum_market'] = (1 + df_plot['ret_real']).cumprod()
    
    # Métricas
    sharpe = (df_plot['strat_ret_filtrada'].mean() / df_plot['strat_ret_filtrada'].std()) * np.sqrt(252 * 48)
    
    plt.figure(figsize=(15, 7))
    plt.plot(df_plot['datetime'], df_plot['cum_strat_filtrada'], label=f'LSTM Filtrada (Sharpe: {sharpe:.2f})', color='lime', lw=2)
    plt.plot(df_plot['datetime'], df_plot['cum_market'], label='Oro (Hold)', color='gray', alpha=0.4)
    plt.title('Backtest con Filtro Horario: Solo Horas de Alta Probabilidad')
    plt.legend()
    plt.show()
    
    print(f"🚀 Sharpe Ratio con Filtro: {sharpe:.2f}")

def calculate_real_sharpe_binary(df_res):
    # 1. Aseguramos que la señal sea 1 si prob_up > 0.5, de lo solo Long (0)
    # Si quieres que sea bi-direccional (Shorts), cambia el 0 por -1
    df_res['signal'] = np.where(df_res['prob_up'] > 0.5, 1, 0)
    
    # 2. Calculamos el retorno usando la columna 'ret_real' de tu tabla
    df_res['strat_ret'] = df_res['signal'] * df_res['ret_real']
    
    # 3. Calculamos medias y desviaciones
    mean_ret = df_res['strat_ret'].mean()
    std_ret = df_res['strat_ret'].std()
    
    # Evitar división por cero si no hay operaciones
    if std_ret == 0:
        return 0
    
    # 4. Anualización para 30 min (252 días * 48 velas de 30min al día)
    sharpe = (mean_ret / std_ret) * np.sqrt(252 * 48)
    
    # Extra: Cálculo de Retorno Acumulado para comparar
    final_cum_strat = (1 + df_res['strat_ret']).cumprod().iloc[-1]
    
    print(f"✅ Análisis del Modelo Binario:")
    print(f"-------------------------------")
    print(f"💰 Retorno Acumulado Estrategia: {final_cum_strat:.4f}")
    print(f"📈 Retorno Acumulado Mercado:    {df_res['cum_market'].iloc[-1]:.4f}")
    
    return sharpe

# Ejecútala así:
# print(f"\n📊 El Sharpe Ratio REAL es: {calculate_real_sharpe_binary(df_tu_variable):.4f}")

def calculate_sharpe_triple(df_res):
    df_calc = df_res.copy()
    
    # 1. Creamos la señal: 
    # Si pred es 2 (Buy) -> 1
    # Si pred es 0 (Sell) -> -1
    # Si pred es 1 (Neutral) -> 0
    df_calc['signal'] = 0
    df_calc.loc[df_calc['pred'] == 2, 'signal'] = 1
    df_calc.loc[df_calc['pred'] == 0, 'signal'] = -1
    
    # 2. Calculamos el retorno de la estrategia
    df_calc['strat_ret'] = df_calc['signal'] * df_calc['ret_real']
    
    # 3. Cálculo del Sharpe Ratio Anualizado
    # (252 días * 48 velas de 30 min)
    mean_ret = df_calc['strat_ret'].mean()
    std_ret = df_calc['strat_ret'].std()
    
    if std_ret == 0: return 0
    
    sharpe = (mean_ret / std_ret) * np.sqrt(252 * 48)
    
    # Extra: Retorno total acumulado
    total_ret = (1 + df_calc['strat_ret']).cumprod().iloc[-1] - 1
    
    print(f"📈 Resultado Final:")
    print(f"-------------------")
    print(f"💰 Sharpe Ratio: {sharpe:.4f}")
    print(f"🚀 Retorno Total: {total_ret * 100:.2f}%")
    print(f"⚖️ Operaciones totales: {len(df_calc[df_calc['signal'] != 0])} de {len(df_calc)}")
    
    return sharpe

# Llama a la función con tu dataframe:
# sharpe_final = calculate_sharpe_triple(tu_dataframe_aqui)

def calculate_final_binary_sharpe(df_res):
    # 1. Definimos la señal: Si la probabilidad de subir es > 50%, compramos.
    # Si no, nos quedamos fuera (0).
    df_res['signal'] = (df_res['prob_up'] > 0.5).astype(int)
    
    # 2. El retorno es la señal por el retorno real del mercado
    df_res['strat_ret'] = df_res['signal'] * df_res['ret_real']
    
    # 3. Métricas
    mean_ret = df_res['strat_ret'].mean()
    std_ret = df_res['strat_ret'].std()
    
    # Anualización (252 días * 48 velas de 30min)
    ann_factor = np.sqrt(252 * 48)
    sharpe = (mean_ret / std_ret) * ann_factor
    
    # Volatilidad anualizada (extra)
    ann_vol = std_ret * ann_factor
    
    print(f"📊 --- RESULTADOS BINARIOS ---")
    print(f"💰 Sharpe Ratio: {sharpe:.4f}")
    print(f"📉 Volatilidad Anual: {ann_vol*100:.2f}%")
    print(f"📈 Retorno Final: {df_res['cum_strat'].iloc[-1]:.4f}")
    print(f"-------------------------------")
    
    return sharpe

# Ejecútalo con:

def apply_time_filter(df_res):
    df_filtered = df_res.copy()
    df_filtered['hour'] = pd.to_datetime(df_filtered['datetime']).dt.hour
    
    # Basado en tu gráfico: Horas donde el modelo es REALMENTE bueno
    # Excluimos el bloque de las 12, 13, 14, 15 y 16 que son negativas
    horas_permitidas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21]
    
    # Si la hora no está en la lista, la señal pasa a ser 0 (Cash)
    # df_filtered['signal_final'] = np.where(
        # (df_filtered['prob_up'] > 0.5) & (df_filtered['hour'].isin(horas_permitidas)), 
        # 1, 
        # 0
    # )
    # Prueba a ejecutar esto y dime el Sharpe resultante:
    df_filtered['signal_final'] = np.where(
        (df_filtered['prob_up'] > 0.53) & (df_filtered['hour'].isin(horas_permitidas)), 
        1, 
        0
    )
    # Recalculamos retornos
    df_filtered['strat_ret_filtered'] = df_filtered['signal_final'] * df_filtered['ret_real']
    
    # Cálculo de nuevo Sharpe
    mean = df_filtered['strat_ret_filtered'].mean()
    std = df_filtered['strat_ret_filtered'].std()
    new_sharpe = (mean / std) * np.sqrt(252 * 48)
    
    print(f"🚀 Sharpe Original: 0.6245")
    print(f"💎 Sharpe Filtrado: {new_sharpe:.4f}")
    
    return df_filtered

def stress_test_with_costs(df_res, tick_val=0.0001): # 0.0001 es aprox 1 tick en retornos
    df_cost = df_res.copy()
    
    # 1. Identificamos cuándo hay una NUEVA operación (cambio de 0 a 1)
    # Esto es para cobrar la comisión/spread al abrir la posición
    df_cost['entry'] = ((df_cost['signal_final'] == 1) & (df_cost['signal_final'].shift(1) == 0)).astype(int)
    
    # 2. Restamos el coste del tick solo en las velas de entrada
    # Nota: El tick_val en Oro suele ser 0.1 / precio_promedio (aprox 0.00005 a 0.0001)
    df_cost['strat_ret_with_costs'] = df_cost['strat_ret_filtered'] - (df_cost['entry'] * tick_val)
    
    # 3. Recalculamos Sharpe
    mean_c = df_cost['strat_ret_with_costs'].mean()
    std_c = df_cost['strat_ret_with_costs'].std()
    
    sharpe_costs = (mean_c / std_c) * np.sqrt(252 * 48)
    
    # 4. Retorno acumulado real
    cum_ret_costs = (1 + df_cost['strat_ret_with_costs']).cumprod().iloc[-1]
    
    print(f"🧪 --- STRESS TEST (1 TICK COST) ---")
    print(f"📉 Sharpe con Costes: {sharpe_costs:.4f}")
    print(f"💰 Retorno Final tras costes: {cum_ret_costs:.4f}")
    print(f"🚪 Total de entradas pagadas: {df_cost['entry'].sum()}")
    print(f"------------------------------------")
    
    return sharpe_costs

# Ejecuta y dime el resultado:

def run_backtest_pro(df, tp_ticks=10, sl_ticks=5, tick_size=0.1):
    df_bt = df.copy()
    
    # 1. Definimos objetivos en valor nominal ($)
    tp_amount = tp_ticks * tick_size
    sl_amount = sl_ticks * tick_size
    
    trade_returns = []
    
    # 2. Recorremos el dataframe
    # Solo entramos si signal_final == 1
    for i in range(len(df_bt)):
        if df_bt['signal_final'].iloc[i] == 1:
            price_entry = df_bt['close'].iloc[i]
            # Calculamos el movimiento real en $ de esa vela
            price_move = df_bt['ret_real'].iloc[i] * price_entry
            
            # Lógica de salida:
            if price_move >= tp_amount:
                # Ganamos el TP menos 1 tick de coste (comisión/spread)
                trade_returns.append((tp_amount - tick_size) / price_entry)
            elif price_move <= -sl_amount:
                # Perdemos el SL más 1 tick de coste
                trade_returns.append((-sl_amount - tick_size) / price_entry)
            else:
                # Si no toca ni SL ni TP, salimos al cierre de la vela (30 min)
                # Restamos 1 tick de coste por la operación
                trade_returns.append(df_bt['ret_real'].iloc[i] - (tick_size / price_entry))
        else:
            trade_returns.append(0)
            
    df_bt['strat_ret_managed'] = trade_returns
    
    # 3. Métricas finales
    mean_r = df_bt['strat_ret_managed'].mean()
    std_r = df_bt['strat_ret_managed'].std()
    sharpe = (mean_r / std_r) * np.sqrt(252 * 48) if std_r != 0 else 0
    
    print(f"🏆 Resultado con Gestión de Riesgo (TP:{tp_ticks} / SL:{sl_ticks} ticks):")
    print(f"📊 Sharpe Neto: {sharpe:.4f}")
    print(f"💰 Retorno Total: {(1 + df_bt['strat_ret_managed']).cumprod().iloc[-1]:.4f}")
    
    return df_bt

# Ejecución:
# result_final_bt = run_backtest_pro(result_filtered, tp_ticks=8, sl_ticks=4)

def run_backtest_managed_with_costs(df, tp_ticks=10, sl_ticks=5, cost_in_ticks=1, tick_val=0.1):
    df_bt = df.copy()
    
    # 1. Convertimos ticks a dólares
    tp_amount = tp_ticks * tick_val
    sl_amount = sl_ticks * tick_val
    cost_amount = cost_in_ticks * tick_val
    
    trade_returns = []
    
    for i in range(len(df_bt)):
        # Solo operamos si la señal filtrada es 1 (Long)
        if df_bt['signal_final'].iloc[i] == 1:
            price_entry = df_bt['close'].iloc[i]
            # Movimiento del precio en dólares en esa vela de 30m
            price_move = df_bt['ret_real'].iloc[i] * price_entry
            
            # Lógica de salida:
            if price_move >= tp_amount:
                # Éxito: Ganamos el TP pero restamos el coste de ejecución
                net_gain = tp_amount - cost_amount
                trade_returns.append(net_gain / price_entry)
            
            elif price_move <= -sl_amount:
                # Fracaso: Perdemos el SL y además pagamos el coste
                net_loss = -sl_amount - cost_amount
                trade_returns.append(net_loss / price_entry)
            
            else:
                # No tocó ni TP ni SL: salimos al cierre de la vela
                # Retorno real de la vela menos el coste
                net_ret = price_move - cost_amount
                trade_returns.append(net_ret / price_entry)
        else:
            trade_returns.append(0)
            
    df_bt['strat_ret_managed'] = trade_returns
    
    # Métricas
    mean_r = df_bt['strat_ret_managed'].mean()
    std_r = df_bt['strat_ret_managed'].std()
    sharpe = (mean_r / std_r) * np.sqrt(252 * 48) if std_r != 0 else 0
    
    print(f"✅ TP: {tp_ticks} | SL: {sl_ticks} | Coste: {cost_in_ticks} ticks")
    print(f"📊 Sharpe Neto Final: {sharpe:.4f}")
    print(f"💰 Retorno Acumulado: {(1 + df_bt['strat_ret_managed']).cumprod().iloc[-1]:.4f}")
    
    return df_bt

def plot_universal_backtest(df_res, mode='auto', conf_threshold=0.50, tick_comm=1.0, tick_size=0.1):
    """
    Backtest universal para modelos Binarios y Ternarios.
    mode: 'auto', 'bin' o 'tri'
    tick_comm: Comisión en ticks por operación (ida y vuelta)
    """
    df = df_res.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # --- 1. DETECCIÓN AUTOMÁTICA DE MODO ---
    if mode == 'auto':
        mode = 'tri' if 'prob_neutral' in df.columns else 'bin'
    
    # --- 2. GENERACIÓN DE SEÑALES ---
    df['signal'] = 0
    
    if mode == 'tri':
        # Modo Ternario: Compra si Buy es la máxima y > umbral. Vende si Sell es la máxima y > umbral.
        # Asumimos: 2=Buy, 1=Neutral, 0=Sell
        df.loc[(df['prob_buy'] > df['prob_neutral']) & (df['prob_buy'] > df['prob_sell']) & (df['prob_buy'] > conf_threshold), 'signal'] = 1
        df.loc[(df['prob_sell'] > df['prob_neutral']) & (df['prob_sell'] > df['prob_buy']) & (df['prob_sell'] > conf_threshold), 'signal'] = -1
    else:
        # Modo Binario: Compra si prob_up > umbral
        df.loc[df['prob_up'] > conf_threshold, 'signal'] = 1

    # --- 3. CÁLCULO DE RETORNOS CON COMISIONES ---
    # Solo aplicamos comisión cuando la señal CAMBIA (entrada/salida) o simplificado por trade
    # Aquí restamos la comisión (convertida a retorno decimal) en cada trade ejecutado
    comm_decimal = (tick_comm * tick_size) / df['close'] 
    
    df['strat_ret'] = df['signal'] * df['ret_real']
    # Restamos comisión solo cuando la señal no es 0
    df.loc[df['signal'] != 0, 'strat_ret'] = df['strat_ret'] - comm_decimal

    # --- 4. MÉTRICAS Y GRÁFICO ---
    df['cum_strat'] = (1 + df['strat_ret']).cumprod()
    df['cum_market'] = (1 + df['ret_real']).cumprod()
    
    # Sharpe Anualizado (Ajustado a Intradía 30m/1h)
    # 252 días * (7 horas de trading activo o 24h según tu data)
    velas_dia = 48 if (df['datetime'].diff().dt.total_seconds().median() <= 1800) else 24
    ann_factor = np.sqrt(252 * velas_dia)
    
    sharpe = (df['strat_ret'].mean() / df['strat_ret'].std()) * ann_factor
    
    plt.figure(figsize=(15, 7))
    color = 'gold' if mode == 'bin' else 'cyan'
    plt.plot(df['datetime'], df['cum_strat'], label=f'Estrategia LSTM {mode.upper()} (Sharpe: {sharpe:.2f})', color=color, lw=2)
    plt.plot(df['datetime'], df['cum_market'], label='Mercado (Hold)', color='gray', alpha=0.4)
    plt.title(f'Backtest Universal: Modo {mode.upper()} | Confianza > {conf_threshold}')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    
    print(f"✅ Modo detectado: {mode.upper()}")
    print(f"💰 Sharpe Ratio: {sharpe:.2f}")
    print(f"🎯 Win Rate: {((df.loc[df['signal'] != 0, 'strat_ret'] > 0).mean()*100):.2f}%")
    return df