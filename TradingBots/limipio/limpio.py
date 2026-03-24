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

def clean(df, asset='gc1'):
    df = df.copy()
    # 1. Eliminar Sábados (Día 5) - Ruido absoluto detectado en el EDA
    df = df[df['datetime'].dt.weekday != 5]
    # 2. Definir el inicio de la sesión (Trading Day)
    # En tus datos el hueco es de 00:00 a 01:00. 
    # Restamos 1 hora para que la vela de la 01:05 sea la primera del "día"
    df['trading_date'] = (df['datetime'] - pd.Timedelta(hours=1)).dt.date
    # 3. Filtrar el ruido del "Break" (15 min antes y 5 min después)
    # Minutos totales del día:
    minutes = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    # Filtramos: antes de las 00:00 (cierre) y justo al abrir (01:00)
    # Esto elimina spreads gigantes que la LSTM no puede predecir
    mask_noise = (minutes >= 1425) | (minutes == 60) 
    df = df[~mask_noise]
    # 4. Eliminar duplicados y asegurar orden temporal
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
    # 5. Resetear índice para que la secuencia de la LSTM sea continua
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
        "openint": "last"
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

def plot_master_market_analysis(df, asset_name="Future", tick_size=0.1, 
                                price_plots=True, tick_plots=True, max_autocorr=10000):
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
    # VWAP Intradía (se reinicia cada día)
    df_diag['tp'] = (df_diag['high'] + df_diag['low'] + df_diag['close']) / 3
    df_diag['pv'] = df_diag['tp'] * df_diag['volume']
    
    # Agrupamos por fecha para el acumulado diario
    date_group = df_diag.groupby(df_diag.index.date)
    df_diag['vwap'] = date_group['pv'].cumsum() / date_group['volume'].cumsum()
    
    # Z-Score del VWAP (Desviación)
    vwap_std = date_group['tp'].transform(lambda x: x.expanding().std())
    df_diag['vwap_zscore'] = (df_diag['close'] - df_diag['vwap']) / (vwap_std + 0.001)
    
    # Efficiency Ratio (Fractalidad a 30 min)
    lookback_eff = 30
    net_chg = df_diag['close'].diff(lookback_eff).abs()
    path_chg = df_diag['close'].diff().abs().rolling(window=lookback_eff).sum()
    df_diag['efficiency_ratio'] = net_chg / (path_chg + 0.001)

    # Métricas de Ticks
    df_diag['vol_per_tick'] = df_diag['volume'] / (df_diag['ticks_range'] + 0.1)
    df_diag['intensity'] = df_diag['vol_per_tick']
    df_diag['speed'] = df_diag['ticks_range'] # Ticks por minuto

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # 3. Definición Dinámica de la Rejilla (Grid)
    price_rows = 5 if price_plots else 0
    tick_rows = 6 if tick_plots else 0 # Aumentado para VWAP y Speed
    total_rows = price_rows + tick_rows
    
    fig = plt.figure(figsize=(18, 4 * total_rows))
    gs = fig.add_gridspec(total_rows, 2)
    plt.suptitle(f"MASTER DASHBOARD PROFESIONAL: {asset_name}", fontsize=22, y=0.99, fontweight='bold')

    row = 0
    
    # --- SECCIÓN DE PRECIO ---
    if price_plots:
        # Fila 1: Precio Diario
        ax0 = fig.add_subplot(gs[row, :])
        df_daily = df_diag['close'].resample('D').last().dropna()
        ax0.plot(df_daily.index, df_daily.values, color='#1a508b', linewidth=1.5)
        ax0.set_title("Evolución del Precio (Cierre Diario)")
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
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
    df['target_ret_30m'] = np.log(df['close'].shift(-1) / df['close'])
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

def generate_features(df, spec):
    df = df.copy()
    
    # 1. Tiempo
    df = add_time_features(df)
    
    # 2. Ticks (usando tu spec)
    df = add_tick_features(df, spec['tick_size'])
    
    # 3. Indicadores
    df = add_indicator_features(df)
    
    df = add_advanced_market_features(df)
    
    # 4. Limpieza final de NaNs (creados por las medias móviles)
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

def audit_and_clean_dict(df_dict, feature_cols):
    """
    Escanea el diccionario buscando NaNs e Infinitos en las columnas de interés.
    Limpia y reporta el estado de cada DataFrame.
    """
    cleaned_dict = {}
    
    for timeframe, df in df_dict.items():
        print(f"--- 🔍 Auditando {timeframe} ---")
        
        # 1. Detectar Infinitos (común en divisiones por vol_per_tick)
        inf_count = np.isinf(df[feature_cols]).sum().sum()
        if inf_count > 0:
            print(f"⚠️ ¡Atención! Encontrados {inf_count} valores infinitos. Reemplazando por NaN...")
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # 2. Detectar NaNs
        nan_count = df[feature_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️ Encontrados {nan_count} valores NaN (probablemente por rolling windows).")
            # Dropeamos las filas que no tengan las features completas
            df_cleaned = df.dropna(subset=feature_cols)
            print(f"✅ Filas restantes tras dropna: {len(df_cleaned)}")
        else:
            df_cleaned = df.copy()
            print("💎 Datos limpios. Sin NaNs.")

        cleaned_dict[timeframe] = df_cleaned
        print("-" * 25)
        
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

def build_lstm_model_30m(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True), # Más neuronas para 30m
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dense(3, activation='softmax') 
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005), # LR más bajo para mayor estabilidad
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_lstm_model_bin(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        # Cambiamos a 1 neurona con Sigmoid para clasificación binaria
        Dense(1, activation='sigmoid') 
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy', # Pérdida binaria
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def run_walk_forward_gold(df, feature_cols, target_col='target_class', 
                          train_size=40000, test_size=10000, step=10000, 
                          lookback=20, epochs=15, batch_size=64):
    """
    Ejecuta un esquema de validación Walk-Forward con ventana móvil.
    """

    all_predictions = []
    last_model = None
    last_scaler = None
    
    # 1. Asegurar limpieza y orden
    df_wf = df.copy().sort_values('datetime').reset_index(drop=True)
    
    # 2. Bucle de Ventanas Móviles
    for start in range(0, len(df_wf) - train_size - test_size, step):
        end_train = start + train_size
        end_test = end_train + test_size
        
        # Separación de datos
        train_df = df_wf.iloc[start:end_train].copy()
        test_df = df_wf.iloc[end_train:end_test].copy()
        
        # --- A) ESCALADO (Fit solo en TRAIN) ---
        scaler = RobustScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
        # --- B) CREAR TENSORES ---
        # Usamos tu función create_lstm_dataset (debe estar definida globalmente)
        X_train, y_train = create_lstm_dataset(train_df, feature_cols, target_col, lookback)
        X_test, y_test = create_lstm_dataset(test_df, feature_cols, target_col, lookback)
        
        # --- C) MODELO ---
        # Re-inicializamos para cada ventana para que aprenda el régimen actual
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model_30m(input_shape) # Tu función de arquitectura
        
        es = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
        
        print(f"\n🔄 Entrenando Ventana: {df_wf['datetime'].iloc[start]} --> {df_wf['datetime'].iloc[end_train]}")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  verbose=0, callbacks=[es])
        
        # --- D) PREDICCIÓN ---
        preds = model.predict(X_test)
        pred_classes = np.argmax(preds, axis=1)
        
        # --- E) RECOPILAR RESULTADOS ---
        # Sincronizamos con las fechas y retornos para el análisis posterior
        res_df = pd.DataFrame({
            'datetime': df_wf['datetime'].iloc[end_train + lookback : end_test].values,
            'actual': y_test,
            'pred': pred_classes,
            'prob_0': preds[:, 0], # Prob Sell
            'prob_1': preds[:, 1], # Prob Neutral
            'prob_2': preds[:, 2], # Prob Buy
            'ret_real': df_wf['target_ret_30m'].iloc[end_train + lookback : end_test].values
        })
        
        all_predictions.append(res_df)
        last_model = model # Nos quedamos con el último modelo (el más "actual")
        last_scaler = scaler
        
        acc = (pred_classes == y_test).mean()
        print(f"✅ Ventana completada. Acc: {acc:.2%}")

    # Unir todos los resultados
    full_results = pd.concat(all_predictions, ignore_index=True)
    
    return {
        'model': last_model,
        'scaler': last_scaler,
        'results': full_results
    }
    
def run_walk_forward_bin(df, feature_cols, target_col='target_bin', 
                         train_size=40000, test_size=10000, step=10000, 
                         lookback=20, epochs=15, batch_size=64):
    """
    Ejecuta validación Walk-Forward para clasificación BINARIA (Sube/No sube).
    """
    all_predictions = []
    last_model = None
    last_scaler = None
    
    df_wf = df.copy().sort_values('datetime').reset_index(drop=True)
    
    for start in range(0, len(df_wf) - train_size - test_size, step):
        end_train = start + train_size
        end_test = end_train + test_size
        
        train_df = df_wf.iloc[start:end_train].copy()
        test_df = df_wf.iloc[end_train:end_test].copy()
        
        # A) ESCALADO
        scaler = RobustScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
        # B) TENSORES
        X_train, y_train = create_lstm_dataset(train_df, feature_cols, target_col, lookback)
        X_test, y_test = create_lstm_dataset(test_df, feature_cols, target_col, lookback)
        
        # C) MODELO BINARIO
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model_bin(input_shape)
        
        es = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
        
        print(f"\n🔄 Entrenando Ventana Binaria: {df_wf['datetime'].iloc[start]} --> {df_wf['datetime'].iloc[end_train]}")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  verbose=0, callbacks=[es])
        
        # D) PREDICCIÓN (Probabilidad 0.0 a 1.0)
        probs = model.predict(X_test).ravel()
        # Umbral estándar de 0.5 para la clase predicha
        pred_classes = (probs > 0.5).astype(int)
        
        # E) RESULTADOS
        res_df = pd.DataFrame({
            'datetime': df_wf['datetime'].iloc[end_train + lookback : end_test].values,
            'actual': y_test,
            'pred': pred_classes,
            'prob_up': probs, # Probabilidad de que el precio suba
            'ret_real': df_wf['target_ret_30m'].iloc[end_train + lookback : end_test].values
        })
        
        all_predictions.append(res_df)
        last_model = model
        last_scaler = scaler
        
        acc = (pred_classes == y_test).mean()
        print(f"✅ Ventana completada. Acc: {acc:.2%} | Buy Ratio: {pred_classes.mean():.2%}")

    full_results = pd.concat(all_predictions, ignore_index=True)
    
    return {
        'model': last_model,
        'scaler': last_scaler,
        'results': full_results
    }