from tradepy.load import import_dataset, clean, wavelet_denoising, resample_ohlcv, daily_ohlcv_cummulative
from tradepy.features import generate_features
import json
import numpy as np

def prepare_all(asset, minutes, return_horizon_min, json_config_path="features_config.json"):
    df_min, spec = import_dataset(asset=asset)
    df_min_clean = clean(df_min, asset=asset)
    df_min_clean["close"] = wavelet_denoising(df_min_clean['close'].values.copy())
    df_resampled = resample_ohlcv(df_min_clean, period=f"{minutes}min")
    df_cumulative = daily_ohlcv_cummulative(df_resampled)
    with open(json_config_path) as f:
        config = json.load(f)

    config["global"]["sampling_minutes"] = minutes
    config["global"]["return_horizon_min"] = return_horizon_min
    config["global"]["tick_size"] = spec["tick_size"]
    
    df_final = generate_features(df_cumulative, config_json=config, dropna_strategy='any')
    return df_final, spec

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