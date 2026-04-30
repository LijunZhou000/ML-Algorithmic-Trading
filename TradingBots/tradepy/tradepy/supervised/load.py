import json
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from tqdm import tqdm
from torch import amp
from sklearn.preprocessing import RobustScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from tradepy.data.loader import load_future
from tradepy.config.config import load_config, load_symbols, load_feature_config
from tradepy.data.cleaner import add_trading_date_by_gap
from tradepy.data.resampler import resample_ohlcv, daily_ohlcv_cummulative
from tradepy.features.generate import generate_features
from tradepy.features.quality_check import data_quality_report

def prepare_all(symbol, minutes, return_horizon_min, json_config_path="features_config.json"):
    df, spec = load_future(symbol)
    df = add_trading_date_by_gap(df)
    df_resampled = resample_ohlcv(df, period=f"{minutes}min")
    df_cumulative = daily_ohlcv_cummulative(df_resampled)
    config = load_feature_config()

    config["global"]["sampling_minutes"] = minutes
    config["global"]["return_horizon_min"] = return_horizon_min
    config["global"]["tick_size"] = spec["tick_size"]
    
    df_final = generate_features(df_cumulative, config)
    return df_final, spec

