from features.registry import FEATURE_REGISTRY, feature

# Importar submódulos para que los decoradores se ejecuten
from features.momentum import oscillators
from features.trend import moving_averages
from features.volatility import bands
from features.volume import flow
from features.structural import candles
from features.temporal import time_features