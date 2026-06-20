from ib_insync import *
import pandas as pd
from tradepy.data.loader import load_specs
from tradepy.config.config import load_config, load_symbols, load_feature_config, load_exclude_config


def get_data(specs, symbol):
    info = specs[symbol]
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=3)
    future = Future(
    symbol=symbol,
    lastTradeDateOrContractMonth="202606",
    exchange=info['exchange'],
    currency=info['currency'],
    includeExpired=True
)

    data = ib.reqHistoricalData(future, "", "6 M", "1 min", "TRADES", 1, 1, False, [])
    ib.disconnect()
    return data

if __name__ == "__main__":
    specs = load_specs()
    symbols = load_symbols()
    ASSET = symbols[5]
    get_data(specs, ASSET)