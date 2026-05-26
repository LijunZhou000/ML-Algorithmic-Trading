import logging, json
from pathlib import Path

LOG_BASE = Path("/var/log/tradingbot")

def get_logger(category: str) -> logging.Logger:
    logger = logging.getLogger(category)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_BASE / category / f"{category}.log")
        handler.setFormatter(logging.Formatter('%(message)s'))  # JSON puro
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

def log_movement(ticker, direction, price, size):
    get_logger("movements").info(json.dumps({
        "level": "info",
        "ticker": ticker,
        "direction": direction,   # "long" / "short"
        "price": price,
        "size": size,
    }))

def log_balance(balance, equity, drawdown):
    get_logger("balance").info(json.dumps({
        "level": "info",
        "balance": balance,
        "equity": equity,
        "drawdown_pct": drawdown,
    }))