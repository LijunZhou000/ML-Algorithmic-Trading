from dataclasses import dataclass


# ─────────────────────────────────────────────
# IB CONFIG
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1


# ─────────────────────────────────────────────
# RETRY CONFIG
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 3
    retry_wait: int = 2