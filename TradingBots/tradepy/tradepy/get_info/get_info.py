import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ib_insync import IB, Future, MarketOrder

from tradepy.paths import ensure_dirs
from tradepy.config import RetryConfig

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def clean_margin(val: str | None) -> float:
    DOUBLE_MAX = "1.7976931348623157E308"

    if val is None or str(val) == DOUBLE_MAX:
        return 0.0

    try:
        return abs(float(val))
    except (ValueError, TypeError):
        return 0.0


def get_front_month(details: list):
    today = datetime.now(timezone.utc).strftime('%Y%m%d')

    valid = [
        d for d in details
        if d.contract.lastTradeDateOrContractMonth >= today
    ]

    if not valid:
        return None

    return min(valid, key=lambda d: d.contract.lastTradeDateOrContractMonth)


def compute_min_profit_ticks(fee_per_side: float, slippage_ticks: int, tick_value: float) -> float:
    if tick_value <= 0:
        return 0.0

    fee_roundtrip = fee_per_side * 2
    slippage_roundtrip = slippage_ticks * tick_value * 2

    return round((fee_roundtrip + slippage_roundtrip) / tick_value, 1)


def fetch_with_retry(
    ib: IB,
    contract: Future,
    retry_config: RetryConfig
):
    for attempt in range(1, retry_config.max_retries + 1):
        details = ib.reqContractDetails(contract)

        if details:
            return details

        log.warning(
            f"Intento {attempt}/{retry_config.max_retries} sin resultado, "
            f"esperando {retry_config.retry_wait}s..."
        )
        ib.sleep(retry_config.retry_wait)

    return []


# ─────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────
def fetch_futures_dynamic(
    ib: IB,
    static_path: Path,
    output_path: Path,
    retry_config: RetryConfig,
) -> dict:

    ensure_dirs()

    if not static_path.exists():
        raise FileNotFoundError(f"No se encuentra: {static_path}")

    with open(static_path) as f:
        static = json.load(f)

    log.info(f"Cargados {len(static)} activos desde {static_path}")

    futures_dynamic = {}
    failed = []

    for key, info in static.items():

        log.info(f"Procesando {key} ({info['ib_symbol']})...")

        contract = Future(
            symbol=info['ib_symbol'],
            exchange=info['exchange'],
            currency=info['currency'],
        )

        details = fetch_with_retry(ib, contract, retry_config)

        if not details:
            log.error(f"No info para {key}")
            failed.append(key)
            continue

        detail = get_front_month(details)

        if detail is None:
            log.error(f"Sin contratos vigentes: {key}")
            failed.append(key)
            continue

        qualified = detail.contract
        ib.qualifyContracts(qualified)

        # ─────────────────────────────
        # MÁRGENES
        # ─────────────────────────────
        try:
            order = MarketOrder('BUY', 1)
            analysis = ib.whatIfOrder(qualified, order)

            init_margin = clean_margin(analysis.initMarginChange)
            maint_margin = clean_margin(analysis.maintMarginChange)

            if init_margin == 0:
                init_margin = abs(
                    float(analysis.initMarginAfter or 0) -
                    float(analysis.initMarginBefore or 0)
                )

            if maint_margin == 0:
                maint_margin = abs(
                    float(analysis.maintMarginAfter or 0) -
                    float(analysis.maintMarginBefore or 0)
                )

        except Exception as e:
            log.warning(f"WhatIf falló ({key}): {e}")

            init_margin = info.get("fallback_margin", 0.0)
            maint_margin = init_margin * 0.9

        # ─────────────────────────────
        # TICK DATA
        # ─────────────────────────────
        tick_size = detail.minTick
        multiplier = float(qualified.multiplier)
        tick_value = round(tick_size * multiplier, 4)

        min_profit_ticks = compute_min_profit_ticks(
            info['approx_total_fee_per_side'],
            info['min_slippage_ticks'],
            tick_value,
        )

        # ─────────────────────────────
        # OUTPUT
        # ─────────────────────────────
        futures_dynamic[key] = {
            "name": detail.longName,
            "ib_symbol": qualified.symbol,
            "conId": qualified.conId,
            "exchange": qualified.exchange,
            "multiplier": multiplier,
            "tick_size": tick_size,
            "tick_value": tick_value,
            "currency": qualified.currency,
            "trading_hours": detail.tradingHours,
            "liquid_hours": detail.liquidHours,
            "approx_total_fee_per_side": info['approx_total_fee_per_side'],
            "min_slippage_ticks": info['min_slippage_ticks'],
            "max_spread_ticks": info['max_spread_ticks'],
            "min_profit_ticks": min_profit_ticks,
            "initial_margin": round(init_margin, 2),
            "maint_margin": round(maint_margin, 2),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            f"OK {key} | tick={tick_value} | "
            f"margin={init_margin:.2f} | "
            f"profit_ticks={min_profit_ticks}"
        )

    # ─────────────────────────────
    # SAVE
    # ─────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(futures_dynamic, f, indent=4)

    log.info(
        f"Guardado {len(futures_dynamic)} OK · {len(failed)} fallidos → {output_path}"
    )

    if failed:
        log.warning(f"Fallidos: {failed}")

    return futures_dynamic