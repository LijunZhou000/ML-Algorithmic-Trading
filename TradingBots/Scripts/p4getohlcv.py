from ib_insync import IB, Future
import pandas as pd
from tradepy.data.loader import load_specs
import time

# ============================================================
# 1) Resolver automáticamente el contrato activo del IBEX
# ============================================================

def resolve_ibex_contract(spec):
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=2)

    # Pedimos todos los futuros del IBEX
    chain = ib.reqContractDetails(
        Future(symbol=spec["ib_symbol"], exchange=spec["exchange"], currency=spec["currency"])
    )

    # Filtramos solo los que tienen expiry
    valid = [c.contract for c in chain if c.contract.lastTradeDateOrContractMonth]

    # Ordenamos por expiración
    valid = sorted(valid, key=lambda c: c.lastTradeDateOrContractMonth)

    ib.disconnect()
    return valid[0]  # contrato activo más cercano


# ============================================================
# 2) Descargar 1Y OHLCV del IBEX usando el contrato resuelto
# ============================================================

def get_ibex_1y_ohlcv(spec):
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=3)

    # Obtenemos el contrato correcto (con expiry real)
    contract = resolve_ibex_contract(spec)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="6 M",
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
        keepUpToDate=False
    )

    df = pd.DataFrame([{
        "time": b.date,
        "open": b.open,
        "high": b.high,
        "low": b.low,
        "close": b.close,
        "volume": b.volume
    } for b in bars])

    ib.disconnect()
    return df


# ============================================================
# 3) Ejemplo de uso
# ============================================================

if __name__ == "__main__":
    print("Descargando 1Y OHLCV del futuro del IBEX (contrato activo real)...")
    specs = load_specs()
    for ticker, spec in specs.items():
        df = get_ibex_1y_ohlcv(spec)
        df.to_parquet(f"../Data/bt/{ticker}1min.parquet", index=False)
        time.sleep(2)  # Evitar sobrecargar la API de IB
