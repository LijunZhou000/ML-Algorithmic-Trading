from ib_insync import IB, Future
from datetime import datetime


def parse_date(s):
    if not s:
        return None
    s = s.split()[0]
    try:
        if len(s) == 6:
            return datetime(int(s[:4]), int(s[4:6]), 1)
        return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None


def front_contract(ib, symbol, preferred_exchange=None):
    # Busca todos los contratos de futuros para ese símbolo
    details = ib.reqContractDetails(Future(symbol))
    if not details:
        return None

    # Si se indica exchange preferido, filtramos
    if preferred_exchange:
        filtered = [d for d in details if d.contract.exchange == preferred_exchange]
        if filtered:
            details = filtered

    # Tomamos el front month >= primer día de este mes
    today = datetime.today().replace(day=1)
    candidates = []
    for d in details:
        dt = parse_date(d.contract.lastTradeDateOrContractMonth)
        if dt and dt >= today:
            candidates.append((dt, d.contract))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def main():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=7)

    # 3 = delayed; 1 = realtime si tienes subscripción
    ib.reqMarketDataType(3)  # datos diferidos gratuitos si están disponibles[web:1][web:5][web:7]

    prefs = {
        'GB': 'ICEUS',
        'CL': 'NYMEX',
        'ES': 'CME',
        'GC': 'COMEX'
    }

    try:
        for sym in ['GB', 'CL', 'ES', 'GC']:
            contract = front_contract(ib, sym, preferred_exchange=prefs.get(sym))
            if not contract:
                print(sym, 'no contract found')
                continue

            qualified = ib.qualifyContracts(contract)
            if not qualified:
                print(sym, 'no qualified contract')
                continue
            qc = qualified[0]

            # Petición de market data diferida en streaming (no snapshot)
            ticker = ib.reqMktData(qc, snapshot=False)
            ib.sleep(3)  # espera unos segundos a que lleguen los ticks delayed
            print(
                sym,
                qc.localSymbol or qc.conId,
                'bid:', ticker.bid,
                'ask:', ticker.ask,
                'last:', ticker.last
            )
            ib.cancelMktData(ticker)

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
