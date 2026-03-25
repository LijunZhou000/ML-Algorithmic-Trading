from ib_insync import *
import json
from order_utils import add_sl_tp_to_existing_position

def rebuild_contract_from_position(ib, pos):
    c = Contract()
    c.conId = pos.contract.conId
    c.exchange = pos.contract.exchange or "COMEX"
    c.secType = "FUT"
    c.currency = pos.contract.currency or "USD"
    ib.qualifyContracts(c)
    return c

def main():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=10)

    # 1. Obtener la posición abierta
    positions = ib.positions()
    if not positions:
        print("No hay posiciones abiertas.")
        return

    pos = positions[0]

    # 2. Reconstruir contrato completo
    contract = rebuild_contract_from_position(ib, pos)

    print("Contrato reconstruido:", contract)

    # 3. Valores recomendados
    tp = 4485.00
    sl = 4510.00

    # 4. Cantidad (si estás short 1 contrato → -1)
    quantity = pos.position

    # 5. Añadir SL/TP
    add_sl_tp_to_existing_position(ib, contract, tp, sl, quantity)

    ib.disconnect()

if __name__ == "__main__":
    main()
