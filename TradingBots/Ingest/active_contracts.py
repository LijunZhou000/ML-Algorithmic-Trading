import json
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)

# Cargamos tus definiciones previas
with open('instrument_definitions.json', 'r') as f:
    instruments = json.json.load(f)

active_contracts = {}

for symbol, data in instruments.items():
    # Creamos el contrato genérico
    fut = Future(symbol=symbol, exchange=data['exchange'], currency=data['currency'])
    
    # Pedimos TODOS los vencimientos disponibles
    details = ib.reqContractDetails(fut)
    
    if details:
        # Ordenamos por fecha de vencimiento y tomamos los 2 primeros (Front y Back month)
        # Normalmente el Front Month es el que tiene el volumen
        sorted_contracts = sorted(details, key=lambda x: x.contract.lastTradeDateOrContractMonth)
        
        # Guardamos el contrato actual (Front Month)
        front_contract = sorted_contracts[0].contract
        
        active_contracts[symbol] = {
            "localSymbol": front_contract.localSymbol,
            "conId": front_contract.conId,
            "expiry": front_contract.lastTradeDateOrContractMonth,
            "multiplier": front_contract.multiplier,
            "liquid_hours": sorted_contracts[0].liquidHours
        }
        print(f"✅ Contrato activo para {symbol}: {front_contract.localSymbol} (Vence: {front_contract.lastTradeDateOrContractMonth})")

# Guardar el JSON dinámico
with open('active_contracts.json', 'w') as f:
    json.dump(active_contracts, f, indent=4)

ib.disconnect()