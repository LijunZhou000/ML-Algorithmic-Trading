from ib_insync import *
import json

# Conexión a TWS o IB Gateway (puerto 7497 para Demo, 7496 para Real)
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

tickers_to_fetch = [
    # Oro: Prueba con COMEX si NYMEX falla
    {'symbol': 'GC', 'exch': 'COMEX', 'curr': 'USD', 'mult': '100'},
    
    # S&P 500: Este ya te funcionó bien
    {'symbol': 'ES', 'exch': 'CME', 'curr': 'USD', 'mult': '50'},
    
    # Petróleo: Este ya te funcionó bien
    {'symbol': 'CL', 'exch': 'NYMEX', 'curr': 'USD', 'mult': '1000'},
    
    # Libra: Asegúrate de que el multiplicador sea 62500
    # {'symbol': '6B', 'exch': 'CME', 'curr': 'USD', 'mult': '62500'}
    # En la API de IB, para futuros de divisas a veces el símbolo es GBP y la clase 6B
    {'symbol': 'GBP', 'exch': 'CME', 'curr': 'USD', 'mult': '62500'}
]

futures_db = {}

for item in tickers_to_fetch:
    # Añadimos el multiplicador a la definición para ser ultra específicos
    contract = Future(symbol=item['symbol'], 
                      exchange=item['exch'], 
                      currency=item['curr'], 
                      multiplier=item['mult'])
    
    details = ib.reqContractDetails(contract)
    # ... resto del script igual
    
    if details:
        d = details[0] # Tomamos el primer contrato encontrado (usualmente el más cercano)
        summary = d.contract
        
        # Intentamos obtener el margen (requiere simular una orden "WhatIf")
        # Nota: Esto es una estimación de margen inicial
        
        futures_db[item['symbol']] = {
            "name": d.longName,
            "ib_symbol": summary.symbol,
            "conId": summary.conId,
            "exchange": summary.exchange,
            "multiplier": float(summary.multiplier),
            "tick_size": d.minTick,
            "tick_value": d.minTick * float(summary.multiplier),
            "currency": summary.currency,
            "trading_hours": d.tradingHours,
            "liquid_hours": d.liquidHours,
            # Estimación de costes fijos (Comisión + Exchange Fees)
            "approx_total_fee_per_side": 2.50, 
            "min_slippage_ticks": 1 
        }
        print(f"✅ Datos de {item['symbol']} obtenidos.")
    else:
        print(f"❌ No se encontró info para {item['symbol']}")

# Guardar a JSON
with open('instrument_definitions.json', 'w') as f:
    json.dump(futures_db, f, indent=4)

ib.disconnect()