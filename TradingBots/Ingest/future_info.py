from ib_insync import *
import json

# Configuración de conexión
IB_HOST = '127.0.0.1'
IB_PORT = 7497  # 7497 para Demo, 7496 para Real
CLIENT_ID = 1

def fetch_futures_specs():
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return

    # Definición de activos con el mapeo de nombres de KEY
    # 'key': El nombre que quieres en el JSON final
    # 'symbol': El símbolo que entiende la API de IB
    tickers_to_fetch = [
        {'key': 'GC', 'symbol': 'GC', 'exch': 'COMEX', 'curr': 'USD', 'mult': '100'},
        {'key': 'ES', 'symbol': 'ES', 'exch': 'CME', 'curr': 'USD', 'mult': '50'},
        {'key': 'CL', 'symbol': 'CL', 'exch': 'NYMEX', 'curr': 'USD', 'mult': '1000'},
        {'key': 'BP', 'symbol': 'GBP', 'exch': 'CME', 'curr': 'USD', 'mult': '62500'} 
    ]

    futures_db = {}

    for item in tickers_to_fetch:
        print(f"🔍 Procesando {item['key']} ({item['symbol']})...")
        
        # 1. Definir y calificar el contrato
        contract = Future(symbol=item['symbol'], 
                          exchange=item['exch'], 
                          currency=item['curr'], 
                          multiplier=item['mult'])
        
        details = ib.reqContractDetails(contract)
        
        if not details:
            print(f"❌ No se encontró info para {item['symbol']}")
            continue

        # Tomamos el contrato más cercano (front month)
        d = details[0]
        qualified_contract = d.contract
        ib.qualifyContracts(qualified_contract)

        # 2. Simular orden para obtener márgenes (WhatIf)
        order = MarketOrder('BUY', 1)
        analysis = ib.whatIfOrder(qualified_contract, order)

        # Manejo de valores nulos o "Double Max" de la API de IB
        def clean_margin(val):
            if val == "1.7976931348623157E308" or val is None:
                return 0.0
            return abs(float(val))

        init_margin = clean_margin(analysis.initMarginChange)
        maint_margin = clean_margin(analysis.maintMarginChange)

        # Si el cambio es 0, intentamos con la diferencia absoluta
        if init_margin == 0:
            init_margin = abs(float(analysis.initMarginAfter) - float(analysis.initMarginBefore))
        if maint_margin == 0:
            maint_margin = abs(float(analysis.maintMarginAfter) - float(analysis.maintMarginBefore))

        # 3. Construir el objeto de datos
        futures_db[item['key']] = {
            "name": d.longName,
            "ib_symbol": qualified_contract.symbol,
            "conId": qualified_contract.conId,
            "exchange": qualified_contract.exchange,
            "multiplier": float(qualified_contract.multiplier),
            "tick_size": d.minTick,
            "tick_value": round(d.minTick * float(qualified_contract.multiplier), 2),
            "currency": qualified_contract.currency,
            "trading_hours": d.tradingHours,
            "liquid_hours": d.liquidHours,
            "approx_total_fee_per_side": 2.50,
            "min_slippage_ticks": 1,
            "initial_margin": round(init_margin, 2),
            "maint_margin": round(maint_margin, 2)
        }
        print(f"✅ Datos y márgenes de {item['key']} obtenidos.")

    # 4. Guardar en el archivo final
    with open('futuros_specs.json', 'w') as f:
        json.dump(futures_db, f, indent=4)
    
    print(f"\n📁 Archivo 'futuros_specs.json' actualizado con {len(futures_db)} activos.")
    
    ib.disconnect()

if __name__ == "__main__":
    fetch_futures_specs()