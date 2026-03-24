from ib_insync import *
import json

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=10)

with open('instrument_definitions.json', 'r') as f:
    instruments = json.load(f)

for symbol, data in instruments.items():
    contract = Contract(conId=data['conId'], exchange=data['exchange'])
    ib.qualifyContracts(contract)
    
    order = MarketOrder('BUY', 1)
    analysis = ib.whatIfOrder(contract, order)
    
    # En cuentas Demo o con datos retardados, a veces estos campos 
    # vienen en 'initMarginChange' o 'maintMarginChange'
    initial = float(analysis.initMarginChange) if analysis.initMarginChange != "1.7976931348623157E308" else 0.0
    maint = float(analysis.maintMarginChange) if analysis.maintMarginChange != "1.7976931348623157E308" else 0.0

    # Si sigue saliendo 0, intentamos con la diferencia post-orden
    if initial == 0:
        # A veces el impacto se ve en la diferencia de margen total
        initial = float(analysis.initMarginAfter) - float(analysis.initMarginBefore)

    data['initial_margin'] = abs(initial)
    data['maint_margin'] = abs(maint)
    
    print(f"📊 {symbol}: Margen Inicial ${data['initial_margin']}")
    
# Guardamos el JSON actualizado
with open('instrument_definitions.json', 'w') as f:
    json.dump(instruments, f, indent=4)

ib.disconnect()