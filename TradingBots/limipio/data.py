from ib_insync import *
import pandas as pd
from datetime import datetime

HOST = '127.0.0.1'
PORT = 7497 
CLIENT_ID = 10

def download_liquid_contracts():
    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID)
        print("✅ Conectado a IBKR")

        gold_generic = Future('GC', exchange='COMEX', currency='USD')
        details = ib.reqContractDetails(gold_generic)
        
        # Obtenemos la fecha actual en formato YYYYMM para filtrar
        current_date = datetime.now().strftime('%Y%m')
        
        # 1. Extraemos contratos
        contracts = [d.contract for d in details]
        
        # 2. FILTRO CLAVE: Solo contratos con vencimiento entre 2025 y finales de 2026
        # Esto evita los contratos vacíos de 2030 que vimos antes.
        liquid_contracts = [
            c for c in contracts 
            if '202501' <= c.lastTradeDateOrContractMonth <= '202612'
        ]
        
        # Ordenamos por vencimiento (el más reciente primero)
        liquid_contracts.sort(key=lambda x: x.lastTradeDateOrContractMonth, reverse=True)
        
        # Tomamos los 4 más relevantes (puedes ajustar el número)
        target_contracts = liquid_contracts[:4]
        
        for contract in target_contracts:
            symbol = contract.localSymbol
            expiry = contract.lastTradeDateOrContractMonth
            print(f"\n🚀 Procesando contrato LÍQUIDO: {symbol} (Vence: {expiry})")
            
            # Intentamos bajar 6 meses de datos
            # Nota: Si el contrato es muy nuevo, bajará menos tiempo automáticamente
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='6 M', 
                barSizeSetting='1 hour', 
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )
            
            if bars:
                df = util.df(bars)
                df.rename(columns={'date': 'datetime'}, inplace=True)
                filename = f"GOLD_FUT_{symbol}_{expiry}.csv"
                df.to_csv(filename, index=False)
                print(f"💾 Guardado: {filename} ({len(df)} velas)")
            else:
                # Si TRADES falla (a veces pasa en demo), probamos con MIDPOINT
                print(f"⚠️ Sin trades para {symbol}, probando MIDPOINT...")
                bars = ib.reqHistoricalData(contract, '', '6 M', '1 hour', 'MIDPOINT', False, 1)
                if bars:
                    df = util.df(bars)
                    df.to_csv(f"GOLD_FUT_{symbol}_MIDPOINT.csv", index=False)
                    print(f"💾 Guardado MIDPOINT para {symbol}")

        ib.disconnect()
        print("\n✨ Proceso finalizado.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_liquid_contracts()