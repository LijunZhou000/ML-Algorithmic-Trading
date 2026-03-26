from ib_insync import *
import json

from order_utils import go_long

def main():
    ib = IB()
    # clientId=3 está bien, asegúrate de que no haya otro script usándolo
    ib.connect('127.0.0.1', 7497, clientId=3)

    with open("../Ingest/futuros_specs.json") as f:
        specs = json.load(f)

    # 1. Seleccionamos el activo (Oro en este caso)
    spec = specs["GC"] 

    # 2. Lógica de cierre:
    # Si en TWS ves que tu posición es -1 (Short), 
    # necesitas comprar (go_long) exactamente 1 para llegar a 0.
    print("Enviando orden de compra para cerrar posición Short...")
    
    trade = go_long(
        ib, 
        spec, 
        quantity=1  # La cantidad que tenías en Short
    )

    # 3. Verificación
    # Esto espera un poco para confirmar que la orden se ejecutó
    ib.sleep(1) 
    if trade.orderStatus.status == 'Filled':
        print("Posición cerrada con éxito. Ahora estás en 0.")
    else:
        print(f"Estado de la orden: {trade.orderStatus.status}")

    ib.disconnect()

if __name__ == "__main__":
    main()
