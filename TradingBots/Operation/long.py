from ib_insync import *
import json

from order_utils import go_long

def main():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=3)

    with open("futuros_specs.json") as f:
        specs = json.load(f)

    spec = specs["GC"]  # Gold

    # Ejemplo: long 1 contrato con TP y SL
    long_with_bracket(
        ib,
        spec,
        quantity=1,
        tp=2400.0,   # take profit
        sl=2350.0    # stop loss
    )

    ib.disconnect()

if __name__ == "__main__":
    main()
