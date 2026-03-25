from ib_insync import *
import json

from order_utils import go_short

def main():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=4)

    with open("futuros_specs.json") as f:
        specs = json.load(f)

    spec = specs["GC"]

    short_with_bracket(
        ib,
        spec,
        quantity=1,
        tp=2300.0,
        sl=2350.0
    )

    ib.disconnect()

if __name__ == "__main__":
    main()
