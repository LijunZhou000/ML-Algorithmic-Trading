from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

print("Posiciones abiertas:")
for p in ib.positions():
    print(p.contract.conId, p.contract.localSymbol)
