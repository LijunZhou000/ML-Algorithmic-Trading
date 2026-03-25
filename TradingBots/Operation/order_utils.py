from ib_insync import *

def get_future_contract(spec):
    """
    Crea un contrato de futuro de IB a partir del JSON.
    """
    contract = Future(
        symbol=spec["ib_symbol"],
        lastTradeDateOrContractMonth="",   # vacío → usa conId
        conId=spec["conId"],
        exchange=spec["exchange"],
        currency=spec["currency"],
        multiplier=str(int(spec["multiplier"]))
    )
    return contract

def place_order(ib, contract, action, quantity=1):
    """
    Envía una orden de mercado LONG o SHORT.
    action = 'BUY' o 'SELL'
    """
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)  # deja tiempo para que TWS procese
    print(f"Orden enviada: {action} {quantity} contratos de {contract.symbol}")
    return trade

def go_long(ib, spec, quantity=1):
    contract = get_future_contract(spec)
    return place_order(ib, contract, "BUY", quantity)

def go_short(ib, spec, quantity=1):
    contract = get_future_contract(spec)
    return place_order(ib, contract, "SELL", quantity)

def create_bracket_order(action, quantity, take_profit_price, stop_loss_price):
    """
    Crea una bracket order (orden principal + TP + SL).
    action = 'BUY' o 'SELL'
    """
    # Orden principal
    parent = MarketOrder(action, quantity)
    parent.orderId = None  # IB asignará ID

    # Take Profit
    tp_action = "SELL" if action == "BUY" else "BUY"
    take_profit = LimitOrder(tp_action, quantity, take_profit_price)
    take_profit.parentId = parent.orderId

    # Stop Loss
    stop_loss = StopOrder(tp_action, quantity, stop_loss_price)
    stop_loss.parentId = parent.orderId

    return parent, take_profit, stop_loss

def place_bracket(ib, contract, action, quantity, tp, sl):
    parent, take_profit, stop_loss = create_bracket_order(action, quantity, tp, sl)

    # Enviar orden principal
    trade = ib.placeOrder(contract, parent)
    ib.sleep(1)

    # IB asigna orderId → ahora podemos enviar TP y SL
    take_profit.parentId = parent.orderId
    stop_loss.parentId = parent.orderId

    ib.placeOrder(contract, take_profit)
    ib.placeOrder(contract, stop_loss)

    print(f"Bracket enviada: {action} {quantity} | TP={tp} | SL={sl}")
    return trade

def long_with_bracket(ib, spec, quantity, tp, sl):
    contract = get_future_contract(spec)
    return place_bracket(ib, contract, "BUY", quantity, tp, sl)

def short_with_bracket(ib, spec, quantity, tp, sl):
    contract = get_future_contract(spec)
    return place_bracket(ib, contract, "SELL", quantity, tp, sl)

def update_bracket_orders(ib, new_tp=None, new_sl=None):
    """
    Actualiza TP y/o SL de las órdenes abiertas.
    new_tp: nuevo precio límite (take profit)
    new_sl: nuevo precio stop (stop loss)
    """
    open_orders = ib.openOrders()

    tp_order = None
    sl_order = None

    # Identificar órdenes TP y SL
    for o in open_orders:
        if o.order.orderType == "LMT":   # Take Profit
            tp_order = o
        elif o.order.orderType == "STP":  # Stop Loss
            sl_order = o

    if tp_order is None and sl_order is None:
        print("No hay órdenes TP/SL abiertas para actualizar.")
        return

    # Actualizar TP
    if new_tp is not None and tp_order is not None:
        tp_order.order.lmtPrice = new_tp
        ib.placeOrder(tp_order.contract, tp_order.order)
        print(f"TP actualizado a {new_tp}")

    # Actualizar SL
    if new_sl is not None and sl_order is not None:
        sl_order.order.auxPrice = new_sl
        ib.placeOrder(sl_order.contract, sl_order.order)
        print(f"SL actualizado a {new_sl}")

    ib.sleep(1)
    
def add_sl_tp_to_existing_position(ib, contract, tp_price, sl_price, quantity):
    """
    Añade TP y SL a una posición ya abierta (sin bracket original).
    Crea dos órdenes independientes: LIMIT (TP) y STOP (SL).
    """
    # Take Profit
    tp_order = LimitOrder("BUY" if quantity < 0 else "SELL", abs(quantity), tp_price)
    ib.placeOrder(contract, tp_order)

    # Stop Loss
    sl_order = StopOrder("BUY" if quantity < 0 else "SELL", abs(quantity), sl_price)
    ib.placeOrder(contract, sl_order)

    ib.sleep(1)
    print(f"TP y SL añadidos: TP={tp_price}, SL={sl_price}")
    
def rebuild_contract_from_position(ib, pos):
    """
    Reconstruye un contrato completo usando el conId de la posición abierta.
    Esto evita el error 321 (falta el mercado).
    """
    c = Contract()
    c.conId = pos.contract.conId
    c.exchange = pos.contract.exchange or "COMEX"  # o CME según el activo
    c.secType = "FUT"
    c.currency = pos.contract.currency or "USD"

    # Pedimos a IB que complete el contrato
    ib.qualifyContracts(c)
    return c
