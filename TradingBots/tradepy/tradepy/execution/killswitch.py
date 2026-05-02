
import argparse
import asyncio
import logging
from typing import List, Tuple

from ib_async import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


async def flat_all_futures(ib, execute: bool = False, cancel_orders: bool = False, only_symbols: List[str] = None) -> List[Tuple[str, int, str, int, str]]:
	"""Cierra (flat) todas las posiciones de tipo FUT en la cuenta.
	- `execute=False` hace dry-run (no envía órdenes).
	- `cancel_orders=True` intentará cancelar órdenes activas para cada contrato antes del cierre.
	- `only_symbols`: lista opcional de símbolos (ej. ['GC','SI']) para filtrar.

	Devuelve una lista de tuplas: (localSymbol, conId, action, qty, status)
	"""
	results = []
	positions = ib.positions()
	if not positions:
		logger.info("No positions returned by IB.")
		return results

	open_orders = []
	if cancel_orders:
		try:
			open_orders = await ib.reqAllOpenOrdersAsync()
		except Exception as e:
			logger.warning(f"Could not fetch open orders: {e}")
			open_orders = []

	for p in positions:
		contract = p.contract
		sec_type = getattr(contract, 'secType', '') or getattr(contract, 'secType', '')
		if str(sec_type).upper() != 'FUT':
			continue

		local_symbol = getattr(contract, 'localSymbol', getattr(contract, 'symbol', None))
		if only_symbols and local_symbol:
			if not any(s.upper() in str(local_symbol).upper() for s in only_symbols):
				continue

		qty = int(p.position) if p.position else 0
		if qty == 0:
			continue

		action = 'SELL' if qty > 0 else 'BUY'
		qty_abs = abs(qty)

		if cancel_orders and open_orders:
			for t in open_orders:
				try:
					if t.contract.conId == contract.conId:
						ib.cancelOrder(t.order)
						logger.info(f"Cancelled order {t.order.orderId} for {local_symbol}")
				except Exception as e:
					logger.warning(f"Error cancelling order for {local_symbol}: {e}")

		if not execute:
			logger.info(f"[DRY-RUN] {action} {qty_abs} -> {local_symbol} (conId={contract.conId})")
			results.append((local_symbol, contract.conId, action, qty_abs, 'DRY-RUN'))
			continue

		try:
			order = MarketOrder(action, qty_abs, tif='DAY')
			ib.placeOrder(contract, order)
			logger.info(f"Placed MarketOrder {action} {qty_abs} on {local_symbol} (conId={contract.conId})")
			results.append((local_symbol, contract.conId, action, qty_abs, 'SENT'))
		except Exception as e:
			logger.exception(f"Failed to place order for {local_symbol}: {e}")
			results.append((local_symbol, contract.conId, action, qty_abs, f'ERROR: {e}'))

		await asyncio.sleep(0.2)

	return results


async def main():
	parser = argparse.ArgumentParser(description='Killswitch: flat total de futuros (dry-run por defecto).')
	parser.add_argument('--host', default='127.0.0.1', help='IB host (default 127.0.0.1)')
	parser.add_argument('--port', type=int, default=7497, help='IB port (default 7497)')
	parser.add_argument('--client-id', type=int, default=1, help='IB clientId')
	parser.add_argument('--execute', action='store_true', help='Enviar órdenes realmente (por defecto hace dry-run)')
	parser.add_argument('--cancel-orders', action='store_true', help='Cancelar órdenes activas para cada contrato antes del flat')
	parser.add_argument('--only-symbols', nargs='+', help='Lista de símbolos para filtrar (ej: GC SI)')
	args = parser.parse_args()

	ib = IB()
	await ib.connectAsync(args.host, args.port, clientId=args.client_id, readonly=False)
	logger.info(f"Connected to IB {args.host}:{args.port} (clientId={args.client_id})")

	try:
		if not args.execute:
			logger.info('Running in dry-run mode. Use --execute to place real orders.')

		results = await flat_all_futures(ib, execute=args.execute, cancel_orders=args.cancel_orders, only_symbols=args.only_symbols)

		if not results:
			logger.info('No FUT positions found to close.')
		else:
			logger.info('Summary:')
			for r in results:
				logger.info(f"{r[0]} (conId={r[1]}): {r[2]} {r[3]} -> {r[4]}")

	finally:
		try:
			await ib.disconnectAsync()
		except Exception:
			pass


if __name__ == '__main__':
	asyncio.run(main())

