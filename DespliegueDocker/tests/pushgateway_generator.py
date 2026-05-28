#!/usr/bin/env python3
"""
Generador simple de métricas hacia Pushgateway para pruebas.

Ejemplo:
  python pushgateway_generator.py --interval 5 --pushgateway http://localhost:9091 --job test_job

Variables de entorno:
  PUSHGATEWAY_URL  URL del pushgateway (ej: http://localhost:9091)
"""
import argparse
import os
import random
import signal
import sys
import time
from datetime import datetime

import requests


def iso_now():
    return datetime.utcnow().isoformat() + "Z"


def main():
    parser = argparse.ArgumentParser(description="Push random metrics to Pushgateway periodically")
    parser.add_argument("--interval", "-i", type=float, default=5.0, help="Intervalo en segundos entre envíos")
    parser.add_argument("--pushgateway", "-u", default=os.getenv("PUSHGATEWAY_URL", "http://localhost:9091"), help="URL del Pushgateway")
    parser.add_argument("--job", "-j", default="test_job", help="Grouping job name en Pushgateway")
    parser.add_argument("--instance", "-n", default=os.getenv("INSTANCE", f"instance-{os.getpid()}"), help="Grouping instance label")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    stop = False

    def handle_signals(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signals)
    signal.signal(signal.SIGTERM, handle_signals)

    counter = 0
    while not stop:
        gauge = random.random() * 100.0
        counter += random.randint(0, 3)
        payload_lines = []
        payload_lines.append(f"# HELP test_random_gauge Valor aleatorio de prueba")
        payload_lines.append(f"# TYPE test_random_gauge gauge")
        payload_lines.append(f"test_random_gauge {gauge}")
        payload_lines.append(f"# HELP test_random_counter Contador de prueba")
        payload_lines.append(f"# TYPE test_random_counter counter")
        payload_lines.append(f"test_random_counter {counter}")
        payload_lines.append(f"# HELP test_timestamp Timestamp de envío")
        payload_lines.append(f"# TYPE test_timestamp gauge")
        payload_lines.append(f"test_timestamp {time.time()}")
        payload = "\n".join(payload_lines) + "\n"

        url = f"{args.pushgateway.rstrip('/')}/metrics/job/{args.job}/instance/{args.instance}"
        try:
            resp = requests.put(url, data=payload, headers={"Content-Type": "text/plain; charset=utf-8"}, timeout=5)
            resp.raise_for_status()
            if args.verbose:
                print(f"[{iso_now()}] Pushed metrics to {url}: gauge={gauge:.2f} counter={counter}")
        except Exception as e:
            print(f"[{iso_now()}] Error pushing to Pushgateway {url}: {e}", file=sys.stderr)

        # Espera intervalo (puede terminar con Ctrl-C)
        slept = 0.0
        while slept < args.interval and not stop:
            time.sleep(0.2)
            slept += 0.2


if __name__ == "__main__":
    main()
