#!/usr/bin/env python3
"""
Generador de logs aleatorios para que promtail los recoja y los envie a Loki.

Escribe archivos en la carpeta `DespliegueDocker/tradingbot_logs/*/*.log` que es la
ruta que monta promtail en la configuración provista.

Ejemplo:
  python loki_log_generator.py --interval 2

Opciones:
  --logdir Ruta base de logs (por defecto: ../tradingbot_logs relativa a DespliegueDocker)
  --interval Intervalo en segundos entre líneas escritas
  --verbose Imprime líneas escritas en consola
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_CATEGORIES = [
    "general",
    "movements",
    "balance",
    "errors",
    "signals",
    "features",
]


def iso_now():
    return datetime.utcnow().isoformat() + "Z"


def ensure_dirs(base: Path, categories):
    base = Path(base)
    for c in categories:
        d = base / c
        d.mkdir(parents=True, exist_ok=True)


def write_line(path: Path, line: str):
    # Append and force flush so promtail can pick it up quickly
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # fsync might fail on some platforms/FS; ignore
            pass


def make_text_line(category: str) -> str:
    level = random.choice(["INFO", "WARN", "ERROR", "DEBUG"])
    messages = [
        "Operación completada",
        "Entrada de posición detectada",
        "Salida de posición detectada",
        "Balance actualizado",
        "Error al comunicarse con broker",
        "Señal generada por el modelo",
    ]
    msg = random.choice(messages)
    return f"{iso_now()} [{level}] ({category}) {msg} id={random.randint(1,99999)}"


def make_json_line() -> str:
    level = random.choice(["info", "warn", "error", "debug"])
    tickers = ["GC", "ES", "CL", "NG", "BP"]
    direction = random.choice(["long", "short"])
    payload = {
        "ts": iso_now(),
        "level": level,
        "ticker": random.choice(tickers),
        "direction": direction,
        "value": round(random.random() * 100, 3),
        "msg": "Feature pipeline event",
    }
    return json.dumps(payload, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Generador de logs para promtail/Loki")
    parser.add_argument("--logdir", "-d", default=str(Path(__file__).resolve().parent.parent / "tradingbot_logs"), help="Directorio base donde crear archivos de log")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Intervalo en segundos entre escrituras")
    parser.add_argument("--categories", "-c", nargs="*", default=DEFAULT_CATEGORIES, help="Categorias a generar (por defecto todas)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    base = Path(args.logdir)
    ensure_dirs(base, args.categories)

    stop = False

    def handler(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Mantener un archivo por categoría
    files = {c: base / c / f"generator_{c}.log" for c in args.categories}

    if args.verbose:
        print(f"Generando logs en: {base.resolve()}")
        for c, p in files.items():
            print(f" - {c}: {p}")

    while not stop:
        for c in args.categories:
            try:
                if c == "features":
                    line = make_json_line()
                elif c == "errors":
                    line = f"{iso_now()} [ERROR] (errors) Excepción simulada id={random.randint(1000,9999)}"
                else:
                    line = make_text_line(c)

                write_line(files[c], line)
                if args.verbose:
                    print(f"WROTE {c}: {line}")
            except Exception as e:
                print(f"Error escribiendo en {c}: {e}", file=sys.stderr)

        slept = 0.0
        while slept < args.interval and not stop:
            time.sleep(0.2)
            slept += 0.2


if __name__ == "__main__":
    main()
