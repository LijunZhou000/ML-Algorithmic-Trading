#!/usr/bin/env python3
"""
Generador de métricas que escribe en una tabla Postgres (demo_metrics).

Ejemplo:
  python tests/sql_writer.py --interval 5 --host postgres --db airflow --user airflow --password airflow --metric demo_metric

Si Postgres corre en Docker Compose use `--host postgres` (nombre del servicio) y execute
el script desde un contenedor conectado a la red `mlnet` o usando `docker exec`.
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

try:
    import psycopg2
    from psycopg2.extras import Json
except Exception as e:
    print("psycopg2 is required. Install with: pip install psycopg2-binary", file=sys.stderr)
    raise


def iso_now():
    return datetime.utcnow().isoformat() + "Z"


def ensure_table(conn):
    ddl = """
    CREATE TABLE IF NOT EXISTS demo_metrics (
        id SERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL DEFAULT now(),
        metric_name TEXT NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        tags JSONB DEFAULT '{}'::jsonb
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Insert random metrics into Postgres demo table")
    parser.add_argument("--host", default=os.getenv("PGHOST", "postgres"), help="DB host (container name in compose: 'postgres')")
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", 5432)), help="DB port")
    parser.add_argument("--db", default=os.getenv("PGDATABASE", "airflow"), help="Database name")
    parser.add_argument("--user", default=os.getenv("PGUSER", "airflow"), help="DB user")
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "airflow"), help="DB password")
    parser.add_argument("--interval", "-i", type=float, default=5.0, help="Intervalo en segundos entre inserciones")
    parser.add_argument("--metric", "-m", default="demo_metric", help="Nombre de la métrica")
    parser.add_argument("--ensure-table", action="store_true", help="Crear la tabla demo si no existe")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    stop = False

    def handle_signals(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signals)
    signal.signal(signal.SIGTERM, handle_signals)

    conn = None
    try:
        conn = psycopg2.connect(host=args.host, port=args.port, dbname=args.db, user=args.user, password=args.password)
        conn.autocommit = False
    except Exception as e:
        print(f"Error connecting to Postgres at {args.host}:{args.port} - {e}", file=sys.stderr)
        sys.exit(2)

    if args.ensure_table:
        try:
            ensure_table(conn)
            if args.verbose:
                print(f"[{iso_now()}] Tabla demo_metrics garantizada en {args.db}")
        except Exception as e:
            print(f"Error creando la tabla: {e}", file=sys.stderr)
            sys.exit(3)

    counter = 0
    while not stop:
        value = round(random.random() * 100.0, 3)
        counter += random.randint(0, 5)
        tags = {"instance": os.getenv("INSTANCE", f"instance-{os.getpid()}"), "source": "sql_writer"}

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO demo_metrics (metric_name, value, tags) VALUES (%s, %s, %s)",
                    (args.metric, value, Json(tags)),
                )
            conn.commit()
            if args.verbose:
                print(f"[{iso_now()}] Inserted metric={args.metric} value={value} tags={json.dumps(tags)}")
        except Exception as e:
            print(f"[{iso_now()}] Error inserting into DB: {e}", file=sys.stderr)
            try:
                conn.rollback()
            except Exception:
                pass

        slept = 0.0
        while slept < args.interval and not stop:
            time.sleep(0.2)
            slept += 0.2

    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
