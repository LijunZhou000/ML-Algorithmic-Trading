#!/usr/bin/env python3
"""
Push the average of the last N rows for a metric from Postgres to Pushgateway.

Designed to run as a small containerized job that periodically reads the
`demo_metrics` table and pushes a Prometheus gauge with the average value.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from typing import Optional

try:
    import psycopg2
except Exception:
    raise

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


stop = False


def handle_signal(sig, frame):
    global stop
    stop = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def get_avg(conn, metric_name: str, limit: int) -> Optional[float]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT AVG(value) FROM (
                SELECT value FROM demo_metrics
                WHERE metric_name = %s
                ORDER BY ts DESC
                LIMIT %s
            ) t
            """,
            (metric_name, limit),
        )
        row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def main() -> None:
    p = argparse.ArgumentParser(description="Push average of last N rows to Pushgateway")
    p.add_argument("--pg-host", default=os.getenv("PGHOST", "postgres"))
    p.add_argument("--pg-port", type=int, default=int(os.getenv("PGPORT", 5432)))
    p.add_argument("--pg-db", default=os.getenv("PGDATABASE", "airflow"))
    p.add_argument("--pg-user", default=os.getenv("PGUSER", "airflow"))
    p.add_argument("--pg-pass", default=os.getenv("PGPASSWORD", "airflow"))
    p.add_argument("--metric", default=os.getenv("METRIC_NAME", "demo_metric"))
    p.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", 10)))
    p.add_argument("--interval", type=int, default=int(os.getenv("INTERVAL", 30))) # seconds
    p.add_argument("--pushgateway", default=os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091"))
    p.add_argument("--job", default=os.getenv("JOB_NAME", "sql_avg_job"))
    args = p.parse_args()

    logging.info("Starting SQL avg pusher: metric=%s limit=%d interval=%ds pushgateway=%s",
                 args.metric, args.limit, args.interval, args.pushgateway)

    while not stop:
        try:
            conn = psycopg2.connect(
                host=args.pg_host,
                port=args.pg_port,
                dbname=args.pg_db,
                user=args.pg_user,
                password=args.pg_pass,
            )
            try:
                avg = get_avg(conn, args.metric, args.limit)
            finally:
                conn.close()

            if avg is None:
                logging.info("No rows found for metric=%s", args.metric)
            else:
                registry = CollectorRegistry()
                g = Gauge(
                    "demo_metrics_avg_last_n",
                    "Average of last N rows from demo_metrics",
                    ["metric_name"],
                    registry=registry,
                )
                g.labels(metric_name=args.metric).set(avg)
                # push to Pushgateway
                push_to_gateway(args.pushgateway, job=args.job, registry=registry)
                logging.info("Pushed avg=%.6f for metric=%s to %s", avg, args.metric, args.pushgateway)

        except Exception:
            logging.exception("Error querying or pushing metric")

        # sleep respecting stop flag
        for _ in range(max(1, args.interval)):
            if stop:
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
