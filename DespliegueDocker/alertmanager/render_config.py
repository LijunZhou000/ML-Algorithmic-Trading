#!/usr/bin/env python3
"""
Renderiza `alertmanager.yml.tmpl` sustituyendo la variable ${WEBHOOK_URL}
y escribe `alertmanager.yml` listo para montar en el contenedor.

Uso:
  # con env var
  WEBHOOK_URL=https://webhook.site/... python alertmanager/render_config.py

  # o pasando la url por argumento
  python alertmanager/render_config.py --webhook https://webhook.site/...

El script no necesita dependencias externas (usa stdlib).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from string import Template


TEMPLATE = Path(__file__).parent / "alertmanager.yml.tmpl"
OUT = Path(__file__).parent / "alertmanager.yml"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--webhook", "-w", help="Webhook URL (overrides WEBHOOK_URL env var)")
    args = p.parse_args()

    webhook = args.webhook or os.getenv("WEBHOOK_URL")
    if not webhook:
        print("No WEBHOOK_URL provided via --webhook or env; using webhook.site default")
        webhook = "https://webhook.site/e087ca8c-4460-4339-a07e-2b39de8492e0"

    if not TEMPLATE.exists():
        print(f"Template not found: {TEMPLATE}")
        return 2

    txt = TEMPLATE.read_text(encoding="utf-8")
    rendered = Template(txt).safe_substitute(WEBHOOK_URL=webhook)
    OUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUT} (WEBHOOK_URL={webhook})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
