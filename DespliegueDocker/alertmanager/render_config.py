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
ENV_FILE = Path(__file__).parent.parent / ".env"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--webhook", "-w", help="Webhook URL (overrides WEBHOOK_URL env var)")
    args = p.parse_args()

    # Load .env (if present) so users can set variables there and run this script
    def load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # don't overwrite already set env vars
            if k and k not in os.environ:
                os.environ[k] = v

    load_dotenv(ENV_FILE)

    webhook = args.webhook or os.getenv("WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        print("No WEBHOOK_URL/SLACK_WEBHOOK_URL provided via --webhook or env; using webhook.site default")
        webhook = "https://webhook.site/e087ca8c-4460-4339-a07e-2b39de8492e0"

    if not TEMPLATE.exists():
        print(f"Template not found: {TEMPLATE}")
        return 2

    # Prepare substitutions including Slack-related variables (defaults if missing)
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL") or webhook
    slack_channel = os.getenv("SLACK_CHANNEL") or "#varios"
    slack_default_text = os.getenv("SLACK_DEFAULT_TEXT") or ""

    txt = TEMPLATE.read_text(encoding="utf-8")
    rendered = Template(txt).safe_substitute(
        WEBHOOK_URL=webhook,
        SLACK_WEBHOOK_URL=slack_webhook,
        SLACK_CHANNEL=slack_channel,
        SLACK_DEFAULT_TEXT=slack_default_text,
    )
    OUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUT} (WEBHOOK_URL={webhook})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
