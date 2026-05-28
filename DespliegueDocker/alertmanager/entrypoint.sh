#!/bin/sh
set -e

# Entrada: /etc/alertmanager/alertmanager.yml.tmpl (plantilla)
# Salida: /etc/alertmanager/alertmanager.yml (config final)

: "${WEBHOOK_URL:=https://webhook.site/your-uuid-here}"

TEMPLATE=/etc/alertmanager/alertmanager.yml.tmpl
OUT=/etc/alertmanager/alertmanager.yml

if [ -f "$TEMPLATE" ]; then
  echo "[entrypoint] Rendering Alertmanager config from template"
  envsubst < "$TEMPLATE" > "$OUT"
else
  echo "[entrypoint] Template $TEMPLATE not found, expecting existing config at $OUT"
fi

echo "[entrypoint] Starting Alertmanager with config $OUT"
exec /bin/alertmanager --config.file="$OUT"
