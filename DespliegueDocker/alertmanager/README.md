Alertmanager: renderizar plantilla y arrancar
=========================================

Este directorio contiene una plantilla `alertmanager.yml.tmpl` que expone la
variable `${WEBHOOK_URL}` para cambiar el receptor webhook sin editar el archivo
manualmente.

Pasos rápidos
------------

1. Renderizar la plantilla localmente (usa la variable de entorno `WEBHOOK_URL` o pasa `--webhook`):

```powershell
# PowerShell (desde la raíz del repo)
$env:WEBHOOK_URL = 'https://webhook.site/e087ca8c-4460-4339-a07e-2b39de8492e0'
python .\alertmanager\render_config.py
```

o con argumento:

```powershell
python .\alertmanager\render_config.py --webhook https://webhook.site/e087ca8c-4460-4339-a07e-2b39de8492e0
```

2. Arrancar Alertmanager (Compose monta `alertmanager/alertmanager.yml` en el contenedor):

```powershell
cd DespliegueDocker
docker compose up -d alertmanager
```

Notas
-----
- No es necesario construir una imagen personalizada; el contenedor usa la
  imagen oficial `prom/alertmanager` y lee la configuración montada desde el host.
- Si cambias `WEBHOOK_URL`, vuelve a ejecutar `render_config.py` y reinicia el
  servicio (`docker compose restart alertmanager`) para aplicar el cambio.
