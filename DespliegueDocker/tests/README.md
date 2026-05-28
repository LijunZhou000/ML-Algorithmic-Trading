Pruebas para Prometheus/Pushgateway y Loki/Promtail
===============================================

Estos scripts generan métricas aleatorias enviadas al Pushgateway y ficheros de log
que `promtail` recoge y reenvía a `loki`.

Requisitos
---------

- Tener corriendo el stack de DespliegueDocker (prometheus, pushgateway, loki, promtail, grafana).
- Python 3.8+ y dependencias (ver `requirements.txt`).

Instalación rápida
------------------

Desde la raíz del proyecto:

```bash
python -m pip install -r DespliegueDocker/tests/requirements.txt
```

Arrancar los servicios (desde `DespliegueDocker`):

```bash
cd DespliegueDocker
docker compose up -d prometheus pushgateway loki promtail grafana
```

Generador de métricas (Pushgateway)
----------------------------------

Ejecuta un cliente que empuja métricas de prueba al Pushgateway:

```bash
python DespliegueDocker/tests/pushgateway_generator.py --interval 5 --pushgateway http://localhost:9091 --job test_job
```

Opciones útiles:
- `--interval` / `-i`: segundos entre pushes (por defecto 5).
- `--pushgateway` / `-u`: URL del pushgateway (por defecto `http://localhost:9091`).
- `--job` / `-j`: job grouping key en pushgateway.
- `--instance` / `-n`: instance grouping key.

Generador de logs (Loki / Promtail)
----------------------------------

Este script escribe archivos en `DespliegueDocker/tradingbot_logs/<categoria>/generator_<categoria>.log`.
Promtail en `docker-compose.yml` monta esa carpeta, por lo que verá y re-enviará las líneas a Loki.

Ejemplo:

```bash
python DespliegueDocker/tests/loki_log_generator.py --interval 2 --verbose
```

Opciones útiles:
- `--logdir` / `-d`: ruta base de logs (por defecto `DespliegueDocker/tradingbot_logs`).
- `--interval` / `-i`: segundos entre escrituras (por defecto 2).
- `--categories` / `-c`: lista de categorías a generar (por defecto: `general movements balance errors signals features`).

Comprobación rápida
-------------------

- Abrir Grafana (`http://localhost:3000`, admin/admin_password) y buscar métricas y logs.
- En Prometheus (`http://localhost:9090`) puedes consultar `test_random_gauge`.
- En Grafana Explore -> Logs seleccionar la fuente Loki y buscar `job="tradingbot"` o los labels configurados.

Notas
-----

- Si ejecutas los scripts desde dentro de contenedores en la misma red Docker, usa `http://pushgateway:9091` como URL.
- El script de logs genera JSON en la categoría `features` (para que coincida con el pipeline de `promtail-config.yml`).
