FROM prom/alertmanager:latest

USER root

# Install envsubst (gettext) to allow substitución de variables en la plantilla
RUN apt-get update \
    && apt-get install -y --no-install-recommends gettext-base ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copiamos el entrypoint (la plantilla la montamos desde el host en docker-compose)
COPY entrypoint.sh /bin/alertmanager-entrypoint.sh
RUN chmod +x /bin/alertmanager-entrypoint.sh

ENTRYPOINT ["/bin/alertmanager-entrypoint.sh"]
