FROM python:3.12.9-slim

ENV PYTHONUNBUFFERED=1

# Dependencias del sistema para compilar paquetes (y TA-Lib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl git ca-certificates \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev pkg-config \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Compilar e instalar TA-Lib (necesario si aparece en requirements)
RUN wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz && \
    tar -xzf /tmp/ta-lib.tar.gz -C /tmp && \
    TADIR=$(tar -tf /tmp/ta-lib.tar.gz | head -1 | cut -f1 -d'/') && \
    cd /tmp/$TADIR && ./configure --prefix=/usr && make && make install && \
    rm -rf /tmp/ta-lib*

WORKDIR /app

# Copy requirements (ruta en el contexto de build: ejecuta el build desde la raíz del repo)
COPY DespliegueDocker/tradingbot/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copiar e instalar el paquete local tradepy en modo editable
COPY TradingBots/tradepy /app/TradingBots/tradepy
RUN pip install --no-cache-dir -e /app/TradingBots/tradepy

# Ajustes finales
WORKDIR /app/TradingBots
CMD ["/bin/bash"]