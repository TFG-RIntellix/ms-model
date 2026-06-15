# =========================
# Stage 1: Builder
# =========================
FROM python:3.12.3-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
    g++ gcc make

    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app/ ./app/

# Instala dependencias y paquete usando pyproject.toml
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --prefix=/install .

# =========================
# Stage 2: Runtime
# =========================
FROM python:3.12.3-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 appuser

# Dependencias instaladas desde pyproject.toml
COPY --from=builder /install /usr/local

# Copiamos el código fuente completo, incluidos ml_artifacts
COPY --chown=appuser:appuser app/ ./app/

USER appuser

HEALTHCHECK --interval=30s \
	--timeout=10s \
	--start-period=5s \
	--retries=3 \
	CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" \
	|| exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
