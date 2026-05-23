# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy project files and install dependencies into a portable prefix
COPY pyproject.toml README.md ./
COPY app/ ./app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install .


# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime tooling used by healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy installed packages and console scripts into runtime
COPY --from=builder /install /usr/local

# Defensive install and verification to guarantee runtime imports and entrypoints
RUN pip install --no-cache-dir packaging "uvicorn[standard]==0.24.0" && \
    python -c "import packaging, uvicorn; print('packaging and uvicorn available')"

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]