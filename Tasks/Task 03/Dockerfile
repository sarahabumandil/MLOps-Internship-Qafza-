# Small image, multi-stage: build deps in one layer, copy only what the
# service needs into the final image. No notebooks, no dev/test deps.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# System deps needed to build psycopg2 / pyarrow wheels stay minimal;
# most deps ship manylinux wheels so this stays small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- runtime image -----------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_CONFIG_PATH=/srv/config/config.yaml

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /srv

COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Only what the running service needs — no notebooks/, no tests/, no scripts
# other than the one used to bootstrap artifacts if the volume is empty.
COPY app ./app
COPY src ./src
COPY config ./config
COPY scripts/bootstrap_train.py ./scripts/bootstrap_train.py
COPY scripts/make_synthetic_ml_table.py ./scripts/make_synthetic_ml_table.py
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh

RUN mkdir -p data/interim data/processed data/models logs data/monitoring reports/figures \
    && chown -R appuser:appuser /srv \
    && chmod +x scripts/entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else sys.exit(1)"

ENTRYPOINT ["./scripts/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
