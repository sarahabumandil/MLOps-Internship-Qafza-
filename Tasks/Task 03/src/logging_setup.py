"""
Central logging configuration.

Call `configure_logging()` once at process start (done in app/main.py and
at the top of every CLI script). Everywhere else just does
`logger = logging.getLogger(__name__)` and logs normally — no print()
statements anywhere in src/ or app/.
"""

from __future__ import annotations

import json
import logging
import logging.config
import sys
from datetime import datetime, timezone

from src.config import get_settings


class JsonFormatter(logging.Formatter):
    """Minimal structured JSON log formatter (no external dependency)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # allow callers to pass structured fields via `extra={"extra_fields": {...}}`
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    settings = get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    formatter_key = "json" if settings.json_log_format else "plain"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            },
            "json": {
                "()": "src.logging_setup.JsonFormatter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": formatter_key,
                "level": settings.log_level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(settings.log_file),
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "formatter": formatter_key,
                "level": settings.log_level,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": settings.log_level,
        },
    }
    logging.config.dictConfig(config)


def log_prediction_event(
    logger: logging.Logger,
    *,
    request_id: str,
    input_payload: dict,
    prediction: int,
    probability: float,
    model_version: str,
    latency_ms: float,
) -> None:
    """One structured log line per prediction — input, output, latency, model version."""
    logger.info(
        "prediction_served",
        extra={
            "extra_fields": {
                "event": "prediction",
                "request_id": request_id,
                "input": input_payload,
                "prediction": prediction,
                "probability": probability,
                "model_version": model_version,
                "latency_ms": round(latency_ms, 2),
            }
        },
    )
