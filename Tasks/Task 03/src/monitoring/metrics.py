"""
Monitoring: service metrics (Prometheus) and prediction logging.

Two separate concerns:
- `metrics.py` here exposes request count / latency / error rate for
  Prometheus to scrape at `/metrics`.
- Every prediction is also appended to `prediction_log.jsonl`
  (config.paths.prediction_log) so it can be joined against the real
  delivery outcome once it's known, and so `check_drift()` has something
  to compare the training distribution against.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

from prometheus_client import Counter, Histogram

from src.config import get_settings

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["route", "status"])
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Prediction request latency in seconds", ["route"])
PREDICTION_LATE_RATE = Counter("prediction_late_total", "Count of predictions by class", ["predicted_class"])
VALIDATION_FAILURES = Counter("order_validation_failures_total", "Count of orders that failed validation")


def record_request(route: str, status: str, duration_seconds: float) -> None:
    REQUEST_COUNT.labels(route=route, status=status).inc()
    REQUEST_LATENCY.labels(route=route).observe(duration_seconds)


def record_prediction(predicted_class: int) -> None:
    PREDICTION_LATE_RATE.labels(predicted_class=str(predicted_class)).inc()


def record_validation_failure() -> None:
    VALIDATION_FAILURES.inc()


def append_prediction_log(record: dict) -> None:
    settings = get_settings()
    path: Path = settings.prediction_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "logged_at": time.time()}
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_recent_predictions(n: int = 1000) -> list[dict]:
    settings = get_settings()
    path = settings.prediction_log_path
    if not path.exists():
        return []
    with open(path) as f:
        lines = f.readlines()[-n:]
    return [json.loads(line) for line in lines]


def check_drift(reference_late_rate: float | None = None) -> dict:
    """Compare the recent predicted-late rate to the training-set late rate.

    This is a simple population-stability style check on the model's own
    output distribution — enough to catch a model that's suddenly
    predicting "late" far more or less often than it did on training data,
    which is usually the first visible symptom of input drift.
    """
    settings = get_settings()
    records = read_recent_predictions(n=5000)

    if len(records) < settings.drift_check_min_predictions:
        return {
            "status": "insufficient_data",
            "n_predictions": len(records),
            "min_required": settings.drift_check_min_predictions,
        }

    recent_rate = statistics.mean(r["prediction"] for r in records)

    if reference_late_rate is None:
        try:
            results_path = settings.results_summary_path
            with open(results_path) as f:
                results = json.load(f)
            # val_pr_auc etc. don't give us the rate directly; use test set
            # late rate saved alongside training if present, else skip.
            reference_late_rate = results.get("train_late_rate")
        except FileNotFoundError:
            reference_late_rate = None

    drift_flag = None
    if reference_late_rate is not None:
        delta = abs(recent_rate - reference_late_rate)
        drift_flag = delta > 0.05  # more than 5 points of absolute drift

    return {
        "status": "ok",
        "n_predictions": len(records),
        "recent_predicted_late_rate": recent_rate,
        "reference_late_rate": reference_late_rate,
        "drift_detected": drift_flag,
    }
