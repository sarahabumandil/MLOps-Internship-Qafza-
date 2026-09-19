"""
Model loading and prediction.

Resolution order for the model, per the task sheet ("the service loads the
model from the registry or the artifact store, not from a local notebook
folder"):

1. MLflow Model Registry, at `mlflow.registered_model_name` /
   `mlflow.model_stage` — used when `MLFLOW_TRACKING_URI` is reachable.
2. Local `data/models/model.joblib` — the artifact Notebook 6 (or
   scripts/bootstrap_train.py) wrote, used for offline/dev/test runs and
   as a fallback if the registry is unreachable.

Either way, the service only ever calls `.predict_proba()` — never fits.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache

import joblib
import pandas as pd

from src.config import get_settings
from src.data import validation as validation_mod
from src.data.preprocessing import build_raw_features
from src.features.transform import get_fitted_artifacts, transform

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: object
    version: str
    threshold: float
    source: str  # "mlflow_registry" or "local_joblib"


def _load_threshold() -> float:
    settings = get_settings()
    try:
        with open(settings.results_summary_path) as f:
            results = json.load(f)
        return float(results.get("chosen_threshold", settings.default_threshold))
    except FileNotFoundError:
        logger.warning("results_summary.json not found, using default threshold from config")
        return settings.default_threshold


def _mlflow_server_reachable(tracking_uri: str, timeout: float = 1.0) -> bool:
    """Fast pre-check so a missing tracking server (e.g. running the API
    outside docker-compose) fails over to the local artifact in well under
    a second instead of through mlflow's multi-minute HTTP retry storm."""
    import socket
    from urllib.parse import urlparse

    if not tracking_uri.startswith("http"):
        return True
    parsed = urlparse(tracking_uri)
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _load_from_mlflow() -> LoadedModel | None:
    settings = get_settings()
    if not _mlflow_server_reachable(settings.mlflow_tracking_uri):
        logger.info(
            "MLflow tracking server at %s unreachable, using local artifact store",
            settings.mlflow_tracking_uri,
        )
        return None
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{settings.mlflow_registered_model_name}/{settings.mlflow_model_stage}"
        model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        versions = client.get_latest_versions(
            settings.mlflow_registered_model_name, stages=[settings.mlflow_model_stage]
        )
        version = versions[0].version if versions else "unknown"

        logger.info("Loaded model from MLflow registry: %s (version=%s)", model_uri, version)
        return LoadedModel(
            model=model,
            version=f"mlflow:{settings.mlflow_registered_model_name}:{version}",
            threshold=_load_threshold(),
            source="mlflow_registry",
        )
    except Exception as exc:  # noqa: BLE001 — any registry failure falls back to local
        logger.warning("Could not load model from MLflow registry (%s); falling back to local artifact", exc)
        return None


def _load_from_local() -> LoadedModel:
    settings = get_settings()
    model = joblib.load(settings.model_path)
    logger.info("Loaded model from local artifact store: %s", settings.model_path)
    return LoadedModel(
        model=model,
        version=f"local:{settings.model_path.name}",
        threshold=_load_threshold(),
        source="local_joblib",
    )


@lru_cache(maxsize=1)
def get_model() -> LoadedModel:
    return _load_from_mlflow() or _load_from_local()


def predict_one(order: dict) -> dict:
    return predict_batch([order])[0]


def predict_batch(orders: list[dict]) -> list[dict]:
    """Full pipeline: validate -> build features -> transform -> predict.

    Each order dict is validated with Great Expectations, features are
    engineered exactly as Notebook 5 did, the fitted transformers are
    applied (never re-fit), and the loaded model scores the batch.
    """
    if not orders:
        return []

    start = time.perf_counter()
    loaded = get_model()

    df = pd.DataFrame(orders)
    df = validation_mod.enforce_validation_policy(df)

    raw_features = build_raw_features(df)
    artifacts = get_fitted_artifacts()
    X = transform(raw_features, artifacts)

    probabilities = loaded.model.predict_proba(X)[:, 1]
    predictions = (probabilities >= loaded.threshold).astype(int)

    latency_ms = (time.perf_counter() - start) * 1000 / max(len(orders), 1)

    results = []
    for pred, prob in zip(predictions, probabilities, strict=True):
        results.append(
            {
                "prediction": int(pred),
                "probability_late": float(prob),
                "threshold_used": loaded.threshold,
                "model_version": loaded.version,
                "request_id": str(uuid.uuid4()),
                "latency_ms": latency_ms,
            }
        )
    return results
