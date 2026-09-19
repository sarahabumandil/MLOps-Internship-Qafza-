"""
FastAPI inference service.

Routes:
  GET  /health         — liveness/readiness check
  GET  /model/info      — model version, threshold, feature count
  POST /predict          — score a single order
  POST /predict/batch    — score a list of orders
  GET  /metrics          — Prometheus scrape endpoint

Run locally:
  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config import get_settings
from src.data.schema import (
    BatchOrderInput,
    BatchPredictionOutput,
    OrderInput,
    PredictionOutput,
)
from src.data.validation import ValidationError
from src.logging_setup import configure_logging, log_prediction_event
from src.models.predictor import get_model, predict_batch
from src.monitoring import metrics as monitoring

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title=settings.api_title, version=settings.api_version)


@app.on_event("startup")
def _warm_up() -> None:
    """Load the model once at startup so the first request isn't slow and
    so a broken artifact fails fast, at boot, not on the first user request."""
    try:
        loaded = get_model()
        logger.info("Model warm-up complete: version=%s source=%s", loaded.version, loaded.source)
    except Exception:
        logger.exception("Model failed to load at startup")
        raise


@app.get("/health")
def health() -> dict:
    try:
        loaded = get_model()
        return {"status": "ok", "model_version": loaded.version}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc


@app.get("/model/info")
def model_info() -> dict:
    loaded = get_model()
    from src.features.transform import get_fitted_artifacts

    artifacts = get_fitted_artifacts()
    return {
        "model_version": loaded.version,
        "model_source": loaded.source,
        "threshold": loaded.threshold,
        "n_features": len(artifacts.feature_list),
        "numeric_features": artifacts.numeric_features,
        "categorical_features": artifacts.categorical_features,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(order: OrderInput, request: Request) -> PredictionOutput:
    start = time.perf_counter()
    try:
        result = predict_batch([order.model_dump()])[0]
    except ValidationError as exc:
        monitoring.record_validation_failure()
        monitoring.record_request("/predict", "validation_error", time.perf_counter() - start)
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "failed_expectations": exc.failed_expectations,
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001
        monitoring.record_request("/predict", "error", time.perf_counter() - start)
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    duration = time.perf_counter() - start
    monitoring.record_request("/predict", "ok", duration)
    monitoring.record_prediction(result["prediction"])
    monitoring.append_prediction_log({**result, "input": order.model_dump()})
    log_prediction_event(
        logger,
        request_id=result["request_id"],
        input_payload=order.model_dump(),
        prediction=result["prediction"],
        probability=result["probability_late"],
        model_version=result["model_version"],
        latency_ms=result["latency_ms"],
    )
    return PredictionOutput(**{k: v for k, v in result.items() if k in PredictionOutput.model_fields})


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch_route(batch: BatchOrderInput, request: Request) -> BatchPredictionOutput:
    start = time.perf_counter()
    try:
        results = predict_batch([o.model_dump() for o in batch.orders])
    except ValidationError as exc:
        monitoring.record_validation_failure()
        monitoring.record_request("/predict/batch", "validation_error", time.perf_counter() - start)
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "failed_expectations": exc.failed_expectations,
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001
        monitoring.record_request("/predict/batch", "error", time.perf_counter() - start)
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc

    duration = time.perf_counter() - start
    monitoring.record_request("/predict/batch", "ok", duration)
    for order, result in zip(batch.orders, results, strict=True):
        monitoring.record_prediction(result["prediction"])
        monitoring.append_prediction_log({**result, "input": order.model_dump()})

    outputs = [
        PredictionOutput(**{k: v for k, v in r.items() if k in PredictionOutput.model_fields})
        for r in results
    ]
    return BatchPredictionOutput(predictions=outputs)


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
