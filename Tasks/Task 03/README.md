# Olist Late-Delivery — Inference Service (Task 3)

Turns the Task 2 notebooks (`sarahabumandil/MLOps-Internship-Qafza-`,
`Tasks/Task 02`) into a real inference service: scripts, tests, an API,
containers, and CI/CD. Training itself still lives in the notebooks —
this repo only loads what they produced and serves predictions.

```
.
├── app/                    # FastAPI service
│   └── main.py
├── config/
│   └── config.yaml          # every path & parameter, nothing hardcoded elsewhere
├── src/
│   ├── config.py             # loads config.yaml, resolves paths & env vars
│   ├── logging_setup.py       # structured logging, console + file
│   ├── data/
│   │   ├── schema.py           # OrderInput / PredictionOutput pydantic models
│   │   ├── preprocessing.py    # Notebook 5's build_raw_features, unchanged logic
│   │   └── validation.py       # Great Expectations order_intake_suite
│   ├── features/
│   │   └── transform.py         # loads & applies fitted_transformers.joblib (never fits)
│   ├── models/
│   │   └── predictor.py          # loads model (MLflow registry -> local joblib), predicts
│   └── monitoring/
│       └── metrics.py             # Prometheus metrics + prediction log for drift checks
├── scripts/
│   ├── make_synthetic_ml_table.py  # stands in for Notebook 1 (see "About the data" below)
│   ├── bootstrap_train.py           # runs Notebooks 2/3/5/6 logic, unmodified, as a script
│   └── entrypoint.sh
├── tests/
│   ├── unit/                # preprocessing, transform, predictor
│   ├── data/                 # schema, GE suite, leakage watch-list
│   └── integration/           # FastAPI routes end to end
├── great_expectations/       # suite defined as code in src/data/validation.py — see its README
├── dvc.yaml, dvc.lock         # data/artifact versioning pipeline
├── Dockerfile, docker-compose.yml
├── .github/workflows/ci.yml   # lint -> test -> build & push
└── .pre-commit-config.yaml
```

## About the data

The uploaded Task 2 repo has the 6 notebooks but an **empty `data/`
folder** — no Olist database, no saved artifacts. Task 3 assumes those
already exist. So `scripts/make_synthetic_ml_table.py` generates a
schema-accurate synthetic dataset standing in for "Notebook 1 run against
the real Olist DB", and `scripts/bootstrap_train.py` runs Notebooks
2/3/5/6's actual logic — label creation, the time-based split, feature
engineering, RandomForest tuning — **unmodified** against it, to produce
real, working artifacts:

- `data/models/fitted_transformers.joblib`
- `data/models/feature_list.json`
- `data/models/model.joblib`
- `data/models/results_summary.json`

**To use your real notebook outputs instead**: run Notebooks 1–6 against
your actual Olist DB, then copy `fitted_transformers.joblib`,
`feature_list.json`, `model.joblib`, and `results_summary.json` into
`data/models/` here (or `dvc pull` them from wherever you push the real
ones). Nothing else changes — `src/features/transform.py` and
`src/models/predictor.py` only ever read these files, they don't care how
they were produced.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # includes requirements.txt
pre-commit install
```

Bootstrap artifacts (or point `data/models/` at your real ones — see above):

```bash
export PYTHONPATH=.
python scripts/bootstrap_train.py
# or, to also version the run with DVC:
dvc repro
```

Run the API locally:

```bash
uvicorn app.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

Run everything with one command, on a clean machine:

```bash
cp .env.example .env   # fill in real secrets
docker compose up --build
```

This brings up Postgres (prediction-log storage), MLflow (tracking +
model registry), and the API. The API's entrypoint bootstraps artifacts
automatically on first run if the mounted volume is empty.

## Try it

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "order_purchase_timestamp": "2026-03-14T10:15:00",
  "order_approved_at": "2026-03-14T10:42:00",
  "order_estimated_delivery_date": "2026-03-28T00:00:00",
  "n_items": 2, "n_distinct_products": 2, "n_distinct_sellers": 1,
  "total_price": 158.90, "total_freight_value": 24.50, "avg_item_price": 79.45,
  "n_payment_installments_rows": 1, "total_payment_value": 183.40, "max_installments": 3,
  "payment_type": "credit_card",
  "customer_zip_code_prefix": 14409, "customer_city": "franca", "customer_state": "SP",
  "product_category_name_english": "housewares", "product_weight_g": 500,
  "product_length_cm": 19, "product_height_cm": 8, "product_width_cm": 13,
  "seller_zip_code_prefix": 13023, "seller_city": "campinas", "seller_state": "SP"
}
EOF
```

Response:
```json
{"prediction": 0, "probability_late": 0.44, "threshold_used": 0.45,
 "model_version": "local:model.joblib", "request_id": "..."}
```

Send an order with an invalid `customer_state` and you'll get a `422`
with the failed Great Expectations rules listed.

## Tests

```bash
export PYTHONPATH=.
pytest                       # unit + data + integration, one command
pytest --cov=src --cov=app   # with coverage
```

38 tests: preprocessing reproducibility, transform never re-fits, the
`OrderInput` schema rejects leaky/ID fields, Great Expectations rejects
bad input, the model loads and predicts the right shape, and every API
route end to end (including the 422 path and `/metrics`).

## Key decisions carried over from the notebooks

- **Split is by time, not random** — production always scores future
  orders from a model trained on the past.
- **No leakage columns become features** — `src/data/preprocessing.py`'s
  `LEAKY_COLS`/`ID_COLS` match Notebook 5 exactly, and `OrderInput`
  (`src/data/schema.py`) structurally can't accept them in the first
  place — there's no field for `order_delivered_customer_date` etc.
- **Transformers are fit on train only, saved as objects, never refit** —
  `src/features/transform.py` only calls `.transform()`.
- **Metric fits the imbalance** — PR-AUC over accuracy; the chosen
  threshold from `results_summary.json` is what the API uses to turn a
  probability into a 0/1 prediction.

## Model resolution order

1. MLflow Model Registry (`mlflow.registered_model_name` /
   `mlflow.model_stage` in `config.yaml`) — used when `MLFLOW_TRACKING_URI`
   is reachable (a fast socket probe avoids a multi-minute hang if it isn't).
2. Local `data/models/model.joblib` — fallback for offline/dev/CI runs.

## Monitoring

- `GET /metrics` exposes request count, latency, and error rate
  (Prometheus format) — point Prometheus/Grafana at it.
- Every prediction is appended to `data/monitoring/prediction_log.jsonl`
  (input, output, latency, model version) so it can later be joined
  against the real delivery date, and so `src/monitoring/metrics.py`'s
  `check_drift()` has a population to compare against the training
  late-rate (`results_summary.json`'s `train_late_rate`).
- **What we'd alert on**: error rate on `/predict` > 2% over 5 minutes;
  p95 latency > 500ms; `check_drift()` flagging more than 5 points of
  absolute drift in the predicted-late rate over a rolling window of at
  least `monitoring.drift_check_min_predictions` requests.

## CI/CD

`.github/workflows/ci.yml`: **lint** (ruff + black) → **test** (pytest,
with a synthetic bootstrap so CI doesn't need real data) → **build & push**
the image to GHCR, only on `main` and only if the previous jobs passed.
`.pre-commit-config.yaml` runs the same lint/format checks plus the fast
unit+data tests before every commit.

## DVC

```bash
dvc repro   # reproduces ml_table -> labels -> split -> features -> model
dvc push    # send data/models/* to the configured remote
dvc pull    # fetch them on a clean machine instead of retraining
```

`dvc.yaml` defines one pipeline stage (`bootstrap_train`) with every
intermediate and final artifact as a tracked output, so any result can be
traced back to the exact code + config that produced it via `dvc.lock`.
The remote is configured in `.dvc/config`; point `DVC_REMOTE_URL` at your
real S3/GCS bucket for production (`dvc remote modify storage url <...>`).

## Explaining every piece

| Folder/file | Why it's there |
|---|---|
| `config/config.yaml` | single source of truth for every path/parameter — no hardcoding elsewhere |
| `src/data/preprocessing.py` | Notebook 5's feature derivation, ported unchanged |
| `src/features/transform.py` | applies (never fits) Notebook 5's saved transformers |
| `src/data/validation.py` | Great Expectations suite + reject/flag/default policy |
| `src/models/predictor.py` | resolves the model (registry → local), runs the full pipeline |
| `src/monitoring/metrics.py` | Prometheus counters/histograms + the prediction log |
| `app/main.py` | FastAPI routes: health, model info, predict, batch predict, metrics |
| `scripts/bootstrap_train.py` | stands in for "Notebooks 1–6 already ran" (see "About the data") |
| `dvc.yaml` / `.dvc/config` | data & artifact versioning, traceable back to code |
| `Dockerfile` / `docker-compose.yml` | small runtime image; full stack (db, mlflow, api) with one command |
| `.github/workflows/ci.yml` | lint → test → build & push, a failing test stops the pipeline |
| `.pre-commit-config.yaml` | same checks, caught before the push |

## Break something on purpose

```bash
# bad data
curl -X POST http://localhost:8000/predict -d '{"customer_state": "ZZ", ...}'  # -> 422

# failing test
sed -i 's/> 0/> 999/' src/data/preprocessing.py && pytest  # -> fails, CI would stop here
```
