"""
Shared configuration for the Task 2 notebooks.

All notebooks import from here so the DB connection and the artifact
paths are defined in exactly one place. This is the thing that keeps
"notebook 5 reads what notebook 4 wrote" from turning into copy-pasted
path strings scattered across six files.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------
# Task 1 leaves you with a local database containing the Olist tables.
# Point DATABASE_URL at it. Examples:
#   sqlite:////absolute/path/to/olist.db
#   postgresql+psycopg2://user:password@localhost:5432/olist
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///../olist.db")


def get_engine():
    return create_engine(DATABASE_URL)


# ---------------------------------------------------------------------------
# Paths — one notebook's output is the next notebook's input
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_MODELS = ROOT / "data" / "models"
REPORTS_FIGURES = ROOT / "reports" / "figures"

for p in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_MODELS, REPORTS_FIGURES]:
    p.mkdir(parents=True, exist_ok=True)

# Notebook 1 artifact
ML_TABLE_PATH = DATA_INTERIM / "ml_table.parquet"

# Notebook 2 artifact
LABELED_TABLE_PATH = DATA_INTERIM / "labeled_table.parquet"

# Notebook 3 artifacts
TRAIN_PATH = DATA_PROCESSED / "train.parquet"
VAL_PATH = DATA_PROCESSED / "val.parquet"
TEST_PATH = DATA_PROCESSED / "test.parquet"

# Notebook 4 artifacts
FINDINGS_PATH = REPORTS_FIGURES / "eda_findings.md"

# Notebook 5 artifacts
FEATURES_TRAIN_PATH = DATA_PROCESSED / "features_train.parquet"
FEATURES_VAL_PATH = DATA_PROCESSED / "features_val.parquet"
FEATURES_TEST_PATH = DATA_PROCESSED / "features_test.parquet"
TRANSFORMERS_PATH = DATA_MODELS / "fitted_transformers.joblib"
FEATURE_LIST_PATH = DATA_MODELS / "feature_list.json"

# Notebook 6 artifacts
MODEL_PATH = DATA_MODELS / "model.joblib"
RESULTS_PATH = DATA_MODELS / "results_summary.json"

RANDOM_STATE = 42
LABEL_COL = "is_late"
