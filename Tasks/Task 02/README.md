# MLOps Task 2 — From Tables to Notebooks

Predicting **late vs. on-time delivery** on the Olist dataset. Six
notebooks, each doing one job, each reading the artifact the previous one
wrote and writing an artifact for the next one.

```
mlops-task2/
├── README.md
├── requirements.txt
├── src/
│   └── config.py              # DB connection + all artifact paths in one place
├── notebooks/
│   ├── 01_read_join_tables.ipynb
│   ├── 02_create_labels.ipynb
│   ├── 03_train_val_test_split.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_train_tune_evaluate.ipynb
├── data/
│   ├── raw/                   # (unused directly — source is the local DB)
│   ├── interim/                # ml_table.parquet, labeled_table.parquet
│   ├── processed/              # train/val/test + feature tables
│   └── models/                 # fitted transformers, feature list, model, results
└── reports/
    └── figures/                 # saved charts + eda_findings.md
```

## Setup

```bash
pip install -r requirements.txt
```

Point the notebooks at your Task 1 database by setting `DATABASE_URL`
before launching Jupyter (defaults to `sqlite:///../olist.db`):

```bash
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/olist"
# or, for the sqlite default, just make sure ../olist.db exists relative to notebooks/
jupyter lab notebooks/
```

Run the notebooks **in order, from a clean start** — 01 → 06. Each one
assumes the previous one's artifact already exists on disk.

## The artifact contract

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | Read & join tables | Olist DB | `data/interim/ml_table.parquet` — one row per order |
| 2 | Create the labels | `ml_table.parquet` | `data/interim/labeled_table.parquet` |
| 3 | Train/val/test split | `labeled_table.parquet` | `data/processed/{train,val,test}.parquet` |
| 4 | EDA (detailed) | `train.parquet` only | `reports/figures/eda_findings.md` + charts |
| 5 | Feature engineering | `{train,val,test}.parquet` | `features_{train,val,test}.parquet`, `fitted_transformers.joblib`, `feature_list.json` |
| 6 | Train, tune, evaluate | `features_{train,val,test}.parquet` | `model.joblib`, `results_summary.json` |

Every notebook only reads the step before it and only writes for the step
after it — no notebook reaches back further than that or forward past it.
This is what lets these become independent production scripts later
without a rewrite: same inputs in, same artifacts out.

## Key decisions baked into this pipeline

- **Split is by time, not random** (Notebook 3) — production always scores
  future orders from a model trained on the past, so the split mirrors
  that instead of leaking future patterns into training via random shuffle.
- **EDA only touches the training split** (Notebook 4) — the validation
  and test sets are never opened before modeling decisions are locked in.
- **No leakage columns become features** (Notebook 5) — anything only
  known after delivery (`order_delivered_customer_date`, `review_score`,
  etc.) is explicitly dropped before the feature table is built.
- **Transformers are fit on train only, saved as objects** (Notebook 5) —
  the production pipeline loads and applies these exact fitted objects;
  it never re-fits on new data.
- **Metric fits the imbalance** (Notebook 6) — PR-AUC over accuracy, with
  a baseline to beat and the test set touched exactly once, at the end.

## Done when

- [ ] Six notebooks run in order from a clean start
- [ ] Every notebook's artifact exists and the next notebook reads it
- [ ] You can explain why the split is time-based rather than random
- [ ] You have a model result compared against a simple baseline in
      `data/models/results_summary.json`

## Notes

- One notebook, one job — nothing here is combined into a single mega-notebook.
- No production scripts yet — that's a later task; this stays notebook-based on purpose.
