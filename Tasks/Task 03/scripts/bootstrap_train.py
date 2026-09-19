"""
Bootstraps the artifacts Notebooks 2, 3, 5 and 6 would normally leave in
data/ after being run against the real Olist database. Training itself
still conceptually "lives in the notebooks" per the task sheet — this
script exists only because no populated Olist DB was available in this
environment. The label/split/feature/model logic below is a line-for-line
port of the notebook cells (see Task 2 notebooks 02, 03, 05, 06); nothing
about *how* the model is built was changed to make the inference pipeline
work.

Usage:
    python scripts/bootstrap_train.py
"""

from __future__ import annotations

import json
import logging

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.make_synthetic_ml_table import make_synthetic_ml_table
from src.config import get_settings
from src.data.preprocessing import build_raw_features
from src.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def _mlflow_server_reachable(tracking_uri: str, timeout: float = 1.0) -> bool:
    """Fast pre-check so we don't wait through mlflow's own multi-minute
    HTTP retry storm when the tracking server simply isn't up (e.g. running
    this script outside docker-compose)."""
    import socket
    from urllib.parse import urlparse

    if not tracking_uri.startswith("http"):
        return True  # e.g. sqlite:///, file: URIs — not a server, nothing to probe
    parsed = urlparse(tracking_uri)
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def notebook_02_create_labels(ml_table: pd.DataFrame, settings) -> pd.DataFrame:
    for c in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        ml_table[c] = pd.to_datetime(ml_table[c])

    labeled = ml_table[ml_table["order_delivered_customer_date"].notna()].copy()
    labeled["delivery_delay_days"] = (
        labeled["order_delivered_customer_date"] - labeled["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    labeled["is_late"] = (labeled["delivery_delay_days"] > 0).astype(int)

    logger.info("Labeled table: %d rows, late rate=%.3f", len(labeled), labeled["is_late"].mean())
    labeled.to_parquet(settings.data_interim / "labeled_table.parquet", index=False)
    return labeled


def notebook_03_split(labeled: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled_sorted = labeled.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(labeled_sorted)
    train_end, val_end = int(n * 0.70), int(n * 0.85)

    train = labeled_sorted.iloc[:train_end].copy()
    val = labeled_sorted.iloc[train_end:val_end].copy()
    test = labeled_sorted.iloc[val_end:].copy()

    for name, df in [("train", train), ("val", val), ("test", test)]:
        logger.info("%s: n=%d late_rate=%.3f", name, len(df), df["is_late"].mean())

    train.to_parquet(settings.data_processed / "train.parquet", index=False)
    val.to_parquet(settings.data_processed / "val.parquet", index=False)
    test.to_parquet(settings.data_processed / "test.parquet", index=False)
    return train, val, test


LEAKY_COLS_FOR_TRAINING = ["delivery_delay_days"]  # is_late itself is the label, kept separately


def notebook_05_feature_engineering(train, val, test, settings):
    train_f = build_raw_features(
        train.drop(columns=["is_late"] + LEAKY_COLS_FOR_TRAINING, errors="ignore").assign(
            is_late=train["is_late"]
        )
    )
    val_f = build_raw_features(
        val.drop(columns=["is_late"] + LEAKY_COLS_FOR_TRAINING, errors="ignore").assign(
            is_late=val["is_late"]
        )
    )
    test_f = build_raw_features(
        test.drop(columns=["is_late"] + LEAKY_COLS_FOR_TRAINING, errors="ignore").assign(
            is_late=test["is_late"]
        )
    )

    candidate_cols = [c for c in train_f.columns if c != "is_late"]
    numeric_features = train_f[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train_f[candidate_cols].select_dtypes(include=["object"]).columns.tolist()
    logger.info("numeric_features=%s", numeric_features)
    logger.info("categorical_features=%s", categorical_features)

    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(train_f[numeric_features])

    scaler = StandardScaler()
    scaler.fit(num_imputer.transform(train_f[numeric_features]))

    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_imputer.fit(train_f[categorical_features])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)
    ohe.fit(cat_imputer.transform(train_f[categorical_features]))

    def _transform(df):
        num = scaler.transform(num_imputer.transform(df[numeric_features]))
        num_df = pd.DataFrame(num, columns=numeric_features, index=df.index)
        cat = ohe.transform(cat_imputer.transform(df[categorical_features]))
        cat_df = pd.DataFrame(cat, columns=ohe.get_feature_names_out(categorical_features), index=df.index)
        out = pd.concat([num_df, cat_df], axis=1)
        out["is_late"] = df["is_late"].values
        return out

    features_train = _transform(train_f)
    features_val = _transform(val_f)
    features_test = _transform(test_f)

    features_train.to_parquet(settings.data_processed / "features_train.parquet", index=False)
    features_val.to_parquet(settings.data_processed / "features_val.parquet", index=False)
    features_test.to_parquet(settings.data_processed / "features_test.parquet", index=False)

    joblib.dump(
        {
            "num_imputer": num_imputer,
            "scaler": scaler,
            "cat_imputer": cat_imputer,
            "ohe": ohe,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        settings.transformers_path,
    )

    feature_list = [c for c in features_train.columns if c != "is_late"]
    with open(settings.feature_list_path, "w") as f:
        json.dump(feature_list, f, indent=2)

    logger.info("Saved %d features -> %s", len(feature_list), settings.feature_list_path)
    return features_train, features_val, features_test


def notebook_06_train_tune_evaluate(features_train, features_val, features_test, settings):
    label_col = settings.label_col
    X_train, y_train = features_train.drop(columns=[label_col]), features_train[label_col]
    X_val, y_val = features_val.drop(columns=[label_col]), features_val[label_col]
    X_test, y_test = features_test.drop(columns=[label_col]), features_test[label_col]

    if _mlflow_server_reachable(settings.mlflow_tracking_uri):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    else:
        # No MLflow tracking server reachable (e.g. running this script
        # outside docker-compose) — fall back to a local sqlite-backed
        # store so the run is still tracked, just not centrally.
        logger.warning(
            "MLflow tracking server at %s unreachable, falling back to local sqlite:///mlflow.db",
            settings.mlflow_tracking_uri,
        )
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="bootstrap_train") as run:
        baseline = DummyClassifier(strategy="stratified", random_state=settings.random_state)
        baseline.fit(X_train, y_train)
        baseline_pr_auc = average_precision_score(y_val, baseline.predict_proba(X_val)[:, 1])

        logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
        logreg.fit(X_train, y_train)
        lr_probs = logreg.predict_proba(X_val)[:, 1]
        logreg_pr_auc = average_precision_score(y_val, lr_probs)

        mlflow.log_metric("baseline_stratified_pr_auc_val", baseline_pr_auc)
        mlflow.log_metric("baseline_logreg_pr_auc_val", logreg_pr_auc)

        param_grid = {"n_estimators": [200, 400], "max_depth": [6, 12, None], "min_samples_leaf": [1, 5]}
        best_score, best_params, best_model = -1, None, None
        for params in ParameterGrid(param_grid):
            model = RandomForestClassifier(
                random_state=settings.random_state, class_weight="balanced", n_jobs=-1, **params
            )
            model.fit(X_train, y_train)
            score = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
            if score > best_score:
                best_score, best_params, best_model = score, params, model

        logger.info("Best params: %s, val PR-AUC=%.4f", best_params, best_score)
        mlflow.log_params(best_params)
        mlflow.log_metric("val_pr_auc", best_score)

        val_probs = best_model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
        best_idx = np.nanargmax(f1s[:-1])
        best_threshold = float(thresholds[best_idx])

        test_probs = best_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= best_threshold).astype(int)
        test_pr_auc = average_precision_score(y_test, test_probs)
        test_roc_auc = roc_auc_score(y_test, test_probs)
        test_f1 = f1_score(y_test, test_preds)

        logger.info("TEST PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f", test_pr_auc, test_roc_auc, test_f1)
        mlflow.log_metric("test_pr_auc", test_pr_auc)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("test_f1_late_class", test_f1)
        mlflow.log_metric("chosen_threshold", best_threshold)

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        try:
            result = mlflow.register_model(
                f"runs:/{run.info.run_id}/model", settings.mlflow_registered_model_name
            )
            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name=settings.mlflow_registered_model_name,
                version=result.version,
                stage=(
                    settings.mlflow_model_stage.replace("Production", "Production").replace(
                        "Staging", "Staging"
                    )
                    if settings.mlflow_model_stage in ("Production", "Staging")
                    else "None"
                ),
            )
            logger.info("Registered model version %s as %s", result.version, settings.mlflow_model_stage)
        except Exception:
            logger.warning(
                "MLflow model registry not reachable — model is still saved locally.", exc_info=True
            )

        joblib.dump(best_model, settings.model_path)

        results_summary = {
            "baseline_stratified_pr_auc_val": float(baseline_pr_auc),
            "baseline_logreg_pr_auc_val": float(logreg_pr_auc),
            "model": "RandomForestClassifier",
            "best_params": best_params,
            "chosen_threshold": best_threshold,
            "val_pr_auc": float(best_score),
            "test_pr_auc": float(test_pr_auc),
            "test_roc_auc": float(test_roc_auc),
            "test_f1_late_class": float(test_f1),
            "train_late_rate": float(y_train.mean()),
            "mlflow_run_id": run.info.run_id,
        }
        with open(settings.results_summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)

        return results_summary


def main():
    settings = get_settings()
    logger.info("Bootstrapping synthetic ml_table (stand-in for Notebook 1 against the real DB)")
    ml_table = make_synthetic_ml_table()
    ml_table.to_parquet(settings.data_interim / "ml_table.parquet", index=False)

    labeled = notebook_02_create_labels(ml_table, settings)
    train, val, test = notebook_03_split(labeled, settings)
    features_train, features_val, features_test = notebook_05_feature_engineering(train, val, test, settings)
    results = notebook_06_train_tune_evaluate(features_train, features_val, features_test, settings)

    logger.info("Bootstrap complete. Results: %s", json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
