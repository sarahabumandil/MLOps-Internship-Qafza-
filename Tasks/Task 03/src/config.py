"""
Single source of truth for configuration.

Every module in this project imports paths and parameters from here instead
of hardcoding them. Values come from config/config.yaml, with
${VAR:-default} placeholders resolved against environment variables so
secrets and per-environment settings (DB URL, MLflow URI, log level) never
have to be edited into the YAML itself.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("APP_CONFIG_PATH", ROOT / "config" / "config.yaml"))

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    """Recursively resolve ${VAR:-default} placeholders using os.environ."""
    if isinstance(value, str):

        def _sub(match: re.Match) -> str:
            var_name, _, default = match.groups()
            return os.environ.get(var_name, default if default is not None else "")

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)
    return _interpolate(raw)


class Settings:
    """Typed, convenient accessor over the raw config dict."""

    def __init__(self):
        self._cfg = load_config()

    # -- raw dict access -----------------------------------------------
    @property
    def raw(self) -> dict:
        return self._cfg

    # -- project ----------------------------------------------------------
    @property
    def label_col(self) -> str:
        return self._cfg["project"]["label_col"]

    @property
    def random_state(self) -> int:
        return self._cfg["project"]["random_state"]

    # -- paths (resolved absolute, created on first access if missing) ----
    def _path(self, key: str) -> Path:
        p = ROOT / self._cfg["paths"][key]
        return p

    @property
    def data_raw(self) -> Path:
        return self._path("data_raw")

    @property
    def data_interim(self) -> Path:
        return self._path("data_interim")

    @property
    def data_processed(self) -> Path:
        return self._path("data_processed")

    @property
    def data_models(self) -> Path:
        return self._path("data_models")

    @property
    def reports_figures(self) -> Path:
        return self._path("reports_figures")

    @property
    def prediction_log_path(self) -> Path:
        return ROOT / self._cfg["paths"]["prediction_log"]

    # -- artifacts ----------------------------------------------------
    @property
    def transformers_path(self) -> Path:
        return ROOT / self._cfg["artifacts"]["fitted_transformers"]

    @property
    def feature_list_path(self) -> Path:
        return ROOT / self._cfg["artifacts"]["feature_list"]

    @property
    def model_path(self) -> Path:
        return ROOT / self._cfg["artifacts"]["model"]

    @property
    def results_summary_path(self) -> Path:
        return ROOT / self._cfg["artifacts"]["results_summary"]

    # -- mlflow ----------------------------------------------------------
    @property
    def mlflow_tracking_uri(self) -> str:
        return self._cfg["mlflow"]["tracking_uri"]

    @property
    def mlflow_experiment_name(self) -> str:
        return self._cfg["mlflow"]["experiment_name"]

    @property
    def mlflow_registered_model_name(self) -> str:
        return self._cfg["mlflow"]["registered_model_name"]

    @property
    def mlflow_model_stage(self) -> str:
        return self._cfg["mlflow"]["model_stage"]

    # -- database ----------------------------------------------------------
    @property
    def database_url(self) -> str:
        return self._cfg["database"]["url"]

    # -- validation ----------------------------------------------------------
    @property
    def ge_root(self) -> Path:
        return ROOT / self._cfg["validation"]["ge_root"]

    @property
    def ge_suite_name(self) -> str:
        return self._cfg["validation"]["suite_name"]

    @property
    def validation_on_failure(self) -> str:
        return self._cfg["validation"]["on_failure"]

    # -- logging ----------------------------------------------------------
    @property
    def log_level(self) -> str:
        return self._cfg["logging"]["level"]

    @property
    def log_dir(self) -> Path:
        return ROOT / self._cfg["logging"]["log_dir"]

    @property
    def log_file(self) -> Path:
        return self.log_dir / self._cfg["logging"]["log_file"]

    @property
    def json_log_format(self) -> bool:
        return self._cfg["logging"]["json_format"]

    # -- api ----------------------------------------------------------
    @property
    def api_title(self) -> str:
        return self._cfg["api"]["title"]

    @property
    def api_version(self) -> str:
        return self._cfg["api"]["version"]

    @property
    def api_host(self) -> str:
        return self._cfg["api"]["host"]

    @property
    def api_port(self) -> int:
        return int(self._cfg["api"]["port"])

    # -- inference ----------------------------------------------------------
    @property
    def default_threshold(self) -> float:
        return float(self._cfg["inference"]["default_threshold"])

    # -- monitoring ----------------------------------------------------------
    @property
    def drift_check_min_predictions(self) -> int:
        return int(self._cfg["monitoring"]["drift_check_min_predictions"])

    def ensure_dirs(self) -> None:
        for p in [
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.data_models,
            self.reports_figures,
            self.log_dir,
            self.prediction_log_path.parent,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
