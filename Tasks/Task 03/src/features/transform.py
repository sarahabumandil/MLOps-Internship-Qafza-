"""
Applies the fitted transformers saved by Notebook 5 (`fitted_transformers.joblib`)
to new, raw-feature-engineered orders.

Hard rule, per the task sheet: **never call `.fit()` here.** Only
`.transform()`, using exactly the objects the notebook fitted on the
training split.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FittedArtifacts:
    num_imputer: object
    scaler: object
    cat_imputer: object
    ohe: object
    numeric_features: list[str]
    categorical_features: list[str]
    feature_list: list[str]
    transformers_path: Path
    feature_list_path: Path


def load_fitted_artifacts() -> FittedArtifacts:
    settings = get_settings()
    t_path = settings.transformers_path
    f_path = settings.feature_list_path

    if not t_path.exists():
        raise FileNotFoundError(
            f"Fitted transformers not found at {t_path}. Run scripts/bootstrap_train.py "
            "(or Notebook 5) first, or point config.paths.data_models at the model registry."
        )
    if not f_path.exists():
        raise FileNotFoundError(f"Feature list not found at {f_path}.")

    bundle = joblib.load(t_path)
    with open(f_path) as f:
        feature_list = json.load(f)

    logger.info("Loaded fitted transformers from %s (%d features)", t_path, len(feature_list))

    return FittedArtifacts(
        num_imputer=bundle["num_imputer"],
        scaler=bundle["scaler"],
        cat_imputer=bundle["cat_imputer"],
        ohe=bundle["ohe"],
        numeric_features=bundle["numeric_features"],
        categorical_features=bundle["categorical_features"],
        feature_list=feature_list,
        transformers_path=t_path,
        feature_list_path=f_path,
    )


@lru_cache(maxsize=1)
def get_fitted_artifacts() -> FittedArtifacts:
    return load_fitted_artifacts()


def transform(df: pd.DataFrame, artifacts: FittedArtifacts | None = None) -> pd.DataFrame:
    """Apply the fitted imputers/scaler/encoder to a raw-feature-engineered
    dataframe (output of src.data.preprocessing.build_raw_features).

    Missing expected columns are added as NaN so the imputers can still
    handle them consistently with how they were fit.
    """
    artifacts = artifacts or get_fitted_artifacts()

    for col in artifacts.numeric_features + artifacts.categorical_features:
        if col not in df.columns:
            df[col] = pd.NA

    num = artifacts.scaler.transform(artifacts.num_imputer.transform(df[artifacts.numeric_features]))
    num_df = pd.DataFrame(num, columns=artifacts.numeric_features, index=df.index)

    cat = artifacts.ohe.transform(artifacts.cat_imputer.transform(df[artifacts.categorical_features]))
    cat_df = pd.DataFrame(
        cat, columns=artifacts.ohe.get_feature_names_out(artifacts.categorical_features), index=df.index
    )

    out = pd.concat([num_df, cat_df], axis=1)

    # guarantee exact column order/set the model was trained on
    for col in artifacts.feature_list:
        if col not in out.columns:
            out[col] = 0.0
    out = out[artifacts.feature_list]

    return out
