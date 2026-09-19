from __future__ import annotations

import pandas as pd

from src.data.preprocessing import build_raw_features
from src.features.transform import get_fitted_artifacts, transform


class TestFeatureTransform:
    def test_loads_fitted_artifacts(self):
        artifacts = get_fitted_artifacts()
        assert artifacts.numeric_features
        assert artifacts.categorical_features
        assert artifacts.feature_list
        assert len(artifacts.feature_list) > 0

    def test_transform_never_fits(self, sample_order_df):
        """Calling transform twice on different data must not change the
        fitted objects' learned parameters (e.g. scaler mean)."""
        artifacts = get_fitted_artifacts()
        mean_before = artifacts.scaler.mean_.copy()

        raw = build_raw_features(sample_order_df)
        transform(raw, artifacts)

        other = sample_order_df.copy()
        other["total_price"] = 99999.0
        raw2 = build_raw_features(other)
        transform(raw2, artifacts)

        assert (artifacts.scaler.mean_ == mean_before).all()

    def test_output_matches_feature_list_exactly(self, sample_order_df):
        artifacts = get_fitted_artifacts()
        raw = build_raw_features(sample_order_df)
        out = transform(raw, artifacts)
        assert list(out.columns) == artifacts.feature_list

    def test_output_shape(self, sample_order_df):
        artifacts = get_fitted_artifacts()
        raw = build_raw_features(sample_order_df)
        out = transform(raw, artifacts)
        assert out.shape == (1, len(artifacts.feature_list))

    def test_handles_missing_optional_columns(self, sample_order):
        """product_category_name_english etc. are Optional in the schema —
        the imputers must handle a missing value without raising."""
        order = dict(sample_order)
        order["product_category_name_english"] = None
        order["product_weight_g"] = None
        df = pd.DataFrame([order])
        artifacts = get_fitted_artifacts()
        raw = build_raw_features(df)
        out = transform(raw, artifacts)
        assert not out.isna().any().any()
