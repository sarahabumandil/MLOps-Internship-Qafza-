from __future__ import annotations

import pandas as pd

from src.data.preprocessing import ID_COLS, LEAKY_COLS, build_raw_features


class TestBuildRawFeatures:
    def test_adds_derived_columns(self, sample_order_df):
        out = build_raw_features(sample_order_df)
        for col in [
            "purchase_dow",
            "purchase_month",
            "purchase_hour",
            "estimated_delivery_window_days",
            "approval_lag_hours",
            "cross_state",
        ]:
            assert col in out.columns

    def test_drops_leaky_and_id_columns(self, sample_order_df):
        df = sample_order_df.copy()
        df["order_id"] = "order_1"
        df["order_status"] = "delivered"
        out = build_raw_features(df)
        for col in LEAKY_COLS + ID_COLS:
            assert col not in out.columns

    def test_cross_state_flag(self, sample_order_df):
        same_state = build_raw_features(sample_order_df)
        assert same_state["cross_state"].iloc[0] == 0

        df = sample_order_df.copy()
        df["seller_state"] = "RJ"
        cross = build_raw_features(df)
        assert cross["cross_state"].iloc[0] == 1

    def test_estimated_delivery_window_is_positive_for_future_estimate(self, sample_order_df):
        out = build_raw_features(sample_order_df)
        assert out["estimated_delivery_window_days"].iloc[0] > 0

    def test_missing_approval_defaults_to_purchase_time(self, sample_order):
        order = dict(sample_order)
        order["order_approved_at"] = None
        df = pd.DataFrame([order])
        out = build_raw_features(df)
        assert out["approval_lag_hours"].iloc[0] == 0

    def test_reproducible_on_same_input(self, sample_order_df):
        out1 = build_raw_features(sample_order_df)
        out2 = build_raw_features(sample_order_df)
        pd.testing.assert_frame_equal(out1, out2)
