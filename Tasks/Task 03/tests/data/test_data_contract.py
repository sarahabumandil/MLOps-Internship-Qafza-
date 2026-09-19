from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.data.preprocessing import ID_COLS, LEAKY_COLS
from src.data.schema import OrderInput
from src.data.validation import enforce_validation_policy, validate_orders
from src.data.preprocessing import build_raw_features
import pandas as pd


class TestSchema:
    def test_valid_order_parses(self, sample_order):
        order = OrderInput(**sample_order)
        assert order.customer_state == "SP"

    def test_rejects_negative_price(self, sample_order):
        bad = dict(sample_order)
        bad["total_price"] = -1.0
        with pytest.raises(PydanticValidationError):
            OrderInput(**bad)

    def test_rejects_zero_items(self, sample_order):
        bad = dict(sample_order)
        bad["n_items"] = 0
        with pytest.raises(PydanticValidationError):
            OrderInput(**bad)

    def test_state_is_uppercased(self, sample_order):
        low = dict(sample_order)
        low["customer_state"] = "sp"
        order = OrderInput(**low)
        assert order.customer_state == "SP"

    def test_rejects_bad_state_length(self, sample_order):
        bad = dict(sample_order)
        bad["customer_state"] = "SAO"
        with pytest.raises(PydanticValidationError):
            OrderInput(**bad)

    def test_schema_never_accepts_leaky_or_id_fields(self, sample_order):
        """OrderInput must not define any field that Notebook 5 marks as
        leaky or ID-only — that's the whole point of the input contract."""
        for banned in LEAKY_COLS + ID_COLS:
            assert banned not in OrderInput.model_fields


class TestGreatExpectationsValidation:
    def test_valid_order_passes(self, sample_order_df):
        result = validate_orders(sample_order_df)
        assert result.success

    def test_invalid_state_fails(self, invalid_order):
        df = pd.DataFrame([invalid_order])
        result = validate_orders(df)
        assert not result.success
        assert any("customer_state" in f for f in result.failed_expectations)

    def test_reject_policy_raises_by_default(self, invalid_order):
        """config/config.yaml's default validation.on_failure is 'reject'."""
        from src.config import get_settings
        from src.data.validation import ValidationError

        settings = get_settings()
        assert settings.validation_on_failure == "reject"

        df = pd.DataFrame([invalid_order])
        with pytest.raises(ValidationError):
            enforce_validation_policy(df)


class TestLeakageWatchlist:
    def test_build_raw_features_output_excludes_leaky_columns(self, sample_order_df):
        df = sample_order_df.copy()
        df["order_delivered_customer_date"] = "2026-03-30"
        df["review_score"] = 5
        out = build_raw_features(df)
        assert "order_delivered_customer_date" not in out.columns
        assert "review_score" not in out.columns
