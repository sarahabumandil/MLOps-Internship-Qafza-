from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_order() -> dict:
    return {
        "order_purchase_timestamp": "2026-03-14T10:15:00",
        "order_approved_at": "2026-03-14T10:42:00",
        "order_estimated_delivery_date": "2026-03-28T00:00:00",
        "n_items": 2,
        "n_distinct_products": 2,
        "n_distinct_sellers": 1,
        "total_price": 158.90,
        "total_freight_value": 24.50,
        "avg_item_price": 79.45,
        "n_payment_installments_rows": 1,
        "total_payment_value": 183.40,
        "max_installments": 3,
        "payment_type": "credit_card",
        "customer_zip_code_prefix": 14409,
        "customer_city": "franca",
        "customer_state": "SP",
        "product_category_name_english": "housewares",
        "product_weight_g": 500,
        "product_length_cm": 19,
        "product_height_cm": 8,
        "product_width_cm": 13,
        "seller_zip_code_prefix": 13023,
        "seller_city": "campinas",
        "seller_state": "SP",
    }


@pytest.fixture
def sample_order_df(sample_order) -> pd.DataFrame:
    return pd.DataFrame([sample_order])


@pytest.fixture
def invalid_order(sample_order) -> dict:
    """Fails Great Expectations validation (unknown state) while still
    passing Pydantic's own field-level checks, so it reaches src.data.validation
    rather than being rejected earlier by FastAPI's request parsing."""
    bad = dict(sample_order)
    bad["customer_state"] = "ZZ"  # not a real BR state
    return bad
