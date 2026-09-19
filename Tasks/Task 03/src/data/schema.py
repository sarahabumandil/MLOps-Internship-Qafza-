"""
Request/response schemas.

`OrderInput` mirrors exactly the pre-delivery columns of Notebook 1's
`ml_table` (order + item/payment aggregates + product/seller/customer
dimension columns) minus everything Notebook 5 marked leaky or ID-only.
This is intentional: the service must only ever see what is known at the
moment an order is placed and approved — the same contract Notebook 5
enforces with `LEAKY_COLS` / `ID_COLS`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class OrderInput(BaseModel):
    # timestamps needed to derive purchase_dow / month / hour, approval_lag,
    # estimated_delivery_window_days (Notebook 5, build_raw_features)
    order_purchase_timestamp: datetime
    order_approved_at: Optional[datetime] = None
    order_estimated_delivery_date: datetime

    # item-level aggregates (order grain, as produced by Notebook 1's items_agg)
    n_items: int = Field(ge=1)
    n_distinct_products: int = Field(ge=1)
    n_distinct_sellers: int = Field(ge=1)
    total_price: float = Field(ge=0)
    total_freight_value: float = Field(ge=0)
    avg_item_price: float = Field(ge=0)

    # payment aggregates (Notebook 1's payments_agg)
    n_payment_installments_rows: int = Field(ge=1)
    total_payment_value: float = Field(ge=0)
    max_installments: int = Field(ge=1)
    payment_type: str

    # customer dimension
    customer_zip_code_prefix: int
    customer_city: str
    customer_state: str

    # product dimension
    product_category_name_english: Optional[str] = None
    product_weight_g: Optional[float] = Field(default=None, ge=0)
    product_length_cm: Optional[float] = Field(default=None, ge=0)
    product_height_cm: Optional[float] = Field(default=None, ge=0)
    product_width_cm: Optional[float] = Field(default=None, ge=0)

    # seller dimension
    seller_zip_code_prefix: int
    seller_city: str
    seller_state: str

    @field_validator("customer_state", "seller_state")
    @classmethod
    def two_letter_state(cls, v: str) -> str:
        if v is not None and len(v) != 2:
            raise ValueError("state must be a 2-letter code, e.g. 'SP'")
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class BatchOrderInput(BaseModel):
    orders: list[OrderInput]


class PredictionOutput(BaseModel):
    prediction: int = Field(description="1 = predicted late, 0 = predicted on-time")
    probability_late: float
    threshold_used: float
    model_version: str
    request_id: str


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
