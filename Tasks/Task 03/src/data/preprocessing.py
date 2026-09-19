"""
Raw feature construction — this is Notebook 5's `build_raw_features`
refactored into a module function, unchanged in logic. It must keep
producing exactly what the notebook produced on the same input, because
that's the contract Task 3 requires.

This step only *derives* columns (dates -> day-of-week, windows, lags,
cross_state); it does not impute, scale, or encode — that happens in
src/features/transform.py using the fitted objects from Notebook 5.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that only exist after delivery, or are IDs. A production request
# never contains these fields — OrderInput (src/data/schema.py) enforces
# that at the API boundary already, but we guard again here in case this
# function is ever called on a wider dataframe (e.g. offline batch scoring).
LEAKY_COLS = [
    "order_delivered_customer_date",
    "delivery_delay_days",
    "order_delivered_carrier_date",
    "review_score",
    "order_status",
]
ID_COLS = ["order_id", "customer_id", "customer_unique_id", "product_id", "seller_id"]


def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns Notebook 5 added, in place logic preserved."""
    df = df.copy()

    for col in ("order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    df["estimated_delivery_window_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # approval time may be missing for a brand-new order not yet approved;
    # fall back to purchase time so approval_lag_hours is 0 rather than NaN,
    # and let the numeric imputer handle any remaining edge cases.
    approved_at = df["order_approved_at"].fillna(df["order_purchase_timestamp"])
    df["approval_lag_hours"] = (
        approved_at - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    df["cross_state"] = (df["customer_state"] != df["seller_state"]).astype(int)

    drop_cols = [c for c in LEAKY_COLS + ID_COLS if c in df.columns]
    drop_cols += [c for c in (
        "order_purchase_timestamp", "order_approved_at",
        "order_estimated_delivery_date", "max_shipping_limit_date",
    ) if c in df.columns]

    df = df.drop(columns=drop_cols, errors="ignore")

    logger.debug("build_raw_features produced %d columns for %d rows", df.shape[1], df.shape[0])
    return df


def orders_to_dataframe(orders: list[dict]) -> pd.DataFrame:
    """Convert a list of validated OrderInput dicts into a DataFrame."""
    return pd.DataFrame(orders)
