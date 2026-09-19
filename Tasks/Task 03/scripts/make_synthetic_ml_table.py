"""
Generates a synthetic `ml_table.parquet` with the exact schema Notebook 1
(`01_read_join_tables.ipynb`) produces from the real Olist database.

WHY THIS SCRIPT EXISTS: Task 3 assumes Notebooks 1-6 already ran against
the real Olist DB and left their artifacts in data/. The uploaded Task 2
repo has the notebooks but an empty data/ folder (no DB, no artifacts) —
so there is nothing for the inference pipeline to load. This script
stands in for "run Notebook 1 against the real DB", producing a
synthetic-but-realistic ml_table with the same columns, dtypes, and
plausible relationships (e.g. cross-state and long shipping windows
correlate with lateness) so that scripts/bootstrap_train.py can then run
the *actual* Notebook 2/3/5/6 logic — unmodified — to produce real
fitted_transformers.joblib / feature_list.json / model.joblib artifacts
for the API to load. Nothing about Notebooks 2-6's logic is changed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_settings

BR_STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "PE", "CE", "ES", "PA", "AM"]
CATEGORIES = [
    "bed_bath_table",
    "health_beauty",
    "sports_leisure",
    "furniture_decor",
    "computers_accessories",
    "housewares",
    "watches_gifts",
    "telephony",
    "auto",
    "toys",
    "cool_stuff",
    "garden_tools",
    "perfumery",
    "baby",
]
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]


def make_synthetic_ml_table(n_orders: int = 12_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    purchase_dates = (
        pd.to_datetime("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 540, size=n_orders), unit="D")
        + pd.to_timedelta(rng.integers(0, 86400, size=n_orders), unit="s")
    )
    purchase_dates = pd.Series(purchase_dates).sort_values().reset_index(drop=True)

    approval_lag_h = rng.exponential(scale=3, size=n_orders).clip(0, 96)
    approved_at = purchase_dates + pd.to_timedelta(approval_lag_h, unit="h")

    est_window_days = rng.integers(7, 45, size=n_orders)
    estimated_delivery = purchase_dates + pd.to_timedelta(est_window_days, unit="D")

    customer_state = rng.choice(BR_STATES, size=n_orders, p=_state_weights())
    seller_state = rng.choice(BR_STATES, size=n_orders, p=_state_weights())
    cross_state = (customer_state != seller_state).astype(int)

    n_items = rng.poisson(1.4, size=n_orders).clip(1, 12) + 1
    avg_item_price = rng.lognormal(mean=3.6, sigma=0.9, size=n_orders).clip(5, 3000)
    total_price = avg_item_price * n_items
    total_freight_value = (rng.lognormal(mean=2.3, sigma=0.6, size=n_orders) * (1 + 0.4 * cross_state)).clip(
        5, 500
    )

    n_distinct_products = np.minimum(n_items, rng.integers(1, 4, size=n_orders))
    n_distinct_sellers = np.ones(n_orders, dtype=int)

    n_payment_installments_rows = rng.integers(1, 3, size=n_orders)
    max_installments = rng.integers(1, 13, size=n_orders)
    total_payment_value = total_price + total_freight_value
    payment_type = rng.choice(PAYMENT_TYPES, size=n_orders, p=[0.73, 0.19, 0.05, 0.03])

    product_category_name_english = rng.choice(CATEGORIES, size=n_orders)
    product_weight_g = rng.lognormal(mean=6.5, sigma=1.1, size=n_orders).clip(50, 40000)
    product_length_cm = rng.uniform(5, 100, size=n_orders)
    product_height_cm = rng.uniform(2, 80, size=n_orders)
    product_width_cm = rng.uniform(5, 90, size=n_orders)

    customer_zip_code_prefix = rng.integers(1000, 99999, size=n_orders)
    seller_zip_code_prefix = rng.integers(1000, 99999, size=n_orders)
    customer_city = rng.choice(
        [
            "sao paulo",
            "rio de janeiro",
            "belo horizonte",
            "curitiba",
            "porto alegre",
            "salvador",
            "brasilia",
            "recife",
        ],
        size=n_orders,
    )
    seller_city = rng.choice(
        ["sao paulo", "campinas", "guarulhos", "joinville", "curitiba", "sao bernardo do campo", "ibiuna"],
        size=n_orders,
    )

    # --- ground-truth lateness process (only used to synthesize labels;
    # deliberately NOT knowable at feature time beyond what's encoded above) ---
    logit = (
        -1.8
        + 0.55 * cross_state
        + 0.35 * (est_window_days < 12).astype(int)
        - 0.30 * (est_window_days > 30).astype(int)
        + 0.20 * (n_items > 4).astype(int)
        + 0.15 * (total_freight_value > 80).astype(int)
        + rng.normal(0, 0.6, size=n_orders)
    )
    prob_late = 1 / (1 + np.exp(-logit))
    is_late_actual = (rng.uniform(size=n_orders) < prob_late).astype(int)

    actual_delay_days = np.where(
        is_late_actual == 1,
        rng.uniform(0.5, 14, size=n_orders),
        -rng.uniform(0.5, 10, size=n_orders),
    )
    delivered_customer_date = estimated_delivery + pd.to_timedelta(actual_delay_days, unit="D")
    delivered_carrier_date = purchase_dates + pd.to_timedelta(rng.uniform(0.5, 4, size=n_orders), unit="D")

    review_score = np.where(
        is_late_actual == 1, rng.integers(1, 4, size=n_orders), rng.integers(3, 6, size=n_orders)
    )

    df = pd.DataFrame(
        {
            "order_id": [f"order_{i:07d}" for i in range(n_orders)],
            "customer_id": [f"customer_{i:07d}" for i in range(n_orders)],
            "order_status": "delivered",
            "order_purchase_timestamp": purchase_dates,
            "order_approved_at": approved_at,
            "order_delivered_carrier_date": delivered_carrier_date,
            "order_delivered_customer_date": delivered_customer_date,
            "order_estimated_delivery_date": estimated_delivery,
            "n_items": n_items,
            "n_distinct_products": n_distinct_products,
            "n_distinct_sellers": n_distinct_sellers,
            "total_price": total_price.round(2),
            "total_freight_value": total_freight_value.round(2),
            "avg_item_price": avg_item_price.round(2),
            "max_shipping_limit_date": purchase_dates
            + pd.to_timedelta(rng.integers(1, 5, size=n_orders), unit="D"),
            "product_id": [f"product_{i:06d}" for i in rng.integers(0, 4000, size=n_orders)],
            "seller_id": [f"seller_{i:05d}" for i in rng.integers(0, 800, size=n_orders)],
            "n_payment_installments_rows": n_payment_installments_rows,
            "total_payment_value": total_payment_value.round(2),
            "max_installments": max_installments,
            "payment_type": payment_type,
            "review_score": review_score,
            "customer_unique_id": [f"cust_unique_{i:07d}" for i in range(n_orders)],
            "customer_zip_code_prefix": customer_zip_code_prefix,
            "customer_city": customer_city,
            "customer_state": customer_state,
            "product_category_name_english": product_category_name_english,
            "product_weight_g": product_weight_g.round(0),
            "product_length_cm": product_length_cm.round(1),
            "product_height_cm": product_height_cm.round(1),
            "product_width_cm": product_width_cm.round(1),
            "seller_zip_code_prefix": seller_zip_code_prefix,
            "seller_city": seller_city,
            "seller_state": seller_state,
        }
    )

    # sprinkle a little realistic missingness, same columns Notebook 4 flags
    for col, rate in [
        ("product_weight_g", 0.01),
        ("review_score", 0.03),
        ("product_category_name_english", 0.02),
    ]:
        mask = rng.uniform(size=n_orders) < rate
        df.loc[mask, col] = np.nan

    return df


def _state_weights() -> list[float]:
    # rough population-proportional weighting so SP/RJ/MG dominate, like real Olist data
    w = np.array([0.42, 0.13, 0.12, 0.07, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])
    return list(w / w.sum())


if __name__ == "__main__":
    settings = get_settings()
    df = make_synthetic_ml_table()
    out_path = settings.data_interim / "ml_table.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved synthetic ml_table {df.shape} -> {out_path}")
