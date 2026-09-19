"""
Data validation with Great Expectations.

Defines the `order_intake_suite`: column types, ranges, and allowed
categories for a new order, run **before** it reaches the model. An
ephemeral (in-memory) GE context is used so the API doesn't need a
filesystem-backed GE project to validate each request — the suite
definition itself is still versioned here as code, which is what actually
matters for reproducibility.

`config.validation.on_failure` decides what happens when validation fails:
- "reject": raise ValidationError, the API returns 422
- "flag":   log a warning, prediction still runs, response is flagged
- "default": missing/out-of-range values get replaced with training-set
             defaults before scoring (imputers already handle NaN, so this
             mode just clips out-of-range numerics instead of rejecting)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import great_expectations as gx
import pandas as pd

from src.config import get_settings

logger = logging.getLogger(__name__)

# Reasonable bounds derived from the Olist training data's EDA (Notebook 4).
# These are intentionally generous — the point is to catch garbage input
# (negative prices, impossible dates, unknown states), not to be a second
# copy of business rules.
NUMERIC_BOUNDS = {
    "n_items": (1, 50),
    "n_distinct_products": (1, 50),
    "n_distinct_sellers": (1, 20),
    "total_price": (0, 50_000),
    "total_freight_value": (0, 10_000),
    "avg_item_price": (0, 50_000),
    "n_payment_installments_rows": (1, 20),
    "total_payment_value": (0, 60_000),
    "max_installments": (1, 24),
    "product_weight_g": (0, 60_000),
    "product_length_cm": (0, 300),
    "product_height_cm": (0, 300),
    "product_width_cm": (0, 300),
}

VALID_BR_STATES = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}

REQUIRED_NOT_NULL = [
    "order_purchase_timestamp", "order_estimated_delivery_date",
    "n_items", "total_price", "total_freight_value",
    "customer_state", "seller_state", "payment_type",
]


class ValidationError(Exception):
    def __init__(self, message: str, failed_expectations: list[str]):
        super().__init__(message)
        self.failed_expectations = failed_expectations


@dataclass
class ValidationResult:
    success: bool
    failed_expectations: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _build_validation_definition():
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("orders_source")
    asset = data_source.add_dataframe_asset(name="orders")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")

    settings = get_settings()
    suite = context.suites.add(gx.ExpectationSuite(name=settings.ge_suite_name))

    for col in REQUIRED_NOT_NULL:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=col, min_value=lo, max_value=hi, mostly=1.0
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="customer_state", value_set=sorted(VALID_BR_STATES))
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="seller_state", value_set=sorted(VALID_BR_STATES))
    )

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(name="order_intake_validation", data=batch_def, suite=suite)
    )
    return validation_definition


def validate_orders(df: pd.DataFrame) -> ValidationResult:
    """Run the order_intake_suite against a dataframe of one or more orders."""
    validation_definition = _build_validation_definition()
    result = validation_definition.run(batch_parameters={"dataframe": df})

    if result.success:
        return ValidationResult(success=True)

    failed = [
        f"{r.expectation_config.type}(column={r.expectation_config.kwargs.get('column')})"
        for r in result.results
        if not r.success
    ]
    logger.warning("Order validation failed: %s", failed, extra={"extra_fields": {"failed_expectations": failed}})
    return ValidationResult(success=False, failed_expectations=failed)


def enforce_validation_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Apply config.validation.on_failure to a validation run.

    Returns the (possibly clipped) dataframe to proceed with, or raises
    ValidationError when the policy is "reject" and validation failed.
    """
    settings = get_settings()
    result = validate_orders(df)

    if result.success:
        return df

    policy = settings.validation_on_failure
    if policy == "reject":
        raise ValidationError("Order failed validation", result.failed_expectations)

    if policy == "flag":
        logger.warning("Proceeding despite validation failure (policy=flag): %s", result.failed_expectations)
        return df

    if policy == "default":
        df = df.copy()
        for col, (lo, hi) in NUMERIC_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)
        logger.warning("Clipped out-of-range numerics to defaults (policy=default): %s", result.failed_expectations)
        return df

    raise ValueError(f"Unknown validation.on_failure policy: {policy}")
