"""Call horizons for fixed-count routing instances; public durations use hours."""

import math


def validate_hours_per_customer(value):
    if value is not None and (
        isinstance(value, bool) or not math.isfinite(value) or value <= 0
    ):
        raise ValueError("time_horizon_per_customer must be finite and positive")
    return value


def horizon_minutes(customer_count, fixed_minutes, hours_per_customer=None):
    """Resolve T for the actual customer count, excluding the depot."""
    if hours_per_customer is None:
        return float(fixed_minutes)
    validate_hours_per_customer(hours_per_customer)
    if customer_count < 1:
        raise ValueError("A scaled horizon requires at least one customer")
    return float(customer_count * hours_per_customer * 60)
