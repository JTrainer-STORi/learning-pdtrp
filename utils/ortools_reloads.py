"""Depot-return allowances shared by the capacity-routing scenario builders."""


DEPOT_RETURN_POLICY = "one_optional_reload_per_unserved_customer"


def depot_return_allowance(customer_count):
    """Allow a reload before every remaining customer, including the first.

    One customer per trip is sufficient for any useful route: consecutive depot
    visits add no capacity benefit, and the final return is already implicit in
    the routing model. Including a possible initial reload also covers a solve
    that starts away from the depot with capacity already used. These optional
    nodes must not be limited by demand packing or demand-generation trip bounds.
    """
    if customer_count < 0:
        raise ValueError("customer count must be nonnegative")
    return customer_count
