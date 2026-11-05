"""Public problem sizes count customers; legacy total-node bounds remain explicit."""

from numbers import Integral


CUSTOMER_SIZE_CONVENTION = "customers_excluding_depot"
LEGACY_SIZE_CONVENTION = "total_nodes_including_depot"


def resolve_node_bounds(min_total=None, max_total=None, min_customers=None, max_customers=None,
                        *, default_min_customers=20, default_max_customers=100):
    """Resolve customer bounds or legacy node bounds to internal total-node bounds."""
    legacy = min_total is not None or max_total is not None
    customers = min_customers is not None or max_customers is not None
    if legacy and customers:
        raise ValueError("Use customer bounds or legacy total-node bounds, not both")
    if legacy:
        lower = default_min_customers if min_total is None else min_total
        upper = default_max_customers if max_total is None else max_total
        minimum = 2
    else:
        lower = default_min_customers if min_customers is None else min_customers
        upper = default_max_customers if max_customers is None else max_customers
        minimum = 1
    if (not isinstance(lower, Integral) or not isinstance(upper, Integral)
            or lower < minimum or upper < lower):
        raise ValueError("Size bounds must be ordered positive integer customer counts (or at least 2 total nodes)")
    return (int(lower), int(upper)) if legacy else (int(lower) + 1, int(upper) + 1)


def annotate_size_metadata(args):
    """Explain saved sizes without reinterpreting legacy checkpoint arguments."""
    if "min_customers" in args and "max_customers" in args:
        args.setdefault("size_convention", CUSTOMER_SIZE_CONVENTION)
    elif "min_total" in args and "max_total" in args:
        args["min_customers"] = args["min_total"] - 1
        args["max_customers"] = args["max_total"] - 1
        args.setdefault("size_convention", LEGACY_SIZE_CONVENTION)
    return args
