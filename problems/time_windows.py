"""Synthetic time windows using the chapter 7 depot-travel opening rule."""

import math

import numpy as np


TIME_WINDOW_SCENARIOS = {"narrow": (30.0, 90.0), "wide": (60.0, 180.0)}


def resolve_time_window_bounds(scenario="wide", minimum=None, maximum=None):
    """Select preset defaults; explicit bounds override them, in minutes."""
    if scenario not in TIME_WINDOW_SCENARIOS:
        raise ValueError("time-window-scenario must be narrow or wide")
    lower, upper = TIME_WINDOW_SCENARIOS[scenario]
    lower = lower if minimum is None else float(minimum)
    upper = upper if maximum is None else float(maximum)
    if not all(math.isfinite(value) for value in (lower, upper)) or not 0 <= lower <= upper:
        raise ValueError("time-window bounds must be finite and satisfy 0 <= minimum <= maximum")
    return lower, upper


def generate_time_windows(arrivals, depot_travel, minimum, maximum, closing_horizon, rng=None):
    """Generate customer windows; inputs exclude the depot and use minutes.

    Immediate windows open at call time plus depot travel. Advance requests
    retain a random opening no earlier than direct depot arrival. Closing times
    use configurable width bounds, clipped to the final closing horizon. This
    specifies our closing-time convention where the thesis leaves sampling
    details incomplete. Draws stay attached to customer indices so changing
    dynamism does not shift subsequent instance or demand generation.
    """
    minimum, maximum = resolve_time_window_bounds("wide", minimum, maximum)
    arrivals = np.asarray(arrivals, dtype=float)
    depot_travel = np.asarray(depot_travel, dtype=float)
    if arrivals.ndim != 1 or arrivals.shape != depot_travel.shape:
        raise ValueError("arrival and depot-travel arrays must have equal one-dimensional shapes")
    if (not np.isfinite(arrivals).all() or not np.isfinite(depot_travel).all()
            or (arrivals < 0).any() or (depot_travel < 0).any()):
        raise ValueError("arrival and depot-travel times must be finite and nonnegative")
    if not math.isfinite(closing_horizon) or closing_horizon < minimum:
        raise ValueError("closing horizon must be finite and at least the minimum window width")

    rng = np.random if rng is None else rng
    starts = arrivals + depot_travel
    advance = arrivals == 0
    # Draw for every customer even when its opening is determined by its call.
    # This fixes random consumption and customer/window pairing across DoDs.
    advance_openings = rng.uniform(0, closing_horizon - minimum, size=arrivals.size)
    starts[advance] = np.maximum(
        depot_travel[advance],
        advance_openings[advance],
    )
    lower_ends = starts + minimum
    upper_ends = np.minimum(starts + maximum, closing_horizon)
    if (lower_ends > upper_ends).any():
        raise ValueError("Cannot fit the minimum time-window width before the closing horizon")
    ends = rng.uniform(lower_ends, upper_ends)
    return np.round(starts, 3), np.round(ends, 3)


def add_time_window_arguments(parser):
    parser.add_argument(
        "--time-window-scenario", "--time_window_scenario", choices=tuple(TIME_WINDOW_SCENARIOS),
        default="wide", help="Synthetic TW width preset: narrow 30–90 or wide 60–180 minutes (default: wide)",
    )
    parser.add_argument(
        "--min-time-window", "--min_time_window", type=float,
        help="Override the preset minimum window width, in minutes",
    )
    parser.add_argument(
        "--max-time-window", "--max_time_window", type=float,
        help="Override the preset maximum window width, in minutes",
    )


def validate_time_window_arguments(parser, args):
    try:
        args.min_time_window, args.max_time_window = resolve_time_window_bounds(
            args.time_window_scenario, args.min_time_window, args.max_time_window
        )
    except ValueError as error:
        parser.error(str(error))
