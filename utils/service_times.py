"""Known service-time priors for planning, independent of realized durations."""

import math

import numpy as np

from problems.utils import sample_lognorm


DEFAULT_SERVICE_TIME_MEAN = 3.0
DEFAULT_SERVICE_TIME_STD = math.sqrt(5.0)


def service_time_parameters(actual):
    """Return the configured prior for a single instance; never fit its samples."""
    mean = float(actual.get("service_times_mean", DEFAULT_SERVICE_TIME_MEAN))
    std = float(actual.get("service_times_std", DEFAULT_SERVICE_TIME_STD))
    if not math.isfinite(mean) or not math.isfinite(std) or mean < 0 or std < 0:
        raise ValueError("Service-time mean and standard deviation must be finite and nonnegative")
    if mean == 0 and std != 0:
        raise ValueError("A zero service-time mean requires zero variance")
    return mean, std


def sample_service_times(actual, count):
    """Sample independent log-normal customer durations in the parent process."""
    mean, std = service_time_parameters(actual)
    if std == 0:
        return np.full(count, mean, dtype=float)
    return np.round(sample_lognorm(mean, std, size=count), decimals=3)


def add_service_time_arguments(parser):
    parser.add_argument(
        "--service-times-mean", type=float, default=DEFAULT_SERVICE_TIME_MEAN,
        help="Known mean customer service duration in minutes (default: 3)",
    )
    parser.add_argument(
        "--service-times-var", type=float, default=5.0,
        help="Known customer service-duration variance in minutes squared (default: 5)",
    )


def validate_service_time_arguments(parser, args):
    if not math.isfinite(args.service_times_var) or args.service_times_var < 0:
        parser.error("service-times-var must be finite and nonnegative")
    try:
        service_time_parameters({
            "service_times_mean": args.service_times_mean,
            "service_times_std": math.sqrt(args.service_times_var),
        })
    except ValueError as error:
        parser.error(str(error))
