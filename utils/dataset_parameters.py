"""Shared physical-time settings for evaluating saved routing instances."""

import math


DATASET_TIME_DEFAULTS = {"speed": 4.0, "time_horizon": 8.0, "latest_end": 2.0}


def dataset_time_options(source):
    get = source.get if isinstance(source, dict) else lambda key, default: getattr(source, key, default)
    options = {key: get(key, default) for key, default in DATASET_TIME_DEFAULTS.items()}
    for key, value in options.items():
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value < 0
                or (key != "latest_end" and value == 0)):
            raise ValueError("Dataset speed and time horizon must be positive; latest-end must be nonnegative; all must be finite")
    return options


def add_dataset_time_arguments(parser):
    parser.add_argument("--speed", type=float, default=4.0, help="Vehicle speed in distance units/hour (default: 4)")
    parser.add_argument("--time-horizon", type=float, default=8.0,
                        help="Dataset call horizon in hours; must match generation (default: 8)")
    parser.add_argument("--latest-end", type=float, default=2.0,
                        help="TW closing allowance after the call horizon, in hours (default: 2)")


def validate_dataset_time_arguments(parser, args):
    try:
        dataset_time_options(args)
    except ValueError as error:
        parser.error(str(error))
