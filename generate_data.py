#!/usr/bin/env python
"""Generate synthetic benchmarks with uniform or normalized ORTEC locations."""

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from evaluate_model_free import nearest_neighbour
from problems.time_windows import add_time_window_arguments, validate_time_window_arguments
from problems.size_conventions import CUSTOMER_SIZE_CONVENTION
from problems.ortec_locations import load_ortec_locations, make_ortec_location_dataset
from utils import load_problem
from utils.data_utils import dataset_to_input, format_dataset_row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("pdtrp", "pdctrp", "pdtrptw", "pdctrptw"), default="pdtrp")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--num-customers", "--num_customers", type=int,
                      help="N customers EXCLUDING the depot (default: 50; 51 nodes)")
    size.add_argument("--num-nodes", type=int,
                      help="Legacy explicit node count INCLUDING the depot; use --num-customers for N")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--dod", type=float, default=0.5, help="Dynamic fraction, floored to an integer customer count")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ortec-locations", type=Path,
        help="Use only this ORTEC file's normalized coordinates, with synthetic timings and Euclidean travel",
    )
    parser.add_argument("--speed", type=float, default=4.0, help="Distance units per hour")
    parser.add_argument("--time-horizon", type=float, default=8.0, help="Call horizon in hours")
    parser.add_argument("--dimension-scale", type=float, default=1.0)
    parser.add_argument("--service-times-mean", type=float, default=3.0, help="Minutes")
    parser.add_argument("--service-times-var", type=float, default=5.0, help="Minutes squared")
    parser.add_argument("--vehicle-capacity", type=float, default=1.0)
    parser.add_argument("--min-trips-required", type=int, default=3)
    parser.add_argument("--max-trips-required", type=int, default=9)
    parser.add_argument("--latest-end", type=float, default=2.0, help="Hours beyond the call horizon")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=1.0)
    add_time_window_arguments(parser)
    args = parser.parse_args(argv)
    validate_time_window_arguments(parser, args)
    args.num_customers = (args.num_nodes - 1 if args.num_nodes is not None else
                          50 if args.num_customers is None else args.num_customers)
    args.num_nodes = args.num_customers + 1
    if args.num_customers < 1 or args.num_samples < 1:
        parser.error("num-customers and num-samples must be positive")
    if not 0 <= args.dod <= 1:
        parser.error("dod must be between 0 and 1")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    positive = (args.speed, args.time_horizon, args.dimension_scale, args.vehicle_capacity)
    nonnegative = (args.service_times_mean, args.service_times_var, args.latest_end, args.gamma, args.theta)
    if (not all(math.isfinite(x) and x > 0 for x in positive)
            or not all(math.isfinite(x) and x >= 0 for x in nonnegative)):
        parser.error("generation parameters must be finite with valid positive/nonnegative ranges")
    if args.time_horizon * 60 < 1:
        parser.error("time-horizon must be at least one minute")
    if args.service_times_mean == 0 and args.service_times_var != 0:
        parser.error("zero service-time mean requires zero variance")
    if args.problem in ("pdctrp", "pdctrptw") and not 1 <= args.min_trips_required <= args.max_trips_required <= args.num_customers:
        parser.error("trip bounds must satisfy 1 <= minimum <= maximum <= customer count")
    return args


def generate(args):
    output = args.output
    metadata_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Dataset or metadata already exists: {output}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    problem = load_problem(args.problem)
    options = dict(
        min_customers=args.num_customers, max_customers=args.num_customers,
        min_dod=args.dod, max_dod=args.dod, batch_size=1, num_samples=args.num_samples,
        n_subregions=1, arrival_weights=[1], arrival_skews=["uniform"],
        speed=args.speed, time_horizon=args.time_horizon, dimension_scale=args.dimension_scale,
        service_times_mean=args.service_times_mean, service_times_var=args.service_times_var,
        neighbors=-1, knn_strat="fixed", latest_end=args.latest_end,
    )
    if args.problem in ("pdctrp", "pdctrptw"):
        options.update(vehicle_capacity=args.vehicle_capacity,
                       min_trips_required_lb=args.min_trips_required,
                       min_trips_required_ub=args.max_trips_required)
    if args.problem in ("pdtrptw", "pdctrptw"):
        options.update(time_window_scenario=args.time_window_scenario,
                       min_time_window=args.min_time_window, max_time_window=args.max_time_window,
                       gamma=args.gamma, theta=args.theta)

    # Keep exactly the sampled instances: no solver-success filtering or replacements.
    locations = None
    if args.ortec_locations is None:
        dataset = problem.make_dataset(**options)
    else:
        locations = load_ortec_locations(args.ortec_locations, args.dimension_scale)
        dataset, source_ids = make_ortec_location_dataset(problem, options, locations, args.seed)
    rows = []
    for index, sample in enumerate(dataset):
        actual = dataset_to_input(sample, args.problem, "cpu")
        tour, times = nearest_neighbour(problem, actual)
        # Legacy loaders require a reference route; this is a feasible heuristic
        # reference, not an optimal or OR-Tools solution.
        # Preserve the original coordinate precision: loaders reconstruct the
        # Euclidean metric before converting their node tensors to float32.
        serialized_sample = {**sample, "all_nodes": dataset.all_nodes[index]}
        rows.append(format_dataset_row(serialized_sample, tour, times))
    payload = "".join(rows)
    source_root = Path(__file__).resolve().parent
    problem_dir = {"pdtrptw": "pdtrp_tw", "pdctrptw": "pdctrp_tw"}.get(args.problem, args.problem)
    sources = [
        "generate_data.py", "evaluate_model_free.py", "utils/data_utils.py",
        "problems/location_generation.py", "problems/utils.py", "problems/time_windows.py",
        "problems/size_conventions.py",
        "problems/request_events.py", "problems/time_window_events.py",
        f"problems/{problem_dir}/problem_{args.problem}.py",
        f"problems/{problem_dir}/state_{args.problem}.py",
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True, capture_output=True,
    ).stdout.strip() or None
    dynamic_count = int(args.dod * args.num_customers)
    metadata = {
        "schema_version": 3, "size_convention": CUSTOMER_SIZE_CONVENTION,
        "customer_order": "sampling_order",
        "distribution": "uniform-uniform" if locations is None else "ortec-locations-uniform",
        "problem": args.problem,
        "seed": args.seed, "num_samples": len(dataset), "num_nodes_including_depot": args.num_nodes,
        "num_customers": args.num_customers, "dynamic_customers": dynamic_count,
        "requested_dod": args.dod, "realized_dod": dynamic_count / args.num_customers,
        "coordinate_bounds": [0.0, args.dimension_scale],
        "dynamic_call_bounds_minutes": [1.0, args.time_horizon * 60],
        "rounding_decimals": 3, "reference_method": "nearest_neighbour",
        "reference_is_optimal": False, "dataset_options": options,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(), "git_revision": revision,
        "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in sources},
    }
    if locations is not None:
        metadata.update({
            "distance_model": "euclidean", "coordinate_rounding_decimals": None,
            "location_source": {key: locations[key] for key in ("filename", "sha256", "normalization")},
            "source_id_convention": "zero_based_including_depot",
            "source_ids": source_ids,
        })
        name = "problems/ortec_locations.py"
        metadata["source_sha256"][name] = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(payload)
    with metadata_path.open("x") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return {"dataset": str(output), "metadata": str(metadata_path),
            "samples": len(dataset), "sha256": metadata["sha256"]}


def main(argv=None):
    args = parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        result = generate(args)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
