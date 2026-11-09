"""Shared scenario construction, arguments, and reporting for OR-Tools baselines."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_msa_ortools import CAPACITY_PROBLEMS, DEMAND_SCALE, TW_PROBLEMS, initial_route_policy
from utils import load_problem
from utils.data_utils import dataset_to_input
from utils.dataset_parameters import add_dataset_time_arguments, dataset_time_options, validate_dataset_time_arguments
from utils.ortools_reloads import DEPOT_RETURN_POLICY, depot_return_allowance
from utils.service_times import add_service_time_arguments, validate_service_time_arguments
from utils.solver_budgets import (
    add_solver_budget_arguments,
    run_solver_budget_sweep,
    validate_solver_budget_arguments,
)


def scenario_from_nodes(
    problem_name, actual, current_node, node_ids, used_capacity, max_trips,
    *, service_time_mean=None,
):
    """Build a scenario, optionally replacing unserved durations by their mean.

    Realized durations remain the default for the explicitly clairvoyant oracle.
    Depot/reload service is a known operational duration, not a customer draw.
    ``max_trips`` is retained for caller compatibility; it does not cap reloads.
    """
    capacity = (
        float(actual["vehicle_capacity"].item())
        if problem_name in CAPACITY_PROBLEMS
        else 1.0
    )
    reload_count = 0
    if problem_name in CAPACITY_PROBLEMS:
        # Preserve the demand-packing order used by the feasible incumbent.
        # Packing must not restrict the model's optional depot returns.
        bins = [[]]
        packed_capacity = int(round(capacity * DEMAND_SCALE))
        bin_loads = [int(round(used_capacity * DEMAND_SCALE))]
        ordered = sorted(
            node_ids,
            key=lambda node: float(actual["demand"][0, node]),
            reverse=True,
        )
        for node in ordered:
            node_demand = int(round(float(actual["demand"][0, node]) * DEMAND_SCALE))
            for index, load in enumerate(bin_loads):
                if load + node_demand <= packed_capacity:
                    bins[index].append(node)
                    bin_loads[index] += node_demand
                    break
            else:
                bins.append([node])
                bin_loads.append(node_demand)
        node_ids = [node for group in bins for node in group]
        reload_count = depot_return_allowance(len(node_ids))

    source_ids = [current_node]
    mapping = [None]
    service = [0.0]
    arrivals = [0.0]
    demand = [used_capacity] if problem_name in CAPACITY_PROBLEMS else []
    windows = [(0.0, 10**6)]
    reload_nodes = []

    if problem_name in CAPACITY_PROBLEMS:
        for _ in range(reload_count):
            reload_nodes.append(len(source_ids))
            source_ids.append(0)
            mapping.append(0)
            service.append(float(actual["service_times"][0, 0]))
            arrivals.append(float(actual["arrival_times"][0, 0]))
            demand.append(-capacity)
            windows.append(
                (
                    float(actual["window_starts"][0, 0]),
                    float(actual["window_ends"][0, 0]),
                )
                if problem_name in TW_PROBLEMS
                else (0.0, 10**6)
            )

    for node in node_ids:
        source_ids.append(node)
        mapping.append(node)
        service.append(
            float(actual["service_times"][0, node])
            if service_time_mean is None else service_time_mean
        )
        arrivals.append(float(actual["arrival_times"][0, node]))
        if problem_name in CAPACITY_PROBLEMS:
            demand.append(float(actual["demand"][0, node]))
        if problem_name in TW_PROBLEMS:
            windows.append(
                (
                    float(actual["window_starts"][0, node]),
                    float(actual["window_ends"][0, node]),
                )
            )
        else:
            windows.append((0.0, 10**6))

    distance = actual["distance_matrix"][0].numpy()
    return {
        "distance": distance[np.ix_(source_ids, source_ids)],
        "service": np.asarray(service),
        "arrival_times": np.asarray(arrivals),
        "speed": float(actual["speed"].item()),
        "mapping": mapping,
        "windows": windows,
        "demand": np.asarray(demand),
        "capacity": capacity,
        "reload_nodes": set(reload_nodes),
        "gamma": float(actual["gamma"].item()) if problem_name in TW_PROBLEMS else 0.0,
        "theta": float(actual["theta"].item()) if problem_name in TW_PROBLEMS else 1.0,
    }


def score(problem, problem_name, actual, tour, visit_times=None):
    if visit_times is None:
        visit_times, _ = problem.get_times(actual, tour)
    if problem_name in TW_PROBLEMS:
        value = problem.get_costs(
            actual, tour, visit_times, actual["gamma"], actual["theta"]
        )[0]
    else:
        value = problem.get_costs(actual, tour)[0]
    return float(value.item())


def parse_args(argv=None, *, method=None):
    """Parse shared options, exposing only the selected method's solver limits."""
    descriptions = {
        "online": "Evaluate online OR-Tools reoptimization using revealed requests.",
        "oracle": "Evaluate the clairvoyant OR-Tools oracle using all requests.",
        None: "Evaluate online and clairvoyant OR-Tools baselines together.",
    }
    parser = argparse.ArgumentParser(description=descriptions[method])
    add_dataset_time_arguments(parser)
    add_solver_budget_arguments(parser)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-ortec")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0, help="First zero-based dataset instance to evaluate")
    parser.add_argument(
        "--max-trips", type=int, default=9,
        help="Legacy compatibility option; depot returns follow the unserved customer count, not a trip cap",
    )
    if method != "oracle":
        add_service_time_arguments(parser)
    if method is None:
        parser.add_argument(
            "--method", choices=("both", "online", "oracle"), default="both",
            help="Evaluate both baselines or only the selected one (default: both)",
        )
    else:
        parser.set_defaults(method=method)
    prefixes = ("online", "static") if method is None else (("online",) if method == "online" else ("static",))
    for prefix in prefixes:
        label = "online reoptimization" if prefix == "online" else "clairvoyant oracle"
        parser.add_argument(
            f"--{prefix}-no-improvement-ms", type=int,
            help=f"Stop each {label} solve after this many milliseconds without a new best objective (default: 60000)",
        )
        parser.add_argument(
            f"--{prefix}-time-ms", type=int,
            help=f"Optional total time limit per {label} solve in milliseconds; unset by default",
        )
    args = parser.parse_args(argv)
    validate_dataset_time_arguments(parser, args)
    validate_solver_budget_arguments(parser, args, prefixes)
    if min(args.num_samples, args.max_trips) < 1:
        parser.error("sample count and max-trips must be positive")
    if args.offset < 0:
        parser.error("offset must be nonnegative")
    if method != "oracle":
        validate_service_time_arguments(parser, args)
    return args


def run_evaluation(args, *, online_evaluator=None, static_evaluator=None):
    """Evaluate the selected methods and write their established result formats."""

    if getattr(args, "solve_seconds", None) is not None:
        prefixes = {"online": ("online",), "oracle": ("static",),
                    "both": ("online", "static")}[args.method]
        return run_solver_budget_sweep(
            args,
            lambda run_args: run_evaluation(
                run_args, online_evaluator=online_evaluator, static_evaluator=static_evaluator
            ),
            prefixes,
        )

    problem = load_problem(args.problem)
    prior_options = {}
    if args.method in ("both", "online"):
        prior_options = {
            "service_times_mean": args.service_times_mean,
            "service_times_var": args.service_times_var,
        }
    dataset = problem.make_dataset(
        filename=args.dataset,
        batch_size=1,
        num_samples=args.num_samples,
        offset=args.offset,
        use_ortec=args.use_ortec,
        **dataset_time_options(args),
        **prior_options,
    )
    if not len(dataset):
        raise ValueError(f"No samples found in dataset {args.dataset} at offset {args.offset}")
    rows = []
    online_termination_reasons = Counter()
    run_online = args.method in ("both", "online")
    run_static = args.method in ("both", "oracle")
    if (run_online and online_evaluator is None) or (run_static and static_evaluator is None):
        raise ValueError("An evaluator is required for each selected method")
    for index, item in enumerate(dataset):
        actual = dataset_to_input(item, args.problem, torch.device("cpu"))
        row = {"sample": args.offset + index}
        if run_online:
            online, online_runtime, online_tour, online_details = online_evaluator(
                args, problem, actual, return_details=True
            )
            online_termination_reasons.update(online_details["termination_reasons"])
            row.update({
                "online_objective": online,
                "online_runtime_seconds": online_runtime,
                "online_solve_count": online_details["solve_count"],
                "online_termination_reasons": json.dumps(
                    online_details["termination_reasons"], sort_keys=True
                ),
                "online_improvement_count": online_details["improvement_count"],
                "online_tour": json.dumps(online_tour),
            })
        if run_static:
            static, static_runtime, static_tour, static_details = static_evaluator(
                args, problem, actual, return_details=True
            )
            row.update({
                "static_objective": static,
                "static_runtime_seconds": static_runtime,
                "static_solver_objective": static_details["solver_objective"],
                "static_solver_status": static_details["solver_status"],
                "static_termination_reason": static_details["termination_reason"],
                "static_improvement_count": static_details["improvement_count"],
                "static_tour": json.dumps(static_tour),
            })
        if args.method == "both":
            row["gap_percent"] = (online / static - 1.0) * 100.0
            # Preserve the existing paired-evaluation CSV column order.
            row = {name: row[name] for name in (
                "sample", "online_objective", "static_objective", "gap_percent",
                "online_runtime_seconds", "online_solve_count", "online_termination_reasons",
                "online_improvement_count", "static_runtime_seconds", "static_solver_objective",
                "static_solver_status", "static_termination_reason", "static_improvement_count",
                "online_tour", "static_tour",
            )}
        rows.append(row)

    summary = {
        "problem": args.problem,
        "dataset_parameters": dataset_time_options(args),
        "dataset": args.dataset,
        "offset": args.offset,
        "method": args.method,
        "samples": len(rows),
        "depot_return_policy": (
            DEPOT_RETURN_POLICY if args.problem in CAPACITY_PROBLEMS else "not_applicable"
        ),
        "initial_route_policy": initial_route_policy(args.problem),
    }
    if getattr(args, "solve_budget_seconds", None) is not None:
        summary["solve_seconds"] = args.solve_budget_seconds
    if run_online:
        summary.update({
            "online_service_time_model": "expected",
            "service_times_mean": args.service_times_mean,
            "service_times_var": args.service_times_var,
            "online_time_ms": args.online_time_ms,
            "online_no_improvement_ms": args.online_no_improvement_ms,
            "online_stopping_rule": (
                "time_limit" if args.online_no_improvement_ms is None else "no_improvement"
            ),
            "online_termination_reasons": dict(sorted(online_termination_reasons.items())),
            "online_objective_mean": float(np.mean([r["online_objective"] for r in rows])),
            "online_objective_std": float(np.std([r["online_objective"] for r in rows])),
            "online_runtime_mean_seconds": float(
                np.mean([r["online_runtime_seconds"] for r in rows])
            ),
        })
    if run_static:
        summary.update({
            "static_time_ms": args.static_time_ms,
            "static_no_improvement_ms": args.static_no_improvement_ms,
            "static_stopping_rule": (
                "time_limit" if args.static_no_improvement_ms is None else "no_improvement"
            ),
            "static_termination_reasons": {
                reason: sum(row["static_termination_reason"] == reason for row in rows)
                for reason in sorted({row["static_termination_reason"] for row in rows})
            },
            "static_objective_mean": float(np.mean([r["static_objective"] for r in rows])),
            "static_objective_std": float(np.std([r["static_objective"] for r in rows])),
            "static_runtime_mean_seconds": float(
                np.mean([r["static_runtime_seconds"] for r in rows])
            ),
        })
    if args.method == "both":
        summary["gap_percent"] = (
            summary["online_objective_mean"] / summary["static_objective_mean"]
            - 1.0
        ) * 100.0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
