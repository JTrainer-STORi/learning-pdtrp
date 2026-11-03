#!/usr/bin/env python
"""Evaluate the clairvoyant OR-Tools oracle with all requests known in advance."""

import time

import torch

from evaluate_msa_ortools import solve_scenario
from utils.ortools_baselines import (
    parse_args as parse_baseline_args,
    run_evaluation,
    scenario_from_nodes,
    score,
)


def evaluate_static(args, problem, actual, *, return_details=False):
    node_ids = list(range(1, actual["all_nodes"].shape[1]))
    scenario = scenario_from_nodes(
        args.problem, actual, 0, node_ids, 0.0, args.max_trips
    )
    started = time.perf_counter()
    details = solve_scenario(
        scenario,
        args.problem,
        args.static_time_ms,
        no_improvement_ms=args.static_no_improvement_ms,
        return_details=True,
    )
    route = details["route"]
    runtime = time.perf_counter() - started
    mapped = [
        scenario["mapping"][node]
        for node in route
        if scenario["mapping"][node] is not None
    ]
    visited_customers = sorted(node for node in mapped if node != 0)
    if visited_customers != node_ids:
        raise RuntimeError("Static OR-Tools solve did not visit every customer")
    tour = [0, *mapped]
    tour_tensor = torch.tensor([tour], dtype=torch.long)
    result = (score(problem, args.problem, actual, tour_tensor), runtime, tour)
    return (*result, details) if return_details else result


def parse_args(argv=None):
    return parse_baseline_args(argv, method="oracle")


def main(argv=None):
    return run_evaluation(parse_args(argv), static_evaluator=evaluate_static)


if __name__ == "__main__":
    main()
