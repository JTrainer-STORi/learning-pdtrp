#!/usr/bin/env python
"""Recompute corrected online and clairvoyant OR-Tools baselines."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_msa_ortools import (
    CAPACITY_PROBLEMS,
    TW_PROBLEMS,
    solve_scenario,
)
from utils import load_problem
from utils.data_utils import dataset_to_input


def scenario_from_nodes(
    problem_name, actual, current_node, node_ids, used_capacity, max_trips
):
    """Build an OR-Tools problem using only the supplied actual customer IDs."""
    source_ids = [current_node]
    mapping = [None]
    service = [0.0]
    demand = [used_capacity] if problem_name in CAPACITY_PROBLEMS else []
    windows = [(0.0, 10**6)]
    reload_nodes = []

    if problem_name in CAPACITY_PROBLEMS:
        for _ in range(max_trips):
            reload_nodes.append(len(source_ids))
            source_ids.append(0)
            mapping.append(0)
            service.append(0.0)
            demand.append(-1.0)
            windows.append((0.0, 10**6))

    current_time = 0.0
    for node in node_ids:
        source_ids.append(node)
        mapping.append(node)
        service.append(float(actual["service_times"][0, node]))
        if problem_name in CAPACITY_PROBLEMS:
            demand.append(float(actual["demand"][0, node]))
        if problem_name in TW_PROBLEMS:
            windows.append(
                (
                    float(actual["window_starts"][0, node]) - current_time,
                    float(actual["window_ends"][0, node]) - current_time,
                )
            )
        else:
            windows.append((0.0, 10**6))

    distance = actual["distance_matrix"][0].numpy()
    return {
        "distance": distance[np.ix_(source_ids, source_ids)],
        "service": np.asarray(service),
        "speed": float(actual["speed"].item()),
        "mapping": mapping,
        "windows": windows,
        "demand": np.asarray(demand),
        "capacity": 1.0,
        "reload_nodes": set(reload_nodes),
        "gamma": float(actual["gamma"].item()) if problem_name in TW_PROBLEMS else 0.0,
        "theta": float(actual["theta"].item()) if problem_name in TW_PROBLEMS else 1.0,
    }


def online_scenario(problem_name, actual, state, max_trips):
    visited = state.visited_[0, 0]
    arrived = ~state.not_arrived_[0, 0]
    node_ids = torch.nonzero((~visited) & arrived).flatten().tolist()
    node_ids = [node for node in node_ids if node != 0]
    scenario = scenario_from_nodes(
        problem_name,
        actual,
        int(state.prev_a.item()),
        node_ids,
        float(state.used_cap.item()) if problem_name in CAPACITY_PROBLEMS else 0.0,
        max_trips,
    )
    if problem_name in TW_PROBLEMS:
        current_time = float(state.cur_times.item())
        scenario["windows"] = [
            scenario["windows"][0],
            *[
                (start - current_time, end - current_time)
                for start, end in scenario["windows"][1:]
            ],
        ]
    return scenario


def score(problem, problem_name, actual, tour):
    visit_times, _ = problem.get_times(actual, tour)
    if problem_name in TW_PROBLEMS:
        value = problem.get_costs(
            actual, tour, visit_times, actual["gamma"], actual["theta"]
        )[0]
    else:
        value = problem.get_costs(actual, tour)[0]
    return float(value.item())


def evaluate_online(args, problem, actual):
    state = problem.make_state(actual)
    tour = [0]
    runtime = 0.0
    while not state.all_finished():
        feasible = ~state.get_mask()[0, 0]
        feasible_customers = feasible.clone()
        feasible_customers[0] = False
        if not feasible_customers.any():
            if not feasible[0]:
                raise RuntimeError("Unfinished online instance has no feasible action")
            selected = 0
        else:
            scenario = online_scenario(
                args.problem, actual, state, args.max_trips
            )
            started = time.perf_counter()
            route = solve_scenario(
                scenario, args.problem, args.online_time_ms
            )
            runtime += time.perf_counter() - started
            selected = next(
                (
                    scenario["mapping"][node]
                    for node in route
                    if scenario["mapping"][node] not in (None, 0)
                    and feasible[scenario["mapping"][node]].item()
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    "Online OR-Tools solve returned no feasible customer action"
                )
        state = state.update(torch.tensor([selected], dtype=torch.long))
        tour.append(selected)
    tour_tensor = torch.tensor([tour], dtype=torch.long)
    return score(problem, args.problem, actual, tour_tensor), runtime, tour


def evaluate_static(args, problem, actual):
    node_ids = list(range(1, actual["all_nodes"].shape[1]))
    scenario = scenario_from_nodes(
        args.problem, actual, 0, node_ids, 0.0, args.max_trips
    )
    started = time.perf_counter()
    route = solve_scenario(scenario, args.problem, args.static_time_ms)
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
    return score(problem, args.problem, actual, tour_tensor), runtime, tour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-ortec")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--online-time-ms", type=int, default=100)
    parser.add_argument("--static-time-ms", type=int, default=3000)
    parser.add_argument("--max-trips", type=int, default=9)
    args = parser.parse_args()
    if min(args.num_samples, args.online_time_ms, args.static_time_ms, args.max_trips) < 1:
        parser.error("sample count, time limits, and max-trips must be positive")

    problem = load_problem(args.problem)
    dataset = problem.make_dataset(
        filename=args.dataset,
        batch_size=1,
        num_samples=args.num_samples,
        use_ortec=args.use_ortec,
    )
    rows = []
    for index, item in enumerate(dataset):
        actual = dataset_to_input(item, args.problem, torch.device("cpu"))
        online, online_runtime, online_tour = evaluate_online(args, problem, actual)
        static, static_runtime, static_tour = evaluate_static(args, problem, actual)
        rows.append(
            {
                "sample": index,
                "online_objective": online,
                "static_objective": static,
                "gap_percent": (online / static - 1.0) * 100.0,
                "online_runtime_seconds": online_runtime,
                "static_runtime_seconds": static_runtime,
                "online_tour": json.dumps(online_tour),
                "static_tour": json.dumps(static_tour),
            }
        )

    summary = {
        "problem": args.problem,
        "dataset": args.dataset,
        "samples": len(rows),
        "online_time_ms": args.online_time_ms,
        "static_time_ms": args.static_time_ms,
        "online_objective_mean": float(np.mean([r["online_objective"] for r in rows])),
        "online_objective_std": float(np.std([r["online_objective"] for r in rows])),
        "static_objective_mean": float(np.mean([r["static_objective"] for r in rows])),
        "static_objective_std": float(np.std([r["static_objective"] for r in rows])),
        "gap_percent": (
            float(np.mean([r["online_objective"] for r in rows]))
            / float(np.mean([r["static_objective"] for r in rows]))
            - 1.0
        )
        * 100.0,
        "online_runtime_mean_seconds": float(
            np.mean([r["online_runtime_seconds"] for r in rows])
        ),
        "static_runtime_mean_seconds": float(
            np.mean([r["static_runtime_seconds"] for r in rows])
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
