#!/usr/bin/env python
"""Evaluate simple online routing heuristics."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from utils import load_problem
from utils.data_utils import dataset_to_input
from utils.dataset_parameters import add_dataset_time_arguments, dataset_time_options, validate_dataset_time_arguments


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_dataset_time_arguments(parser)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-ortec")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--method",
        choices=("nearest_neighbour", "first_come_first_served"),
        default="nearest_neighbour",
    )
    args = parser.parse_args(argv)
    validate_dataset_time_arguments(parser, args)
    return args


def cost(problem, problem_name, model_input, tour, visit_times):
    if problem_name in ("pdtrptw", "pdctrptw"):
        return problem.get_costs(
            model_input,
            tour,
            visit_times,
            model_input["gamma"],
            model_input["theta"],
        )[0]
    return problem.get_costs(model_input, tour)[0]


def nearest_neighbour(problem, model_input):
    """Visit the nearest feasible customer, reloading only when necessary."""
    state = problem.make_state(model_input)
    tour = [0]
    while not state.all_finished():
        feasible = ~state.get_mask()[0, 0]
        feasible_customers = feasible.clone()
        feasible_customers[0] = False
        distances = model_input["distance_matrix"][
            0, state.prev_a.item()
        ].clone()
        if feasible_customers.any():
            distances[~feasible_customers] = float("inf")
            selected = torch.argmin(distances).view(1)
        elif feasible[0]:
            selected = torch.zeros(1, dtype=torch.long)
        else:
            raise RuntimeError("Environment has unfinished work but no feasible action")
        state = state.update(selected)
        if not hasattr(state, "last_action_executed") or state.last_action_executed.item():
            tour.append(selected.item())
    tour_tensor = torch.tensor(tour, dtype=torch.long).unsqueeze(0)
    if hasattr(state, "get_visit_times"):
        visit_times = torch.tensor(state.get_visit_times())
    else:
        visit_times, _ = problem.get_times(model_input, tour_tensor)
    return tour_tensor, visit_times


def first_come_first_served(problem, model_input):
    """Serve revealed customers in arrival order, reloading when necessary.

    Customers with equal arrival times are ordered by their node index. In a
    capacitated environment, the vehicle returns to the depot if the customer
    at the head of the FCFS queue does not fit in its remaining capacity.
    """
    state = problem.make_state(model_input)
    tour = [0]
    arrival_times = model_input["arrival_times"][0]

    while not state.all_finished():
        feasible = ~state.get_mask()[0, 0]
        known_unvisited = ~(
            state.visited_[0, 0] | state.not_arrived_[0, 0]
        )
        known_unvisited[0] = False

        if not known_unvisited.any():
            raise RuntimeError(
                "Environment has unfinished work but no revealed customer"
            )

        priorities = arrival_times.clone()
        priorities[~known_unvisited] = float("inf")
        queue_head = torch.argmin(priorities)

        if feasible[queue_head]:
            selected = queue_head.view(1)
        elif feasible[0]:
            # The queue head is known but does not fit in the remaining
            # capacity. Reload rather than serving a later arrival first.
            selected = torch.zeros(1, dtype=torch.long)
        else:
            raise RuntimeError(
                "FCFS queue head is infeasible and the depot is unavailable"
            )

        state = state.update(selected)
        if not hasattr(state, "last_action_executed") or state.last_action_executed.item():
            tour.append(selected.item())

    tour_tensor = torch.tensor(tour, dtype=torch.long).unsqueeze(0)
    if hasattr(state, "get_visit_times"):
        visit_times = torch.tensor(state.get_visit_times())
    else:
        visit_times, _ = problem.get_times(model_input, tour_tensor)
    return tour_tensor, visit_times


def main():
    args = parse_args()
    if (
        args.method == "first_come_first_served"
        and args.problem not in ("pdtrptw", "pdctrptw")
    ):
        raise ValueError(
            "FCFS evaluation is intended for pdtrptw and pdctrptw"
        )
    problem = load_problem(args.problem)
    dataset_kwargs = {
        "batch_size": 1,
        "num_samples": args.num_samples,
        "use_ortec": args.use_ortec,
        **dataset_time_options(args),
    }
    dataset = problem.make_dataset(filename=args.dataset, **dataset_kwargs)
    heuristic_costs = []
    heuristic_times = []
    heuristic = {
        "nearest_neighbour": nearest_neighbour,
        "first_come_first_served": first_come_first_served,
    }[args.method]
    for index in range(len(dataset)):
        item = dataset[index]
        model_input = dataset_to_input(
            item, args.problem, torch.device("cpu")
        )
        started = time.perf_counter()
        tour, visit_times = heuristic(problem, model_input)
        heuristic_times.append(time.perf_counter() - started)
        heuristic_costs.append(
            cost(
                problem, args.problem, model_input, tour, visit_times
            ).item()
        )

    rows = []
    for index in range(len(dataset)):
        rows.append(
            {
                "sample": index,
                "objective": heuristic_costs[index],
                "runtime_seconds": heuristic_times[index],
            }
        )

    summary = {
        "problem": args.problem,
        "dataset": args.dataset,
        "method": args.method,
        "dataset_parameters": dataset_time_options(args),
        "samples": len(heuristic_costs),
        "objective_mean": float(np.mean(heuristic_costs)),
        "objective_std": float(np.std(heuristic_costs)),
        "runtime_mean_seconds": float(np.mean(heuristic_times)),
        "runtime_std_seconds": float(np.std(heuristic_times)),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
