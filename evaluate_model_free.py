#!/usr/bin/env python
"""Evaluate the online nearest-neighbour heuristic."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from utils import load_problem
from utils.data_utils import dataset_to_input


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-ortec")
    parser.add_argument("--num-samples", type=int, default=100)
    return parser.parse_args()


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
        tour.append(selected.item())
    tour_tensor = torch.tensor(tour, dtype=torch.long).unsqueeze(0)
    visit_times, _ = problem.get_times(model_input, tour_tensor)
    return tour_tensor, visit_times


def main():
    args = parse_args()
    problem = load_problem(args.problem)
    dataset_kwargs = {
        "batch_size": 1,
        "num_samples": args.num_samples,
        "use_ortec": args.use_ortec,
    }
    dataset = problem.make_dataset(filename=args.dataset, **dataset_kwargs)
    heuristic_costs = []
    heuristic_times = []
    for index in range(len(dataset)):
        item = dataset[index]
        model_input = dataset_to_input(
            item, args.problem, torch.device("cpu")
        )
        started = time.perf_counter()
        tour, visit_times = nearest_neighbour(problem, model_input)
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
        "method": "nearest_neighbour",
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
