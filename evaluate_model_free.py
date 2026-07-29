#!/usr/bin/env python
"""Evaluate stored OR-Tools tours and a nearest-neighbour heuristic."""

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
    parser.add_argument("--static-dataset", required=True)
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


def stored_costs(problem, problem_name, dataset):
    values = []
    for index in range(len(dataset)):
        item = dataset[index]
        model_input = dataset_to_input(item, problem_name, torch.device("cpu"))
        tour = item["tour_nodes"].unsqueeze(0)
        visit_times, _ = problem.get_times(model_input, tour)
        values.append(cost(problem, problem_name, model_input, tour, visit_times).item())
    return values


def nearest_neighbour(problem, model_input):
    state = problem.make_state(model_input)
    tour = [0]
    while not state.all_finished():
        distances = model_input["distance_matrix"][
            0, state.prev_a.item()
        ].clone()
        distances[state.get_mask()[0, 0]] = float("inf")
        selected = torch.argmin(distances).view(1)
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
    static_dataset = problem.make_dataset(
        filename=args.static_dataset, **dataset_kwargs
    )

    ortools_costs = stored_costs(problem, args.problem, dataset)
    static_costs = stored_costs(problem, args.problem, static_dataset)
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

    static_mean = float(np.mean(static_costs))
    rows = []
    for index in range(len(dataset)):
        rows.extend(
            (
                {
                    "sample": index,
                    "method": "ortools",
                    "objective": ortools_costs[index],
                    "static_objective": static_costs[index],
                    "runtime_seconds": "",
                },
                {
                    "sample": index,
                    "method": "heuristic",
                    "objective": heuristic_costs[index],
                    "static_objective": static_costs[index],
                    "runtime_seconds": heuristic_times[index],
                },
            )
        )

    summaries = []
    for method, objectives, runtimes in (
        ("ortools", ortools_costs, None),
        ("heuristic", heuristic_costs, heuristic_times),
    ):
        objective_mean = float(np.mean(objectives))
        summaries.append(
            {
                "problem": args.problem,
                "dataset": args.dataset,
                "method": method,
                "samples": len(objectives),
                "objective_mean": objective_mean,
                "objective_std": float(np.std(objectives)),
                "static_objective_mean": static_mean,
                "gap_percent": (objective_mean / static_mean - 1) * 100,
                "runtime_mean_seconds": (
                    float(np.mean(runtimes)) if runtimes is not None else None
                ),
                "runtime_std_seconds": (
                    float(np.std(runtimes)) if runtimes is not None else None
                ),
            }
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.json").open("w") as file:
        json.dump(summaries, file, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
