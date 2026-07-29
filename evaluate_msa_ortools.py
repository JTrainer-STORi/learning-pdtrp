#!/usr/bin/env python
"""Event-driven multiple-scenario OR-Tools baseline.

Uniform scenarios draw new coordinates. ORTEC scenarios draw customers from the
fixed source instance and use its asymmetric road-duration matrix.
"""

import argparse
import json
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from utils import load_problem
from utils.data_utils import dataset_to_input
from problems.utils import generate_scaled_demands, read_vrplib


TW_PROBLEMS = {"pdtrptw", "pdctrptw"}
CAPACITY_PROBLEMS = {"pdctrp", "pdctrptw"}


def make_ortec_context(filename):
    """Load the fixed ORTEC customer pool in the same units as saved datasets."""
    source = read_vrplib(filename)
    duration = source["duration_matrix"].astype(float)
    return {
        "filename": filename,
        "coords": source["coords"].astype(float),
        "distance": duration / duration.max(),
        "demands": source["demands"].astype(float),
        "service_times": source["service_times"].astype(float),
        "time_windows": source["time_windows"].astype(float),
    }


def map_ortec_nodes(item, ortec):
    """Recover source customer IDs and verify them against the saved matrix.

    Generated datasets round coordinates and distances to three decimals.  A
    combination of coordinates and both directed depot arcs distinguishes the
    source nodes; the complete submatrix check guards against a silent mismatch.
    """
    coordinates = np.asarray(item["all_nodes"], dtype=float)
    distance = np.asarray(item["distance_matrix"], dtype=float)
    source_coordinates = ortec["coords"]
    source_distance = ortec["distance"]
    if len(coordinates) == 0 or np.any(coordinates[0] == 0):
        raise ValueError("Cannot recover ORTEC coordinate scaling from depot")
    scale = source_coordinates[0] / coordinates[0]
    recovered_coordinates = coordinates * scale
    source_scale = np.maximum(np.abs(scale), 1.0)
    selected = []
    used = set()
    for node in range(len(coordinates)):
        score = np.linalg.norm(
            (source_coordinates - recovered_coordinates[node]) / source_scale,
            axis=1,
        )
        score += np.abs(source_distance[0] - distance[0, node])
        score += np.abs(source_distance[:, 0] - distance[node, 0])
        if used:
            score[list(used)] = np.inf
        source_id = int(np.argmin(score))
        selected.append(source_id)
        used.add(source_id)
    if selected[0] != 0:
        raise ValueError("Saved ORTEC depot did not map to source depot")
    reconstructed = source_distance[np.ix_(selected, selected)]
    error = float(np.max(np.abs(reconstructed - distance)))
    if error > 0.0011:
        raise ValueError(
            f"Could not reliably map saved ORTEC nodes (matrix error {error:.6f})"
        )
    return selected


def sample_ortec_scenario(
    ortec, problem_name, n_total, dod, min_trips, max_trips
):
    """Sample once from the fixed ORTEC pool without reparsing the VRPLIB file."""
    unsorted_source_ids = np.concatenate(
        ([0], np.random.choice(np.arange(1, len(ortec["coords"])), n_total - 1, replace=False))
    )
    coordinates = ortec["coords"][unsorted_source_ids]
    coordinates = coordinates / coordinates.max(axis=0)
    service = ortec["service_times"][unsorted_source_ids] / 60.0
    customer_windows = (
        ortec["time_windows"][unsorted_source_ids[1:]]
        * 482.0
        / np.max(ortec["time_windows"][1:])
    )
    n_immediate = int(dod * n_total)
    immediate = np.random.choice(n_total - 1, n_immediate, replace=False)
    arrivals = np.zeros(n_total)
    arrivals[immediate + 1] = customer_windows[immediate, 0] - 60.0
    order = np.argsort(arrivals, kind="stable")
    source_ids = unsorted_source_ids[order]
    result = {
        "all_nodes": torch.tensor(np.round(coordinates[order], 3), dtype=torch.float),
        "service_times": torch.tensor(np.round(service[order], 3), dtype=torch.float),
        "arrival_times": torch.tensor(np.round(arrivals[order], 3), dtype=torch.float),
        "source_ids": source_ids.tolist(),
    }
    if problem_name in TW_PROBLEMS:
        windows = np.concatenate(([[0.0, 0.0]], customer_windows), axis=0)[order]
        result["window_starts"] = torch.tensor(np.round(windows[:, 0], 3))
        result["window_ends"] = torch.tensor(np.round(windows[:, 1], 3))
    if problem_name in CAPACITY_PROBLEMS:
        customer_demands = generate_scaled_demands(
            1,
            n_total - 1,
            1.0,
            min_trips,
            max_trips,
            raw_demands=ortec["demands"][unsorted_source_ids[1:]][None, :],
        )[0]
        unsorted_demands = np.concatenate(([0.0], customer_demands))
        result["demand"] = torch.tensor(unsorted_demands[order])
    return result


def solve_scenario(data, problem_name, time_limit_ms):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    size = len(data["distance"])
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    time_scale = 1000
    weight_scale = 1000

    def transit(from_index, to_index):
        source = manager.IndexToNode(from_index)
        target = manager.IndexToNode(to_index)
        return int(
            round(
                (
                    data["distance"][source, target] / data["speed"]
                    + data["service"][source]
                )
                * time_scale
            )
        )

    transit_index = routing.RegisterTransitCallback(transit)

    def objective(from_index, to_index):
        source = manager.IndexToNode(from_index)
        target = manager.IndexToNode(to_index)
        return int(
            round(
                data["theta"]
                * data["distance"][source, target]
                * time_scale
                * weight_scale
            )
        )

    objective_index = routing.RegisterTransitCallback(objective)
    routing.SetArcCostEvaluatorOfAllVehicles(objective_index)

    if problem_name in TW_PROBLEMS:
        routing.AddDimension(transit_index, 10**9, 10**9, True, "Time")
        dimension = routing.GetDimensionOrDie("Time")
        for node, (start, end) in enumerate(data["windows"]):
            if node == 0:
                continue
            index = manager.NodeToIndex(node)
            dimension.CumulVar(index).SetMin(max(0, int(round(start * time_scale))))
            dimension.SetCumulVarSoftUpperBound(
                index,
                max(0, int(round(end * time_scale))),
                max(0, int(round(data["gamma"] * weight_scale))),
            )

    if problem_name in CAPACITY_PROBLEMS:
        scaled_demands = np.rint(data["demand"] * 1000).astype(int)

        def demand(index):
            return int(scaled_demands[manager.IndexToNode(index)])

        demand_index = routing.RegisterUnaryTransitCallback(demand)
        capacity = int(round(data["capacity"] * 1000))
        routing.AddDimension(
            demand_index, capacity, capacity, True, "Capacity"
        )
        dimension = routing.GetDimensionOrDie("Capacity")
        for node in data["reload_nodes"]:
            index = manager.NodeToIndex(node)
            routing.AddDisjunction([index], 0)
        for node in range(1, size):
            if node not in data["reload_nodes"]:
                dimension.SlackVar(manager.NodeToIndex(node)).SetValue(0)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.FromMilliseconds(time_limit_ms)
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        return []
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return route[1:]


def make_scenario(
    problem,
    problem_name,
    actual,
    state,
    n_total,
    dod,
    min_trips,
    max_trips,
    ortec=None,
    actual_source_ids=None,
    n_subregions=1,
    arrival_weights=None,
    arrival_skews=None,
):
    if ortec is not None:
        sampled = sample_ortec_scenario(
            ortec, problem_name, n_total, dod, min_trips, max_trips
        )
    else:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            sampled = problem.make_dataset(
                min_total=n_total,
                max_total=n_total,
                min_dod=dod,
                max_dod=dod,
                batch_size=1,
                num_samples=1,
                n_subregions=n_subregions,
                arrival_weights=arrival_weights,
                arrival_skews=arrival_skews,
                vehicle_capacity=1.0,
                min_trips_required_lb=min_trips,
                min_trips_required_ub=max_trips,
            )[0]
    current_time = float(state.cur_times.item())
    visited = state.visited_[0, 0]
    arrived = ~state.not_arrived_[0, 0]
    known_ids = torch.nonzero((~visited) & arrived).flatten().tolist()
    known_ids = [node for node in known_ids if node != 0]
    future_ids = torch.nonzero(
        sampled["arrival_times"] > current_time
    ).flatten().tolist()
    sampled_source_ids = None
    if ortec is not None:
        sampled_source_ids = sampled["source_ids"]
        # Revealed customers are known not to be future requests. Filtering
        # conditions the sampled scenario on information available so far.
        revealed_source_ids = {
            actual_source_ids[node]
            for node in torch.nonzero(arrived).flatten().tolist()
        }
        future_ids = [
            node
            for node in future_ids
            if sampled_source_ids[node] not in revealed_source_ids
        ]

    coords = [actual["all_nodes"][0, state.prev_a.item()]]
    service = [torch.tensor(0.0)]
    mapping = [None]
    source_ids = (
        [actual_source_ids[state.prev_a.item()]] if ortec is not None else None
    )
    reload_nodes = []
    demand = [float(state.used_cap.item())] if problem_name in CAPACITY_PROBLEMS else []
    windows = [(0.0, 10**6)]

    if problem_name in CAPACITY_PROBLEMS:
        for _ in range(max_trips):
            reload_nodes.append(len(coords))
            coords.append(actual["all_nodes"][0, 0])
            service.append(torch.tensor(0.0))
            mapping.append(0)
            if ortec is not None:
                source_ids.append(0)
            demand.append(-1.0)
            windows.append((0.0, 10**6))

    for node in known_ids:
        coords.append(actual["all_nodes"][0, node])
        service.append(actual["service_times"][0, node])
        mapping.append(node)
        if ortec is not None:
            source_ids.append(actual_source_ids[node])
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

    for node in future_ids:
        coords.append(sampled["all_nodes"][node])
        service.append(sampled["service_times"][node])
        mapping.append(None)
        if ortec is not None:
            source_ids.append(sampled_source_ids[node])
        if problem_name in CAPACITY_PROBLEMS:
            demand.append(float(sampled["demand"][node]))
        if problem_name in TW_PROBLEMS:
            windows.append(
                (
                    max(
                        float(sampled["window_starts"][node]),
                        float(sampled["arrival_times"][node]),
                    )
                    - current_time,
                    float(sampled["window_ends"][node]) - current_time,
                )
            )
        else:
            windows.append((0.0, 10**6))

    coordinates = torch.stack(coords)
    if ortec is not None:
        scenario_distance = ortec["distance"][np.ix_(source_ids, source_ids)]
    else:
        scenario_distance = torch.cdist(coordinates, coordinates).numpy()
    return {
        "distance": scenario_distance,
        "service": torch.stack(service).numpy(),
        "speed": float(actual["speed"].item()),
        "mapping": mapping,
        "windows": windows,
        "demand": np.asarray(demand),
        "capacity": 1.0,
        "reload_nodes": set(reload_nodes),
        "gamma": float(actual["gamma"].item()) if problem_name in TW_PROBLEMS else 0.0,
        "theta": float(actual["theta"].item()) if problem_name in TW_PROBLEMS else 1.0,
    }


def evaluate_instance(args, problem, item, ortec=None):
    actual = dataset_to_input(item, args.problem, torch.device("cpu"))
    actual_source_ids = map_ortec_nodes(item, ortec) if ortec is not None else None
    state = problem.make_state(actual)
    tour = [0]
    solve_time = 0.0
    while not state.all_finished():
        feasible = ~state.get_mask()[0, 0]
        feasible[0] = False
        if not feasible.any():
            selected = torch.tensor([0])
        else:
            votes = []
            for _ in range(args.scenarios):
                scenario = make_scenario(
                    problem,
                    args.problem,
                    actual,
                    state,
                    len(item["all_nodes"]),
                    args.dod,
                    args.min_trips,
                    args.max_trips,
                    ortec,
                    actual_source_ids,
                    args.n_subregions,
                    args.arrival_weights,
                    args.arrival_skews,
                )
                started = time.perf_counter()
                route = solve_scenario(
                    scenario, args.problem, args.solver_time_ms
                )
                solve_time += time.perf_counter() - started
                action = next(
                    (
                        scenario["mapping"][node]
                        for node in route
                        if scenario["mapping"][node] is not None
                        and scenario["mapping"][node] != 0
                        and feasible[scenario["mapping"][node]].item()
                    ),
                    0,
                )
                votes.append(action)
            selected = torch.tensor(
                [Counter(votes).most_common(1)[0][0]], dtype=torch.long
            )
        state = state.update(selected)
        tour.append(selected.item())

    pi = torch.tensor(tour, dtype=torch.long).unsqueeze(0)
    visit_times, _ = problem.get_times(actual, pi)
    if args.problem in TW_PROBLEMS:
        objective = problem.get_costs(
            actual, pi, visit_times, actual["gamma"], actual["theta"]
        )[0].item()
    else:
        objective = problem.get_costs(actual, pi)[0].item()
    return objective, solve_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dod", type=float, required=True)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--solver-time-ms", type=int, default=250)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-trips", type=int, default=3)
    parser.add_argument("--max-trips", type=int, default=9)
    parser.add_argument(
        "--use-ortec",
        help="ORTEC VRPLIB source; scenarios sample its fixed customer pool",
    )
    parser.add_argument(
        "--n-subregions",
        type=int,
        default=1,
        help="Spatial regions used for scenario sampling (1 is uniform on the unit square)",
    )
    parser.add_argument("--arrival-weights", nargs="+", type=int)
    parser.add_argument(
        "--arrival-skews",
        nargs="+",
        choices=("early", "uniform", "late"),
    )
    args = parser.parse_args()
    if args.scenarios < 1 or args.solver_time_ms < 1:
        parser.error("scenarios and solver-time-ms must be positive")
    if args.arrival_weights is not None and len(args.arrival_weights) != args.n_subregions:
        parser.error("arrival-weights must have one value per subregion")
    if args.arrival_skews is not None and len(args.arrival_skews) != args.n_subregions:
        parser.error("arrival-skews must have one value per subregion")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    problem = load_problem(args.problem)
    ortec = make_ortec_context(args.use_ortec) if args.use_ortec else None
    dataset = problem.make_dataset(
        filename=args.dataset,
        batch_size=1,
        num_samples=args.num_samples,
        vehicle_capacity=1.0,
        min_trips_required_lb=args.min_trips,
        min_trips_required_ub=args.max_trips,
        use_ortec=args.use_ortec,
    )
    rows = []
    for index in range(len(dataset)):
        objective, runtime = evaluate_instance(
            args, problem, dataset[index], ortec
        )
        rows.append(
            {"sample": index, "objective": objective, "runtime": runtime}
        )
    result = {
        "problem": args.problem,
        "dataset": args.dataset,
        "scenarios": args.scenarios,
        "solver_time_ms": args.solver_time_ms,
        "seed": args.seed,
        "distribution": (
            "ortec"
            if ortec is not None
            else "skew-spatio-temporal"
            if args.arrival_weights is not None or args.arrival_skews is not None
            else "uniform"
        ),
        "ortec_source": args.use_ortec,
        "samples": rows,
        "objective_mean": float(np.mean([row["objective"] for row in rows])),
        "objective_std": float(np.std([row["objective"] for row in rows])),
        "runtime_mean": float(np.mean([row["runtime"] for row in rows])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
