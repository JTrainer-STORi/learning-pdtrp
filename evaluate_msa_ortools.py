#!/usr/bin/env python
"""Event-driven multiple-scenario OR-Tools baseline.

Uniform scenarios draw new coordinates. ORTEC scenarios draw customers from the
fixed source instance and use its asymmetric road-duration matrix.
"""

import argparse
import json
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from utils import load_problem
from utils.data_utils import dataset_to_input
from utils.dataset_parameters import add_dataset_time_arguments, dataset_time_options, validate_dataset_time_arguments
from utils.ortools_reloads import DEPOT_RETURN_POLICY, depot_return_allowance
from utils.service_times import (
    add_service_time_arguments,
    sample_service_times,
    service_time_parameters,
    validate_service_time_arguments,
)
from utils.solver_budgets import (
    add_solver_budget_arguments,
    run_solver_budget_sweep,
    validate_solver_budget_arguments,
)
from problems.utils import generate_scaled_demands, read_vrplib, sample_ortec_customer_indices
from problems.time_windows import add_time_window_arguments, validate_time_window_arguments
from problems.ortec_locations import load_ortec_locations, make_ortec_location_dataset


TW_PROBLEMS = {"pdtrptw", "pdctrptw"}
CAPACITY_PROBLEMS = {"pdctrp", "pdctrptw"}
DEMAND_SCALE = 1000000


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
    if ortec.get("locations_only", False):
        recovered_coordinates = coordinates
        source_scale = np.ones(2)
    else:
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
    tolerance = 0.0011
    if ortec.get("locations_only", False):
        tolerance = 1e-6 * max(1.0, ortec["normalization"]["dimension_scale"])
        if np.max(np.abs(source_coordinates[selected] - coordinates)) > tolerance:
            raise ValueError("Saved coordinates do not match the ORTEC location source")
    if error > tolerance:
        raise ValueError(
            f"Could not reliably map saved ORTEC nodes (matrix error {error:.6f})"
        )
    return selected


def sample_ortec_scenario(
    ortec, problem_name, n_total, dod, min_trips, max_trips
):
    """Sample once from the fixed ORTEC pool without reparsing the VRPLIB file."""
    scaled_windows = ortec["time_windows"] * 482.0 / np.max(ortec["time_windows"][1:])
    customer_ids = sample_ortec_customer_indices(
        scaled_windows[:, 0], n_total - 1, fully_dynamic=dod == 1.0,
    )
    unsorted_source_ids = np.concatenate(([0], customer_ids))
    coordinates = ortec["coords"][unsorted_source_ids]
    coordinates = coordinates / coordinates.max(axis=0)
    service = ortec["service_times"][unsorted_source_ids] / 60.0
    customer_windows = scaled_windows[unsorted_source_ids[1:]]
    n_immediate = int(dod * (n_total - 1))
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


def sample_ortec_location_scenario(ortec, problem, actual, n_total, dod, min_trips, max_trips):
    """Use source geography with the same synthetic priors as the actual dataset."""
    mean, std = service_time_parameters(actual)
    options = dict(
        min_total=n_total, max_total=n_total, min_dod=dod, max_dod=dod,
        batch_size=1, num_samples=1, n_subregions=1,
        arrival_weights=[1], arrival_skews=["uniform"],
        service_times_mean=mean, service_times_var=std ** 2,
        min_time_window=float(actual.get("min_time_window", 60.0)),
        max_time_window=float(actual.get("max_time_window", 180.0)),
        speed=float(actual["speed"].item()) * 60.0,
        time_horizon=float(actual["time_horizon"].item()) / 60.0,
        latest_end=float(actual.get("latest_end", 120.0)) / 60.0,
        dimension_scale=ortec["normalization"]["dimension_scale"],
        vehicle_capacity=1.0, min_trips_required_lb=min_trips, min_trips_required_ub=max_trips,
    )
    seed = int(np.random.randint(0, 2**32, dtype=np.uint32))
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        dataset, source_ids = make_ortec_location_dataset(problem, options, ortec, seed)
    sample = dataset[0]
    sample["source_ids"] = source_ids[0]
    return sample


class NoImprovementTracker:
    """Track strict incumbent improvements with an injectable monotonic clock.

    The first observation starts the patience interval; it is the initial
    incumbent rather than an improvement. Checking expiry does not depend on
    receiving another solution callback.
    """

    def __init__(self, no_improvement_ms, *, clock=None):
        if no_improvement_ms <= 0:
            raise ValueError("no_improvement_ms must be positive")
        self.no_improvement_seconds = no_improvement_ms / 1000.0
        self.clock = time.monotonic if clock is None else clock
        self.best_objective = None
        self.last_improvement_at = None
        self.improvement_count = 0

    def observe(self, objective):
        if self.best_objective is None:
            self.best_objective = objective
            self.last_improvement_at = self.clock()
            return False
        if objective < self.best_objective:
            self.best_objective = objective
            self.last_improvement_at = self.clock()
            self.improvement_count += 1
            return True
        return False

    def expired(self):
        return (
            self.last_improvement_at is not None
            and self.clock() - self.last_improvement_at
            >= self.no_improvement_seconds
        )


def initial_route_policy(problem_name):
    if problem_name == "pdctrptw":
        return "best_of_demand_packing_and_earliest_deadline"
    if problem_name == "pdtrptw":
        return "best_of_customer_order_and_earliest_deadline"
    return "demand_packing" if problem_name in CAPACITY_PROBLEMS else "customer_order"


def _deadline_ordered_route(
    customers, first_nodes, windows, demands=None, capacity=None, reload_nodes=(),
):
    """Visit earliest deadlines first, reloading before capacity is exceeded.

    When first-stage customer choices are supplied, start with an eligible
    observed customer, even if a sampled future request has an earlier deadline.
    Reload-first candidates call this helper with the load already reset.
    Demand units match the integer capacity model. Omit demands for an
    uncapacitated route with no reloads.
    """
    ordered = sorted(customers, key=lambda node: (int(windows[node, 1]), node))
    if first_nodes:
        allowed = set(first_nodes)
        first = next(
            (node for node in ordered
             if node in allowed
             and (demands is None or demands[0] + demands[node] <= capacity)),
            None,
        )
        if first is None:
            return None
        ordered = [first, *[node for node in ordered if node != first]]

    if demands is None:
        return ordered

    route = []
    used = int(demands[0])
    reloads = iter(sorted(reload_nodes))
    for node in ordered:
        demand = int(demands[node])
        if demand > capacity:
            return None
        if used + demand > capacity:
            reload = next(reloads, None)
            if reload is None:
                return None
            route.append(reload)
            used = 0
        route.append(node)
        used += demand
    return route


def _demand_packing_route(customers, demands, capacity, reload_nodes, *, first=None):
    """Build the capacity-feasible seed, optionally fixing its first customer."""
    bins = [[]]
    bin_loads = [int(demands[0])]
    if first is not None:
        bins[0].append(first)
        bin_loads[0] += int(demands[first])
    remaining = sorted((node for node in customers if node != first),
                       key=lambda node: int(demands[node]), reverse=True)
    for node in remaining:
        node_demand = int(demands[node])
        for index, load in enumerate(bin_loads):
            if load + node_demand <= capacity:
                bins[index].append(node)
                bin_loads[index] += node_demand
                break
        else:
            bins.append([node])
            bin_loads.append(node_demand)
    route = []
    reloads = iter(sorted(reload_nodes))
    for index, group in enumerate(bins):
        if index:
            try:
                route.append(next(reloads))
            except StopIteration as error:
                raise RuntimeError("Not enough reload nodes to construct a feasible route") from error
        route.extend(group)
    return route


def solve_scenario(
    data,
    problem_name,
    time_limit_ms=None,
    *,
    no_improvement_ms=None,
    return_details=False,
):
    """Solve a routing scenario using GLS and an explicit stopping rule.

    A time budget retains the historical bounded online/MSA interface. Patience
    instead stops after no strict improvement of the best objective for the
    requested interval; this does not prove optimality. Times use integer units
    of 0.0001, matching the scorer's four-decimal travel times. Service/release/
    window times are rounded to that grid and weights to 0.000001, so objectives
    remain a numeric approximation of floating-point scoring.
    """
    if time_limit_ms is None and no_improvement_ms is None:
        raise ValueError("A positive time limit or no-improvement interval is required")
    if time_limit_ms is not None and (
        not np.isfinite(time_limit_ms) or time_limit_ms <= 0
    ):
        raise ValueError("time_limit_ms must be finite and positive")
    if no_improvement_ms is not None and (
        not np.isfinite(no_improvement_ms) or no_improvement_ms <= 0
    ):
        raise ValueError("no_improvement_ms must be finite and positive")

    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    started = time.monotonic()
    size = len(data["distance"])
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    time_scale = 10000
    weight_scale = 1000000
    demand_scale = DEMAND_SCALE
    objective_scale = time_scale * weight_scale
    distance = np.asarray(data["distance"])
    # Round travel separately from service, as in problem.get_times().
    travel = np.rint(np.round(distance / data["speed"], 4) * time_scale).astype(np.int64)
    service = np.rint(np.asarray(data["service"]) * time_scale).astype(np.int64)
    arrivals = np.rint(
        np.asarray(data.get("arrival_times", np.zeros(size))) * time_scale
    ).astype(np.int64)
    theta = data["theta"] if problem_name in TW_PROBLEMS else 1.0
    gamma = max(0, int(round(data["gamma"] * weight_scale))) if problem_name in TW_PROBLEMS else 0
    lateness_constant = 0

    def transit(from_index, to_index):
        source = manager.IndexToNode(from_index)
        target = manager.IndexToNode(to_index)
        return int(travel[source, target] + service[source])

    transit_index = routing.RegisterTransitCallback(transit)

    def objective(from_index, to_index):
        source = manager.IndexToNode(from_index)
        target = manager.IndexToNode(to_index)
        return int(
            round(
                theta * distance[source, target] * objective_scale
            )
        )

    objective_index = routing.RegisterTransitCallback(objective)
    routing.SetArcCostEvaluatorOfAllVehicles(objective_index)

    if problem_name in TW_PROBLEMS:
        # Include enough room to visit every customer after the latest release
        # or opening time, even for scenarios using large absolute time units.
        windows = np.rint(np.asarray(data["windows"]) * time_scale).astype(np.int64)
        horizon = int(
            max(0, int(arrivals.max()), int(windows[:, 0].max()))
            + (size + 1) * (int(travel.max()) + int(service.max()) + 1)
        )
        routing.AddDimension(transit_index, horizon, horizon, True, "Time")
        dimension = routing.GetDimensionOrDie("Time")
        solver = routing.solver()
        for node, (start, end) in enumerate(windows):
            if node == 0:
                continue
            index = manager.NodeToIndex(node)
            dimension.CumulVar(index).SetMin(max(0, int(start)))
            if node not in data["reload_nodes"]:
                # Relative deadlines may already have passed. Since cumul >= 0,
                # max(cumul-end, 0) = cumul + (-end) whenever end < 0.
                lateness_constant += gamma * max(0, -int(end))
                dimension.SetCumulVarSoftUpperBound(
                    index,
                    max(0, int(end)),
                    gamma,
                )
            # The scorer waits until the request is revealed before beginning
            # travel to it: service_start >= release + incoming travel. A bare
            # release lower bound would let an offline route depart too early.
            for source_index in range(routing.Size()):
                source = manager.IndexToNode(source_index)
                if source == node:
                    continue
                lower_bound = int(arrivals[node] + travel[source, node])
                if lower_bound <= 0:
                    continue
                selected_arc = solver.IsEqualCstVar(routing.NextVar(source_index), index)
                solver.Add(
                    dimension.CumulVar(index)
                    >= lower_bound * selected_arc
                )

    if problem_name in CAPACITY_PROBLEMS:
        scaled_demands = np.rint(data["demand"] * demand_scale).astype(np.int64)

        def demand(index):
            return int(scaled_demands[manager.IndexToNode(index)])

        demand_index = routing.RegisterUnaryTransitCallback(demand)
        capacity = int(round(data["capacity"] * demand_scale))
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

    first_nodes = data.get("first_nodes", [])
    if first_nodes:
        routing.NextVar(routing.Start(0)).SetValues(
            [manager.NodeToIndex(node) for node in first_nodes]
        )

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    if time_limit_ms is not None:
        parameters.time_limit.FromMilliseconds(time_limit_ms)

    # Give OR-Tools an explicit feasible incumbent. First-solution heuristics
    # are unreliable with optional negative-demand reload nodes, even when the
    # capacity model is feasible. Both TW variants also compare deadline order
    # so short searches need not repair an arbitrary customer order from scratch.
    customers = [
        node
        for node in range(1, size)
        if node not in data["reload_nodes"]
    ]
    first_customers = [node for node in first_nodes if node not in data["reload_nodes"]]
    first_reload = next((node for node in first_nodes if node in data["reload_nodes"]), None)
    first = None
    if first_customers:
        if problem_name in CAPACITY_PROBLEMS:
            first = next(
                (
                    node
                    for node in first_customers
                    if scaled_demands[0] + scaled_demands[node] <= capacity
                ),
                None,
            )
        else:
            first = first_customers[0]
        if first is None:
            raise RuntimeError(
                "No first-stage customer fits the current vehicle capacity: "
                f"used={float(data['demand'][0]):.6f}, "
                f"first_demands={[round(float(data['demand'][node]), 6) for node in first_customers]}"
            )
        customers = [first, *[node for node in customers if node != first]]
    initial_route = []
    if problem_name in CAPACITY_PROBLEMS:
        initial_route = _demand_packing_route(
            customers, scaled_demands, capacity, data["reload_nodes"], first=first,
        )
    else:
        initial_route = customers

    initial_routes = [
        ("demand_packing" if problem_name in CAPACITY_PROBLEMS else "customer_order",
         initial_route)
    ] if not first_nodes or first is not None else []
    if problem_name in TW_PROBLEMS and initial_routes:
        deadline_route = _deadline_ordered_route(
            customers, first_customers, windows,
            scaled_demands if problem_name in CAPACITY_PROBLEMS else None,
            capacity if problem_name in CAPACITY_PROBLEMS else None,
            data["reload_nodes"],
        )
        if deadline_route is not None and deadline_route != initial_route:
            initial_routes.append(("earliest_deadline", deadline_route))

    if first_reload is not None:
        # Returning now is a real first-stage action, even if another customer
        # fits. After this return the load is zero, not used_capacity-capacity.
        reset_demands = scaled_demands.copy()
        reset_demands[0] = 0
        remaining_reloads = data["reload_nodes"] - {first_reload}
        packed = _demand_packing_route(customers, reset_demands, capacity, remaining_reloads)
        initial_routes.append(("demand_packing_after_reload", [first_reload, *packed]))
        if problem_name == "pdctrptw":
            deadline = _deadline_ordered_route(
                customers, [], windows, reset_demands, capacity, remaining_reloads,
            )
            if deadline is not None and deadline != packed:
                initial_routes.append(("earliest_deadline_after_reload", [first_reload, *deadline]))

    # Reading a seed implicitly closes the model. Preserve guided local search
    # when closing it, but do not apply the improvement-search deadline to seed
    # restoration: validating a large capacity route can take longer than a
    # millisecond-scale scenario budget. The capped parameters are passed to
    # SolveFromAssignmentWithParameters below, with no search-budget extension.
    seed_parameters = pywrapcp.DefaultRoutingSearchParameters()
    seed_parameters.CopyFrom(parameters)
    seed_parameters.ClearField("time_limit")
    seed_parameters.solution_limit = 1  # Restore one assignment per candidate route.
    routing.CloseModelWithParameters(seed_parameters)
    initial_assignment = None
    initial_method = None
    initial_objective = None
    initial_objectives = {}
    for method, candidate_route in initial_routes:
        candidate = routing.ReadAssignmentFromRoutes([candidate_route], True)
        if candidate is None:
            initial_objectives[method] = None
            continue
        candidate_objective = candidate.ObjectiveValue()
        initial_objectives[method] = (
            candidate_objective + lateness_constant
        ) / objective_scale
        if initial_objective is None or candidate_objective < initial_objective:
            # ReadAssignmentFromRoutes reuses model-owned assignment storage.
            # Preserve the incumbent before restoring another candidate, even
            # if that later restoration fails or produces a worse route.
            initial_assignment = routing.solver().Assignment(candidate)
            initial_method = method
            initial_objective = candidate_objective

    def result(assignment, termination_reason, tracker=None):
        route = []
        if assignment is not None:
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = assignment.Value(routing.NextVar(index))
            route = route[1:]
        if not return_details:
            return route
        return {
            "route": route,
            "solver_objective": (
                (assignment.ObjectiveValue() + lateness_constant) / objective_scale
                if assignment is not None else None
            ),
            "solver_status": int(routing.status()),
            "termination_reason": termination_reason,
            "elapsed_seconds": time.monotonic() - started,
            "improvement_count": tracker.improvement_count if tracker is not None else 0,
            "time_scale": time_scale,
            "weight_scale": weight_scale,
            "objective_scale": objective_scale,
            "demand_scale": demand_scale,
            "initial_route_policy": initial_route_policy(problem_name),
            "initial_route_method": initial_method,
            "initial_solver_objective": (
                (initial_objective + lateness_constant) / objective_scale
                if initial_objective is not None else None
            ),
            "initial_route_objectives": initial_objectives,
        }

    if initial_assignment is None:
        return result(None, "no_feasible_solution")

    tracker = NoImprovementTracker(
        no_improvement_ms if no_improvement_ms is not None else float("inf")
    )
    tracker.observe(initial_assignment.ObjectiveValue())
    objective_var = routing.CostVar()

    def at_solution():
        tracker.observe(objective_var.Value())

    routing.AddAtSolutionCallback(at_solution)
    stagnation_reached = False

    def stagnation_limit_reached():
        nonlocal stagnation_reached
        if tracker.expired():
            stagnation_reached = True
        return stagnation_reached

    # Keep Python callbacks and the monitor alive throughout every solve. A
    # search limit is checked even when GLS is not producing solution events.
    stagnation_limit = None
    if no_improvement_ms is not None:
        stagnation_limit = routing.solver().CustomLimit(stagnation_limit_reached)
        routing.AddSearchMonitor(stagnation_limit)
    search_started = time.monotonic()
    solution = routing.SolveFromAssignmentWithParameters(
        initial_assignment, parameters
    )
    search_elapsed = time.monotonic() - search_started
    if stagnation_reached:
        termination_reason = "no_improvement"
    elif int(routing.status()) == 7:
        termination_reason = "search_completed"
    elif (
        time_limit_ms is not None
        and (
            search_elapsed >= time_limit_ms / 1000.0 * 0.95
            or int(routing.status()) in (2, 4)
        )
    ):
        termination_reason = "time_limit"
    else:
        termination_reason = "search_completed"
    if solution is None:
        # An expired search can reject its current leaf. The validated seed is
        # still a feasible incumbent, including under very small time budgets.
        solution = initial_assignment
    return result(solution, termination_reason, tracker)


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
    service_mean, service_std = service_time_parameters(actual)
    if ortec is not None:
        if ortec.get("locations_only", False):
            sampled = sample_ortec_location_scenario(
                ortec, problem, actual, n_total, dod, min_trips, max_trips
            )
        else:
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
                service_times_mean=service_mean,
                service_times_var=service_std ** 2,
                min_time_window=float(actual.get("min_time_window", 60.0)),
                max_time_window=float(actual.get("max_time_window", 180.0)),
                speed=float(actual["speed"].item()) * 60.0,
                time_horizon=float(actual["time_horizon"].item()) / 60.0,
                latest_end=float(actual.get("latest_end", 120.0)) / 60.0,
                vehicle_capacity=1.0,
                min_trips_required_lb=min_trips,
                min_trips_required_ub=max_trips,
            )[0]
    current_time = float(state.cur_times.item())
    visited = state.visited_[0, 0]
    arrived = ~state.not_arrived_[0, 0]
    actual_feasible = ~state.get_mask()[0, 0]
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

    # Every unserved customer's realized duration is hidden, even after its
    # request arrives. Draw one independent duration per scenario customer,
    # including future ORTEC customers instead of using their saved durations.
    customer_service = iter(sample_service_times(actual, len(known_ids) + len(future_ids)))
    coords = [actual["all_nodes"][0, state.prev_a.item()]]
    service = [torch.tensor(0.0)]
    arrivals = [-current_time]
    mapping = [None]
    source_ids = (
        [actual_source_ids[state.prev_a.item()]] if ortec is not None else None
    )
    reload_nodes = []
    first_nodes = []
    demand = [float(state.used_cap.item())] if problem_name in CAPACITY_PROBLEMS else []
    capacity = float(actual["vehicle_capacity"].item()) if problem_name in CAPACITY_PROBLEMS else 1.0
    windows = [(0.0, 10**6)]

    if problem_name in CAPACITY_PROBLEMS:
        # Count both observed and sampled future customers. Demand-generation
        # trip bounds must not cap the route's optional depot returns.
        reload_count = depot_return_allowance(len(known_ids) + len(future_ids))
        for _ in range(reload_count):
            reload_nodes.append(len(coords))
            coords.append(actual["all_nodes"][0, 0])
            service.append(actual["service_times"][0, 0])
            arrivals.append(float(actual["arrival_times"][0, 0]) - current_time)
            mapping.append(0)
            if ortec is not None:
                source_ids.append(0)
            demand.append(-capacity)
            windows.append(
                (
                    float(actual["window_starts"][0, 0]) - current_time,
                    float(actual["window_ends"][0, 0]) - current_time,
                )
                if problem_name in TW_PROBLEMS else (0.0, 10**6)
            )

    for node in known_ids:
        if actual_feasible[node].item():
            first_nodes.append(len(coords))
        coords.append(actual["all_nodes"][0, node])
        service.append(torch.tensor(next(customer_service), dtype=torch.float))
        arrivals.append(float(actual["arrival_times"][0, node]) - current_time)
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

    if reload_nodes and actual_feasible[0].item():
        # One representative is sufficient: all optional reloads are the same
        # physical depot. Never force a customer visit before a useful return.
        first_nodes.append(reload_nodes[0])

    for node in future_ids:
        coords.append(sampled["all_nodes"][node])
        service.append(torch.tensor(next(customer_service), dtype=torch.float))
        arrivals.append(float(sampled["arrival_times"][node]) - current_time)
        mapping.append(None)
        if ortec is not None:
            source_ids.append(sampled_source_ids[node])
        if problem_name in CAPACITY_PROBLEMS:
            demand.append(float(sampled["demand"][node]) * capacity)
        if problem_name in TW_PROBLEMS:
            windows.append(
                (
                    float(sampled["window_starts"][node]) - current_time,
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
        "arrival_times": np.asarray(arrivals),
        "speed": float(actual["speed"].item()),
        "mapping": mapping,
        "windows": windows,
        "demand": np.asarray(demand),
        "capacity": capacity,
        "reload_nodes": set(reload_nodes),
        "first_nodes": first_nodes,
        "gamma": float(actual["gamma"].item()) if problem_name in TW_PROBLEMS else 0.0,
        "theta": float(actual["theta"].item()) if problem_name in TW_PROBLEMS else 1.0,
    }


def resolve_worker_count(workers, scenarios=None):
    """Resolve zero to the CPUs available to this process, avoiding idle workers."""
    if workers < 0:
        raise ValueError("workers must be zero (automatic) or positive")
    if workers == 0:
        try:
            workers = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            workers = os.cpu_count() or 1
    if scenarios is not None:
        if scenarios < 1:
            raise ValueError("scenarios must be positive")
        workers = min(workers, scenarios)
    return max(1, workers)


def initialize_scenario_worker():
    """Keep tensor helpers from creating a thread pool inside every process."""
    torch.set_num_threads(1)


def create_scenario_executor(workers):
    """Create a reusable process pool; spawn avoids inherited solver/thread state."""
    if workers < 1:
        raise ValueError("executor workers must be positive")
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_scenario_worker,
    )


def solve_prepared_scenario(job):
    """Solve an already sampled scenario without touching the parent's RNGs."""
    scenario, problem_name, time_limit_ms, no_improvement_ms = job
    details = solve_scenario(
        scenario,
        problem_name,
        time_limit_ms,
        no_improvement_ms=no_improvement_ms,
        return_details=True,
    )
    details["worker_pid"] = os.getpid()
    return details


def evaluate_instance(
    args, problem, item, ortec=None, executor=None, return_details=False
):
    """Evaluate MSA, returning wall-clock runtime rather than summed worker time.

    A supplied executor is reused and remains owned by the caller. Without one,
    ``args.workers`` selects sequential solving or a pool scoped to this call.
    Optional details separate parent batch waiting time from summed per-scenario
    solver durations, which may exceed wall time when scenarios run in parallel.
    """
    started = time.perf_counter()
    workers = resolve_worker_count(getattr(args, "workers", 1), args.scenarios)
    if executor is not None:
        # A caller-provided pool may deliberately differ from args.workers.
        workers = getattr(executor, "_max_workers", workers)
    pool_context = (
        create_scenario_executor(workers)
        if executor is None and workers > 1 else nullcontext(executor)
    )
    with pool_context as pool:
        objective, details = _evaluate_instance(args, problem, item, ortec, pool, workers)
    runtime = time.perf_counter() - started
    details["wall_time_seconds"] = runtime
    if return_details:
        return objective, runtime, details
    return objective, runtime


def _evaluate_instance(args, problem, item, ortec, executor, workers):
    actual = dataset_to_input(item, args.problem, torch.device("cpu"))
    service_mean, service_std = service_time_parameters(actual)
    actual_source_ids = map_ortec_nodes(item, ortec) if ortec is not None else None
    state = problem.make_state(actual)
    tour = [0]
    time_limit_ms = getattr(args, "solver_time_ms", None)
    no_improvement_ms = getattr(args, "solver_no_improvement_ms", 60000)
    solver_wall_time = 0.0
    aggregate_solver_time = 0.0
    scenario_count = 0
    improvement_count = 0
    termination_counts = Counter()
    while not state.all_finished():
        feasible = ~state.get_mask()[0, 0]
        feasible_customers = feasible.clone()
        feasible_customers[0] = False
        if not feasible_customers.any():
            selected = torch.tensor([0])
        else:
            # Sample in the parent and finish the entire batch before solving.
            # Worker count and completion order cannot change the RNG sequence.
            scenarios = []
            for _ in range(args.scenarios):
                scenarios.append(
                    make_scenario(
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
                        getattr(args, "n_subregions", 1),
                        getattr(args, "arrival_weights", None),
                        getattr(args, "arrival_skews", None),
                    )
                )
            jobs = [
                (scenario, args.problem, time_limit_ms, no_improvement_ms)
                for scenario in scenarios
            ]
            batch_started = time.perf_counter()
            if executor is None:
                results = list(map(solve_prepared_scenario, jobs))
            else:
                # executor.map preserves input order, including for voting ties.
                results = list(executor.map(solve_prepared_scenario, jobs))
            solver_wall_time += time.perf_counter() - batch_started
            votes = []
            for scenario, details in zip(scenarios, results):
                scenario_count += 1
                aggregate_solver_time += details["elapsed_seconds"]
                improvement_count += details["improvement_count"]
                termination_counts[details["termination_reason"]] += 1
                route = details["route"]
                if not route:
                    continue
                action = scenario["mapping"][route[0]]
                if (
                    action is None
                    or not feasible[action].item()
                ):
                    raise RuntimeError(
                        "MSA violated its executable first-stage constraint"
                    )
                votes.append(action)
            if not votes:
                raise RuntimeError(
                    "All MSA OR-Tools scenarios failed to return a feasible action"
                )
            selected = torch.tensor(
                [Counter(votes).most_common(1)[0][0]], dtype=torch.long
            )
        state = state.update(selected)
        if not hasattr(state, "last_action_executed") or state.last_action_executed.item():
            tour.append(selected.item())

    pi = torch.tensor(tour, dtype=torch.long).unsqueeze(0)
    if hasattr(state, "get_visit_times"):
        visit_times = torch.tensor(state.get_visit_times())
    else:
        visit_times, _ = problem.get_times(actual, pi)
    if args.problem in TW_PROBLEMS:
        objective = problem.get_costs(
            actual, pi, visit_times, actual["gamma"], actual["theta"]
        )[0].item()
    else:
        objective = problem.get_costs(actual, pi)[0].item()
    return objective, {
        "solver_wall_time_seconds": solver_wall_time,
        "aggregate_solver_seconds": aggregate_solver_time,
        "scenario_count": scenario_count,
        "workers": workers,
        "termination_counts": dict(termination_counts),
        "improvement_count": improvement_count,
        "solver_no_improvement_ms": no_improvement_ms,
        "solver_time_ms": time_limit_ms,
        "service_time_model": "sampled",
        "action_policy": "vote_over_feasible_customers_and_depot",
        "service_times_mean": service_mean,
        "service_times_var": service_std ** 2,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_dataset_time_arguments(parser)
    add_service_time_arguments(parser)
    add_time_window_arguments(parser)
    add_solver_budget_arguments(parser)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dod", type=float, required=True)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument(
        "--solver-no-improvement-ms", type=int,
        help="Stop each GLS scenario after this interval without a strict best improvement (default: 60000)",
    )
    parser.add_argument(
        "--solver-time-ms", type=int,
        help="Optional total search time cap per scenario; unset by default",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel scenario processes (default: 1); 0 uses available CPUs, up to the scenario count",
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0, help="First zero-based dataset instance to evaluate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-trips", type=int, default=3,
        help="Lower trip bound for sampled demand generation",
    )
    parser.add_argument(
        "--max-trips", type=int, default=9,
        help="Upper trip bound for sampled demand generation; does not cap depot returns",
    )
    ortec_mode = parser.add_mutually_exclusive_group()
    ortec_mode.add_argument(
        "--use-ortec",
        help="Legacy ORTEC datasets with the imported road-duration matrix",
    )
    ortec_mode.add_argument(
        "--ortec-locations",
        help="ORTEC coordinate-only source; use synthetic timing priors and Euclidean travel",
    )
    parser.add_argument(
        "--dimension-scale", type=float, default=1.0,
        help="Coordinate square size for --ortec-locations; must match generation (default: 1)",
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
    args = parser.parse_args(argv)
    validate_dataset_time_arguments(parser, args)
    validate_solver_budget_arguments(parser, args, ("solver",))
    if args.scenarios < 1:
        parser.error("scenarios must be positive")
    if args.workers < 0:
        parser.error("workers must be zero (automatic) or positive")
    if args.num_samples < 1:
        parser.error("num-samples must be positive")
    if args.offset < 0:
        parser.error("offset must be nonnegative")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not np.isfinite(args.dimension_scale) or args.dimension_scale <= 0:
        parser.error("dimension-scale must be finite and positive")
    validate_service_time_arguments(parser, args)
    validate_time_window_arguments(parser, args)
    if args.arrival_weights is not None and len(args.arrival_weights) != args.n_subregions:
        parser.error("arrival-weights must have one value per subregion")
    if args.arrival_skews is not None and len(args.arrival_skews) != args.n_subregions:
        parser.error("arrival-skews must have one value per subregion")
    return args


def instance_seed(base_seed, global_sample_index):
    """Derive the same scenario RNG seed regardless of dataset chunk boundaries."""
    if base_seed < 0 or global_sample_index < 0:
        raise ValueError("base seed and global sample index must be nonnegative")
    return int(
        np.random.SeedSequence([base_seed, global_sample_index]).generate_state(1)[0]
    )


def main(argv=None):
    args = parse_args(argv)
    if args.solve_seconds is not None:
        return run_solver_budget_sweep(args, run_evaluation, ("solver",), file_output=True)
    return run_evaluation(args)


def run_evaluation(args):
    problem = load_problem(args.problem)
    if args.ortec_locations:
        ortec = load_ortec_locations(args.ortec_locations, args.dimension_scale)
    else:
        ortec = make_ortec_context(args.use_ortec) if args.use_ortec else None
    dataset = problem.make_dataset(
        filename=args.dataset,
        batch_size=1,
        num_samples=args.num_samples,
        offset=args.offset,
        vehicle_capacity=1.0,
        min_trips_required_lb=args.min_trips,
        min_trips_required_ub=args.max_trips,
        use_ortec=args.use_ortec,
        service_times_mean=args.service_times_mean,
        service_times_var=args.service_times_var,
        min_time_window=args.min_time_window,
        max_time_window=args.max_time_window,
        **dataset_time_options(args),
    )
    if not len(dataset):
        raise ValueError(f"No samples found in dataset {args.dataset} at offset {args.offset}")
    rows = []
    workers = resolve_worker_count(args.workers, args.scenarios)
    pool_context = create_scenario_executor(workers) if workers > 1 else nullcontext(None)
    started = time.perf_counter()
    # Keep workers alive across every decision and dataset instance. Startup and
    # shutdown are included in the whole-dataset evaluation wall time.
    with pool_context as executor:
        for index in range(len(dataset)):
            sample_index = args.offset + index
            sample_seed = instance_seed(args.seed, sample_index)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            objective, runtime, details = evaluate_instance(
                args, problem, dataset[index], ortec,
                executor=executor, return_details=True,
            )
            rows.append(
                {
                    "sample": sample_index,
                    "instance_seed": sample_seed,
                    "objective": objective,
                    "runtime": runtime,
                    **details,
                }
            )
    wall_time = time.perf_counter() - started
    termination_counts = Counter()
    for row in rows:
        termination_counts.update(row["termination_counts"])
    result = {
        "problem": args.problem,
        "dataset_parameters": dataset_time_options(args),
        "dataset": args.dataset,
        "offset": args.offset,
        "scenarios": args.scenarios,
        "solver_time_ms": args.solver_time_ms,
        "solver_no_improvement_ms": args.solver_no_improvement_ms,
        "solver_stopping_rule": (
            "time_limit" if args.solver_no_improvement_ms is None else "no_improvement"
        ),
        "service_time_model": "sampled",
        "service_times_mean": args.service_times_mean,
        "service_times_var": args.service_times_var,
        "workers_requested": args.workers,
        "time_window_scenario": args.time_window_scenario,
        "min_time_window": args.min_time_window,
        "max_time_window": args.max_time_window,
        "workers": workers,
        "depot_return_policy": (
            DEPOT_RETURN_POLICY if args.problem in CAPACITY_PROBLEMS else "not_applicable"
        ),
        "initial_route_policy": initial_route_policy(args.problem),
        "action_policy": "vote_over_feasible_customers_and_depot",
        "seed": args.seed,
        "distribution": (
            "ortec-locations-uniform"
            if args.ortec_locations
            else "ortec"
            if ortec is not None
            else "skew-spatio-temporal"
            if args.arrival_weights is not None or args.arrival_skews is not None
            else "uniform"
        ),
        "ortec_source": args.use_ortec,
        "ortec_location_source": (
            {key: ortec[key] for key in ("filename", "sha256", "normalization")}
            if args.ortec_locations else None
        ),
        "samples": rows,
        "objective_mean": float(np.mean([row["objective"] for row in rows])),
        "objective_std": float(np.std([row["objective"] for row in rows])),
        "runtime_mean": float(np.mean([row["runtime"] for row in rows])),
        "wall_time_seconds": wall_time,
        "solver_wall_time_seconds": sum(row["solver_wall_time_seconds"] for row in rows),
        "aggregate_solver_seconds": sum(row["aggregate_solver_seconds"] for row in rows),
        "scenario_count": sum(row["scenario_count"] for row in rows),
        "termination_counts": dict(termination_counts),
        "improvement_count": sum(row["improvement_count"] for row in rows),
    }
    if getattr(args, "solve_budget_seconds", None) is not None:
        result["solve_seconds"] = args.solve_budget_seconds
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    return result


if __name__ == "__main__":
    main()
