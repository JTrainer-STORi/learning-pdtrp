"""Regression checks for oracle stopping and the objective shared with scoring."""

from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

import evaluate_msa_ortools as msa
import evaluate_ortools_baselines as baselines
import evaluate_online_ortools as online
import evaluate_clairvoyant_ortools as clairvoyant
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW


PROBLEMS = (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW)
TW_PROBLEMS = (PDTRPTW, PDCTRPTW)


def actual_input(distance=None):
    if distance is None:
        distance = [[0.0, 1.0, 1.0], [2.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    size = len(distance)
    coordinates = torch.stack((torch.arange(size), torch.zeros(size)), dim=1).float()
    ends = torch.full((1, size), 1000.0)
    ends[0, 0] = 0.0
    demand = torch.full((1, size), 0.1)
    demand[0, 0] = 0.0
    return {
        "all_nodes": coordinates.unsqueeze(0),
        "distance_matrix": torch.tensor([distance], dtype=torch.float),
        "graph": torch.zeros(1, size, size, dtype=torch.bool),
        "speed": torch.tensor([1.0]),
        "time_horizon": torch.tensor([1000.0]),
        "latest_end": torch.tensor([100.0]),
        "service_times": torch.zeros(1, size),
        "service_times_mean": torch.zeros(1),
        "service_times_std": torch.zeros(1),
        "arrival_times": torch.zeros(1, size),
        "window_starts": torch.zeros(1, size),
        "window_ends": ends,
        "gamma": torch.tensor([1.0]),
        "theta": torch.tensor([1.0]),
        "demand": demand,
        "vehicle_capacity": torch.tensor([[1.0]]),
    }


def solve_actual(problem, actual, *, first_customer=None):
    scenario = baselines.scenario_from_nodes(
        problem.NAME,
        actual,
        0,
        list(range(1, actual["all_nodes"].shape[1])),
        0.0,
        9,
    )
    if first_customer is not None:
        scenario["first_nodes"] = [scenario["mapping"].index(first_customer)]
    details = msa.solve_scenario(scenario, problem.NAME, 100, return_details=True)
    tour = [0, *[scenario["mapping"][node] for node in details["route"]]]
    value = baselines.score(problem, problem.NAME, actual, torch.tensor([tour]))
    return value, tour, details


@pytest.mark.parametrize("problem", PROBLEMS)
def test_zero_arrival_solver_objective_matches_asymmetric_tour_score(problem):
    value, tour, details = solve_actual(problem, actual_input())

    assert tour == [0, 1, 2]
    assert value == pytest.approx(3.0)
    assert details["solver_objective"] == pytest.approx(value, abs=1e-4)
    assert isinstance(details["solver_status"], int)


@pytest.mark.parametrize("problem", TW_PROBLEMS)
def test_oracle_waits_for_revelation_before_travelling_to_customer(problem):
    actual = actual_input()
    actual["arrival_times"] = torch.tensor([[0.0, 100.0, 0.0]])
    actual["window_starts"] = torch.tensor([[0.0, 100.0, 0.0]])
    actual["window_ends"] = torch.tensor([[0.0, 102.0, 101.0]])
    actual["gamma"] = torch.tensor([2.0])

    # Ignoring releases makes [0, 1, 2] look cheaper (3 versus 4), although
    # it actually reaches customer 2 late and costs 5 under project scoring.
    assert baselines.score(
        problem, problem.NAME, actual, torch.tensor([[0, 1, 2]])
    ) == pytest.approx(5.0)
    value, tour, details = solve_actual(problem, actual)

    assert tour == [0, 2, 1]
    assert value == pytest.approx(4.0)
    assert details["solver_objective"] == pytest.approx(value, abs=1e-4)


@pytest.mark.parametrize("problem", TW_PROBLEMS)
def test_fixed_first_customer_objective_includes_release_waiting(problem):
    actual = actual_input()
    actual["arrival_times"] = torch.tensor([[0.0, 100.0, 0.0]])
    actual["window_starts"] = torch.tensor([[0.0, 100.0, 0.0]])
    actual["window_ends"] = torch.tensor([[0.0, 102.0, 101.0]])
    actual["gamma"] = torch.tensor([2.0])

    value, tour, details = solve_actual(problem, actual, first_customer=1)

    assert tour == [0, 1, 2]
    assert value == pytest.approx(5.0)
    assert details["solver_objective"] == pytest.approx(value, abs=1e-4)


def test_capacity_reload_respects_actual_capacity_depot_service_and_weights():
    actual = actual_input()
    actual["vehicle_capacity"] = torch.tensor([[1.5]])
    actual["demand"] = torch.tensor([[0.0, 1.0, 1.0]])
    actual["arrival_times"] = torch.tensor([[0.0, 2.0, 0.0]])
    actual["window_starts"] = torch.tensor([[4.0, 4.0, 0.0]])
    actual["window_ends"] = torch.tensor([[0.0, 3.0, 3.0]])
    actual["service_times"] = torch.tensor([[2.0, 1.25, 0.5]])
    actual["gamma"] = torch.tensor([2.5])
    actual["theta"] = torch.tensor([1.5])

    value, tour, details = solve_actual(PDCTRPTW, actual, first_customer=1)

    assert tour == [0, 1, 0, 2]
    # Five distance units, plus customer lateness 1 + 7.25. Depot lateness
    # is excluded, but its two service units delay the second customer.
    assert value == pytest.approx(1.5 * 5.0 + 2.5 * 8.25)
    assert details["solver_objective"] == pytest.approx(value, abs=1e-4)


@pytest.mark.parametrize("problem", (PDCTRP, PDCTRPTW))
def test_near_capacity_packing_supplies_reload_required_by_integer_model(problem):
    actual = actual_input()
    actual["demand"] = torch.tensor([[0.0, 0.500004, 0.500004]])

    value, tour, details = solve_actual(problem, actual, first_customer=1)

    assert tour == [0, 1, 0, 2]
    assert value == pytest.approx(5.0)
    assert details["solver_objective"] == pytest.approx(value, abs=1e-4)


@pytest.mark.parametrize("problem", (PDCTRP, PDCTRPTW))
@pytest.mark.parametrize("customers", [0, 1, 4])
def test_capacity_scenario_allows_reloads_independently_of_packing_and_trip_bound(problem, customers):
    coords = np.column_stack((np.arange(customers + 1), np.zeros(customers + 1)))
    distance = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    actual = actual_input(distance.tolist())
    for max_trips in (1, 9):
        scenario = baselines.scenario_from_nodes(
            problem.NAME, actual, 0, list(range(1, customers + 1)), 0.0, max_trips
        )
        # These demands all fit on one load. Optional returns must still allow
        # time-aware routes to split that load across additional trips.
        assert len(scenario["reload_nodes"]) == customers
        assert [node for node in scenario["mapping"] if node not in (None, 0)] == list(range(1, customers + 1))
        for node in scenario["reload_nodes"]:
            assert scenario["mapping"][node] == 0
            assert scenario["demand"][node] == -1.0


@pytest.mark.parametrize("problem", (PDCTRP, PDCTRPTW))
def test_capacity_scenario_covers_initial_reload_with_partially_used_capacity(problem):
    coords = np.column_stack((np.arange(4), np.zeros(4)))
    actual = actual_input(np.linalg.norm(coords[:, None] - coords[None, :], axis=-1).tolist())
    actual["demand"] = torch.tensor([[0.0, 0.8, 0.8, 0.9]])
    scenario = baselines.scenario_from_nodes(problem.NAME, actual, 3, [1, 2], 0.9, 1)

    details = msa.solve_scenario(scenario, problem.NAME, 10, return_details=True)
    mapped = [scenario["mapping"][node] for node in details["route"]]

    assert mapped[0] == 0  # The current load must be cleared before either delivery.
    assert mapped.count(0) == 2
    assert sorted(node for node in mapped if node != 0) == [1, 2]
    assert np.isfinite(details["solver_objective"])


def test_extra_optional_reload_admits_a_better_time_window_route(monkeypatch):
    from ortools.constraint_solver import pywrapcp

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    actual = actual_input(np.linalg.norm(coords[:, None] - coords[None, :], axis=-1).tolist())
    actual["demand"] = torch.tensor([[0.0, 0.6, 0.4, 0.6, 0.4]])
    actual["window_starts"] = torch.tensor([[0.0, 0.0, 100.0, 0.0, 100.0]])
    actual["window_ends"] = torch.tensor([[0.0, 5.0, 200.0, 5.0, 200.0]])
    scenario = baselines.scenario_from_nodes("pdctrptw", actual, 0, [1, 2, 3, 4], 0.0, 1)
    nodes = {node: index for index, node in enumerate(scenario["mapping"]) if node not in (None, 0)}
    reloads = sorted(scenario["reload_nodes"])
    # Packing the demands needs only two trips (one reload). Respecting both
    # early deadlines benefits from three trips (two reloads).
    candidate = [nodes[1], reloads[0], nodes[3], nodes[2], reloads[1], nodes[4]]
    original_read = pywrapcp.RoutingModel.ReadAssignmentFromRoutes

    def read_candidate(routing, routes, ignore_inactive):
        # Validate the known counterexample in the actual OR-Tools model,
        # independently of whether a short heuristic search discovers it.
        assignment = original_read(routing, [candidate], ignore_inactive)
        assert assignment is not None
        return assignment

    monkeypatch.setattr(pywrapcp.RoutingModel, "ReadAssignmentFromRoutes", read_candidate)
    details = msa.solve_scenario(scenario, "pdctrptw", 1, return_details=True)
    tour = [0, *[scenario["mapping"][node] for node in details["route"]]]
    value = baselines.score(PDCTRPTW, "pdctrptw", actual, torch.tensor([tour]))

    assert value == pytest.approx(6.0)
    assert details["solver_objective"] == pytest.approx(value, abs=1e-4)
    assert baselines.score(PDCTRPTW, "pdctrptw", actual,
                           torch.tensor([[0, 1, 2, 0, 3, 4]])) == pytest.approx(101.0)


@pytest.fixture
def initial_route_only(monkeypatch):
    """Validate real assignments without letting search conceal seed quality."""
    from ortools.constraint_solver import pywrapcp

    monkeypatch.setattr(
        pywrapcp.RoutingModel, "SolveFromAssignmentWithParameters",
        lambda routing, assignment, parameters: None,
    )


def test_deadline_seed_adds_returns_to_serve_urgent_customers_first(initial_route_only):
    coords = np.array([[0.0, 0.0], *([[1.0, 0.0]] * 4)])
    actual = actual_input(np.linalg.norm(coords[:, None] - coords[None, :], axis=-1).tolist())
    actual["demand"] = torch.tensor([[0.0, 0.6, 0.4, 0.6, 0.4]])
    actual["window_starts"] = torch.tensor([[0.0, 0.0, 100.0, 0.0, 100.0]])
    actual["window_ends"] = torch.tensor([[0.0, 5.0, 200.0, 5.0, 200.0]])

    value, tour, details = solve_actual(PDCTRPTW, actual)

    assert tour == [0, 1, 0, 3, 2, 0, 4]
    assert value == pytest.approx(6.0)
    assert details["initial_route_method"] == "earliest_deadline"
    assert details["initial_route_objectives"] == pytest.approx(
        {"demand_packing": 101.0, "earliest_deadline": 6.0}
    )
    assert details["initial_solver_objective"] == pytest.approx(value)
    assert details["solver_objective"] == pytest.approx(value)


@pytest.mark.parametrize(("gamma", "theta", "method", "tour", "value"), [
    (2.0, 1.0, "earliest_deadline", [0, 2, 1], 4.0),
    (0.5, 2.0, "demand_packing", [0, 1, 2], 6.5),
])
def test_initial_route_selection_uses_full_weighted_objective(
    initial_route_only, gamma, theta, method, tour, value,
):
    actual = actual_input()
    actual["arrival_times"] = torch.tensor([[0.0, 100.0, 0.0]])
    actual["window_starts"] = torch.tensor([[0.0, 100.0, 0.0]])
    actual["window_ends"] = torch.tensor([[0.0, 102.0, 101.0]])
    actual["gamma"] = torch.tensor([gamma])
    actual["theta"] = torch.tensor([theta])

    score, selected_tour, details = solve_actual(PDCTRPTW, actual)

    assert selected_tour == tour
    assert score == pytest.approx(value)
    assert details["initial_route_method"] == method
    assert details["initial_solver_objective"] == pytest.approx(value)
    assert details["solver_objective"] == pytest.approx(value)
    # In the second case the preferred assignment must survive restoring the
    # worse deadline route into OR-Tools' reused assignment storage.
    for candidate, candidate_tour in (
        ("demand_packing", [0, 1, 2]), ("earliest_deadline", [0, 2, 1]),
    ):
        expected = baselines.score(PDCTRPTW, "pdctrptw", actual, torch.tensor([candidate_tour]))
        assert details["initial_route_objectives"][candidate] == pytest.approx(expected)


@pytest.mark.parametrize("rejected", ["demand_packing", "earliest_deadline", "both"])
def test_initial_route_selection_keeps_any_valid_candidate(monkeypatch, initial_route_only, rejected):
    from ortools.constraint_solver import pywrapcp

    actual = actual_input()
    actual["window_ends"] = torch.tensor([[0.0, 100.0, 50.0]])
    scenario = baselines.scenario_from_nodes("pdctrptw", actual, 0, [1, 2], 0.0, 9)
    original_read = pywrapcp.RoutingModel.ReadAssignmentFromRoutes
    attempts = []

    def reject_candidate(routing, routes, ignore_inactive):
        method = "demand_packing" if not attempts else "earliest_deadline"
        attempts.append(method)
        if rejected in (method, "both"):
            return None
        return original_read(routing, routes, ignore_inactive)

    monkeypatch.setattr(pywrapcp.RoutingModel, "ReadAssignmentFromRoutes", reject_candidate)
    details = msa.solve_scenario(scenario, "pdctrptw", 1, return_details=True)

    assert attempts == ["demand_packing", "earliest_deadline"]
    if rejected == "both":
        assert details["route"] == []
        assert details["termination_reason"] == "no_feasible_solution"
        assert details["initial_route_method"] is None
        assert details["initial_solver_objective"] is None
    else:
        expected = "earliest_deadline" if rejected == "demand_packing" else "demand_packing"
        assert details["initial_route_method"] == expected
        tour = [0, *[scenario["mapping"][node] for node in details["route"]]]
        score = baselines.score(PDCTRPTW, "pdctrptw", actual, torch.tensor([tour]))
        assert details["solver_objective"] == pytest.approx(score)
        assert details["initial_route_objectives"][rejected] is None


def test_deadline_seed_respects_first_stage_eligibility_and_used_capacity(initial_route_only):
    actual = actual_input((np.ones((4, 4)) - np.eye(4)).tolist())
    actual["demand"] = torch.tensor([[0.0, 0.5, 0.7, 0.3]])
    actual["window_ends"] = torch.tensor([[0.0, 50.0, 10.0, 1.0]])
    scenario = baselines.scenario_from_nodes("pdctrptw", actual, 0, [1, 2, 3], 0.4, 9)
    nodes = {node: index for index, node in enumerate(scenario["mapping"]) if node not in (None, 0)}
    # Customer 3 is a sampled future request; customer 2 cannot fit before a
    # reload. Only customer 1 is an executable first action at the current load.
    scenario["first_nodes"] = [nodes[2], nodes[1]]
    scenario["mapping"][nodes[3]] = None

    details = msa.solve_scenario(scenario, "pdctrptw", 1, return_details=True)

    assert details["initial_route_method"] == "earliest_deadline"
    assert details["route"][0] == nodes[1]
    assert [node for node in details["route"] if node not in scenario["reload_nodes"]] == [nodes[1], nodes[3], nodes[2]]
    demands = np.rint(scenario["demand"] * msa.DEMAND_SCALE).astype(np.int64)
    used = int(demands[0])
    for node in details["route"]:
        used = 0 if node in scenario["reload_nodes"] else used + int(demands[node])
        assert used <= msa.DEMAND_SCALE


def test_deadline_seed_can_start_with_a_required_reload(initial_route_only):
    actual = actual_input()
    actual["demand"] = torch.tensor([[0.0, 0.8, 0.8]])
    actual["window_ends"] = torch.tensor([[0.0, 1000.0, 0.0]])
    scenario = baselines.scenario_from_nodes("pdctrptw", actual, 0, [1, 2], 0.9, 9)

    details = msa.solve_scenario(scenario, "pdctrptw", 1, return_details=True)
    mapped = [scenario["mapping"][node] for node in details["route"]]

    assert details["initial_route_method"] == "earliest_deadline"
    assert mapped == [0, 2, 0, 1]
    assert np.isfinite(details["solver_objective"])


@pytest.mark.parametrize("problem", (PDCTRP, PDCTRPTW))
def test_msa_reload_allowance_covers_known_and_sampled_customers(monkeypatch, problem):
    coords = np.column_stack((np.arange(7), np.zeros(7)))
    actual = actual_input(np.linalg.norm(coords[:, None] - coords[None, :], axis=-1).tolist())
    actual["demand"] = torch.tensor([[0.0, *([0.6] * 6)]])
    sampled = {
        "all_nodes": actual["all_nodes"][0] + torch.tensor([0.0, 1.0]),
        "arrival_times": torch.tensor([0.0, *([10.0] * 6)]),
        "demand": actual["demand"][0],
        "window_starts": actual["window_starts"][0],
        "window_ends": actual["window_ends"][0],
    }
    fake_problem = SimpleNamespace(make_dataset=lambda **kwargs: [sampled])
    state = problem.make_state(actual)
    scenario = msa.make_scenario(fake_problem, problem.NAME, actual, state, 7, 0.5, 1, 1)
    customers = [node for node in range(1, len(scenario["mapping"])) if node not in scenario["reload_nodes"]]

    assert len(customers) == 12  # Six observed customers plus six future draws.
    assert len(scenario["reload_nodes"]) == 12
    details = msa.solve_scenario(scenario, problem.NAME, 10, return_details=True)
    visited = [node for node in details["route"] if node not in scenario["reload_nodes"]]
    assert sorted(visited) == customers
    assert details["route"][0] in scenario["first_nodes"]
    assert np.isfinite(details["solver_objective"])


@pytest.mark.parametrize("problem", TW_PROBLEMS)
def test_negative_deadline_keeps_overdue_constant_in_objective(problem):
    actual = actual_input([[0.0, 1.0], [1.0, 0.0]])
    actual["window_ends"] = torch.tensor([[0.0, -2.0]])
    actual["gamma"] = torch.tensor([2.0])

    value, _, details = solve_actual(problem, actual)

    assert value == pytest.approx(8.0)
    assert details["solver_objective"] == pytest.approx(value, abs=1e-4)


@pytest.mark.parametrize("problem", TW_PROBLEMS)
def test_small_positive_lateness_weight_is_not_rounded_to_zero(problem):
    actual = actual_input([[0.0, 1.0], [1.0, 0.0]])
    actual["window_ends"] = torch.tensor([[0.0, -10000.0]])
    actual["gamma"] = torch.tensor([0.0004])

    value, _, details = solve_actual(problem, actual)

    assert value == pytest.approx(6.0004, abs=1e-4)
    assert details["solver_objective"] == pytest.approx(value, abs=1e-4)


def test_equal_or_worse_incumbents_do_not_extend_patience():
    now = [0.0]
    tracker = msa.NoImprovementTracker(50, clock=lambda: now[0])
    assert tracker.observe(100) is False
    assert tracker.improvement_count == 0

    now[0] = 0.020
    assert tracker.observe(100) is False
    now[0] = 0.040
    assert tracker.observe(101) is False
    assert tracker.last_improvement_at == 0.0
    assert tracker.best_objective == 100

    # Expiration must be checked even when no new solution callback occurs.
    now[0] = 0.051
    assert tracker.expired() is True


def test_strictly_better_incumbent_restarts_patience():
    now = [0.0]
    tracker = msa.NoImprovementTracker(50, clock=lambda: now[0])
    tracker.observe(100)
    now[0] = 0.040
    assert tracker.observe(99) is True
    assert tracker.improvement_count == 1
    assert tracker.best_objective == 99
    assert tracker.last_improvement_at == 0.040
    now[0] = 0.080
    assert tracker.expired() is False
    now[0] = 0.091
    assert tracker.expired() is True


def search_scenario():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                            [1.0, 1.0], [2.0, 0.3], [0.2, 2.0]])
    distance = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    actual = actual_input(distance.tolist())
    return baselines.scenario_from_nodes("pdtrp", actual, 0, [1, 2, 3, 4, 5], 0.0, 9)


def test_guided_search_stops_after_short_patience():
    scenario = search_scenario()
    started = time.monotonic()
    details = msa.solve_scenario(
        scenario,
        "pdtrp",
        1500,  # Safeguard: patience, rather than this cap, must stop the solve.
        no_improvement_ms=30,
        return_details=True,
    )

    assert sorted(details["route"]) == [1, 2, 3, 4, 5]
    assert details["termination_reason"] == "no_improvement"
    assert details["elapsed_seconds"] >= 0.025
    assert time.monotonic() - started < 1.4
    assert details["improvement_count"] >= 0


def test_explicit_total_budget_can_stop_before_patience():
    scenario = search_scenario()
    details = msa.solve_scenario(
        scenario, "pdtrp", 30, no_improvement_ms=1000, return_details=True
    )

    assert details["termination_reason"] == "time_limit"
    assert sorted(details["route"]) == [1, 2, 3, 4, 5]
    assert np.isfinite(details["solver_objective"])


@pytest.mark.parametrize("patience", [None, 1000])
def test_oracle_total_budget_is_not_extended_by_feasibility_retry(monkeypatch, patience):
    from ortools.constraint_solver import pywrapcp

    routing_model = pywrapcp.RoutingModel
    limits = []

    class NoSearchSolution:
        def __init__(self, manager):
            self.routing = routing_model(manager)

        def __getattr__(self, name):
            return getattr(self.routing, name)

        def SolveFromAssignmentWithParameters(self, assignment, parameters):
            limits.append(parameters.time_limit.ToMilliseconds())
            return None

    monkeypatch.setattr(pywrapcp, "RoutingModel", NoSearchSolution)
    details = msa.solve_scenario(
        search_scenario(), "pdtrp", 20, no_improvement_ms=patience, return_details=True
    )

    assert limits == [20]
    # The already validated initial assignment is still usable as fallback.
    assert sorted(details["route"]) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("problem", (PDCTRP, PDCTRPTW))
def test_short_search_budget_does_not_reject_seed_during_restoration(monkeypatch, problem):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    routing_model = pywrapcp.RoutingModel
    restored_with_limits, search_limits = [], []

    class SlowSeedRestoration:
        def __init__(self, manager):
            self.routing = routing_model(manager)
            self.restoration_limit = None

        def __getattr__(self, name):
            return getattr(self.routing, name)

        def CloseModelWithParameters(self, parameters):
            self.restoration_limit = (parameters.time_limit.ToMilliseconds()
                                      if parameters.HasField("time_limit") else None)
            assert parameters.local_search_metaheuristic == routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            return self.routing.CloseModelWithParameters(parameters)

        def ReadAssignmentFromRoutes(self, routes, ignore_inactive):
            restored_with_limits.append(self.restoration_limit)
            # Emulate feasible seed validation requiring more time than the
            # one-millisecond improvement search; do not add a slow sleep.
            if self.restoration_limit is not None and self.restoration_limit < 50:
                return None
            return self.routing.ReadAssignmentFromRoutes(routes, ignore_inactive)

        def SolveFromAssignmentWithParameters(self, assignment, parameters):
            search_limits.append(parameters.time_limit.ToMilliseconds())
            return None  # Exercise the validated incumbent as the fallback.

    monkeypatch.setattr(pywrapcp, "RoutingModel", SlowSeedRestoration)
    scenario = baselines.scenario_from_nodes(problem.NAME, actual_input(), 0, [1, 2], 0.0, 9)
    details = msa.solve_scenario(scenario, problem.NAME, 1, return_details=True)

    assert restored_with_limits == [None]
    assert search_limits == [1]
    assert sorted(scenario["mapping"][node] for node in details["route"] if node not in scenario["reload_nodes"]) == [1, 2]
    assert np.isfinite(details["solver_objective"])


def test_timed_solver_still_returns_plain_route_by_default():
    scenario = baselines.scenario_from_nodes("pdtrp", actual_input(), 0, [1, 2], 0.0, 9)

    assert msa.solve_scenario(scenario, "pdtrp", 30) == [1, 2]


@pytest.mark.parametrize("module", [baselines, online, clairvoyant])
def test_oracle_defaults_to_patience_with_optional_total_budget(module):
    args = module.parse_args(
        ["--problem", "pdtrp", "--dataset", "input.txt", "--output", "output"]
    )

    if module is not online:
        assert args.static_no_improvement_ms == 60000
        assert args.static_time_ms is None
    if module is not clairvoyant:
        assert args.online_no_improvement_ms == 60000
        assert args.online_time_ms is None


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--static-no-improvement-ms", "0"),
        ("--static-no-improvement-ms", "-1"),
        ("--static-time-ms", "0"),
        ("--static-time-ms", "-1"),
        ("--online-time-ms", "0"),
        ("--online-time-ms", "-1"),
        ("--online-no-improvement-ms", "0"),
        ("--online-no-improvement-ms", "-1"),
        ("--num-samples", "0"),
        ("--max-trips", "0"),
    ],
)
def test_cli_rejects_nonpositive_limits(flag, value):
    module = clairvoyant if flag.startswith("--static") else online
    for parser in (baselines.parse_args, module.parse_args):
        with pytest.raises(SystemExit):
            parser(
                [
                    "--problem", "pdtrp", "--dataset", "input.txt", "--output", "output",
                    flag, value,
                ]
            )


@pytest.mark.parametrize("return_details", [False, True])
def test_evaluate_static_passes_patience_and_preserves_tuple_api(monkeypatch, return_details):
    calls = []
    details = {
        "route": [1, 2],
        "solver_objective": 3.0,
        "solver_status": 1,
        "termination_reason": "no_improvement",
        "elapsed_seconds": 0.03,
        "improvement_count": 0,
    }

    def solve(data, problem_name, time_limit_ms=None, **kwargs):
        calls.append((problem_name, time_limit_ms, kwargs))
        return details if kwargs.get("return_details") else details["route"]

    monkeypatch.setattr(clairvoyant, "solve_scenario", solve)
    args = SimpleNamespace(
        problem="pdtrp", max_trips=9, static_time_ms=None, static_no_improvement_ms=60000
    )
    result = clairvoyant.evaluate_static(args, PDTRP, actual_input(), return_details=return_details)

    assert result[0] == pytest.approx(3.0)
    assert result[2] == [0, 1, 2]
    assert len(result) == (4 if return_details else 3)
    if return_details:
        assert result[3] == details
    assert calls[0][:2] == ("pdtrp", None)
    assert calls[0][2]["no_improvement_ms"] == 60000


def test_online_solves_pass_patience_and_optional_per_decision_cap(monkeypatch):
    calls = []

    def solve(data, problem_name, time_limit_ms=None, **kwargs):
        calls.append((time_limit_ms, kwargs))
        route = [node for node, actual_node in enumerate(data["mapping"]) if actual_node not in (None, 0)]
        if kwargs.get("return_details"):
            return {
                "route": route, "elapsed_seconds": 0.01,
                "termination_reason": "no_improvement", "improvement_count": 0,
                "solver_objective": 3.0, "solver_status": 1,
            }
        return route

    monkeypatch.setattr(online, "solve_scenario", solve)
    args = SimpleNamespace(
        problem="pdtrp", max_trips=9, online_time_ms=17, online_no_improvement_ms=23
    )
    value, _, tour = online.evaluate_online(args, PDTRP, actual_input())

    assert value == pytest.approx(3.0)
    assert tour == [0, 1, 2]
    assert calls
    assert all(limit == 17 and kwargs["no_improvement_ms"] == 23 for limit, kwargs in calls)


def test_msa_solves_pass_patience_and_optional_per_scenario_cap(monkeypatch):
    actual = actual_input()
    calls = []

    def scenario(problem, problem_name, actual, state, *args, **kwargs):
        feasible = ~state.get_mask()[0, 0]
        feasible[0] = False
        customer = int(torch.nonzero(feasible).flatten()[0])
        return {"mapping": [None, customer]}

    def solve(data, problem_name, time_limit_ms=None, **kwargs):
        calls.append((time_limit_ms, kwargs))
        return {
            "route": [1], "elapsed_seconds": 0.01,
            "termination_reason": "no_improvement", "improvement_count": 0,
        }

    monkeypatch.setattr(msa, "make_scenario", scenario)
    monkeypatch.setattr(msa, "solve_scenario", solve)
    monkeypatch.setattr(msa, "dataset_to_input", lambda *args: actual)
    args = SimpleNamespace(
        problem="pdtrp", scenarios=2, solver_time_ms=19, solver_no_improvement_ms=29,
        workers=1, dod=0.5,
        min_trips=1, max_trips=9, n_subregions=1,
        arrival_weights=None, arrival_skews=None,
    )
    value, _ = msa.evaluate_instance(args, PDTRP, {"all_nodes": actual["all_nodes"][0]})

    assert value == pytest.approx(3.0)
    assert len(calls) == 4
    assert all(limit == 19 and kwargs["no_improvement_ms"] == 29 for limit, kwargs in calls)
