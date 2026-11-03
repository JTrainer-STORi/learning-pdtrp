"""Keep online planning independent of hidden realized service durations."""

from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_clairvoyant_ortools as clairvoyant
import evaluate_msa_ortools as msa
import evaluate_online_ortools as online
import evaluate_ortools_baselines as paired
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from utils.service_times import sample_service_times, service_time_parameters


PROBLEMS = (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW)
PLANNING_CLIS = (online, paired, msa)


def cli_args(module):
    arguments = [
        "--problem", "pdtrp", "--dataset", "data.txt", "--output", "results",
    ]
    return arguments + (["--dod", "0.5"] if module is msa else [])


@pytest.mark.parametrize("module", PLANNING_CLIS)
def test_planning_cli_service_time_prior_defaults(module):
    args = module.parse_args(cli_args(module))
    assert args.service_times_mean == 3.0
    assert args.service_times_var == 5.0


@pytest.mark.parametrize("module", PLANNING_CLIS)
@pytest.mark.parametrize(("mean", "variance"), [(4.0, 9.0), (0.0, 0.0)])
def test_planning_cli_configures_service_time_prior(module, mean, variance):
    args = module.parse_args([
        *cli_args(module), "--service-times-mean", str(mean),
        "--service-times-var", str(variance),
    ])
    assert args.service_times_mean == mean
    assert args.service_times_var == variance


@pytest.mark.parametrize("module", PLANNING_CLIS)
@pytest.mark.parametrize("flags", [
    ["--service-times-mean", "-1"],
    ["--service-times-mean", "nan"],
    ["--service-times-mean", "inf"],
    ["--service-times-var", "-1"],
    ["--service-times-var", "nan"],
    ["--service-times-var", "inf"],
    ["--service-times-mean", "0"],
])
def test_planning_cli_rejects_invalid_service_time_prior(module, flags):
    with pytest.raises(SystemExit):
        module.parse_args([*cli_args(module), *flags])


def test_clairvoyant_cli_does_not_need_an_online_service_time_prior():
    args = clairvoyant.parse_args(cli_args(clairvoyant))
    assert not hasattr(args, "service_times_mean")
    assert not hasattr(args, "service_times_var")


def actual_input(mean=3.0, std=math.sqrt(5.0)):
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    return {
        "all_nodes": coords,
        "distance_matrix": torch.cdist(coords, coords),
        "graph": (~torch.eye(4, dtype=torch.bool))[None],
        "arrival_times": torch.tensor([[0.0, 0.0, 0.0, 200.0]]),
        "service_times": torch.tensor([[2.0, 12.0, 17.0, 23.0]]),
        "service_times_mean": torch.tensor([mean]),
        "service_times_std": torch.tensor([std]),
        "speed": torch.tensor([1.0]),
        "time_horizon": torch.tensor([480.0]),
        "latest_end": torch.tensor([120.0]),
        "window_starts": torch.zeros(1, 4),
        "window_ends": torch.tensor([[0.0, 1000.0, 1000.0, 1000.0]]),
        "demand": torch.tensor([[0.0, 0.8, 0.8, 0.1]]),
        "vehicle_capacity": torch.tensor([[1.0]]),
        "gamma": torch.tensor([1.0]),
        "theta": torch.tensor([1.0]),
    }


def future_sample():
    return {
        "all_nodes": torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]),
        "arrival_times": torch.tensor([0.0, 0.0, 80.0, 150.0]),
        "service_times": torch.tensor([2.0, 500.0, 600.0, 700.0]),
        "window_starts": torch.zeros(4),
        "window_ends": torch.tensor([0.0, 1000.0, 1000.0, 1000.0]),
        "demand": torch.tensor([0.0, 0.1, 0.1, 0.1]),
        "source_ids": [0, 4, 5, 6],
    }


def make_scenario(problem, actual, sampled, monkeypatch, *, ortec=False):
    """Fix customer draws so the service sampling can be checked in isolation."""
    state = problem.make_state(actual)
    fake_problem = SimpleNamespace(make_dataset=lambda **kwargs: [sampled])
    context = None
    if ortec:
        monkeypatch.setattr(msa, "sample_ortec_scenario", lambda *args: sampled)
        context = {"distance": np.ones((7, 7)) - np.eye(7)}
    return msa.make_scenario(
        fake_problem, problem.NAME, actual, state, 4, 0.5, 1, 2,
        ortec=context, actual_source_ids=[0, 1, 2, 3] if ortec else None,
    )


def customer_indices(scenario):
    return [index for index, node in enumerate(scenario["mapping"]) if index and node != 0]


def assert_service_roles(scenario, expected_mean=None):
    assert scenario["service"][0] == 0.0  # Current service has already completed.
    for index in scenario["reload_nodes"]:
        assert scenario["service"][index] == 2.0  # Known depot handling duration.
    if expected_mean is not None:
        for index in customer_indices(scenario):
            assert scenario["service"][index] == expected_mean


@pytest.mark.parametrize("problem", PROBLEMS)
def test_online_planning_uses_mean_without_unserved_realizations(problem):
    actual = actual_input(mean=6.5, std=2.0)
    state = problem.make_state(actual)
    hidden_changed = deepcopy(actual)
    hidden_changed["service_times"][0, 1:] = torch.tensor([120.0, 170.0, 230.0])
    original_services = actual["service_times"].clone()

    first = online.online_scenario(problem.NAME, actual, state, 2)
    second = online.online_scenario(problem.NAME, hidden_changed, state, 2)

    assert first["mapping"] == second["mapping"]
    np.testing.assert_array_equal(first["service"], second["service"])
    assert_service_roles(first, expected_mean=6.5)
    torch.testing.assert_close(actual["service_times"], original_services)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_online_legacy_inputs_use_known_default_not_empirical_realizations(problem):
    actual = actual_input()
    del actual["service_times_mean"]
    del actual["service_times_std"]
    state = problem.make_state(actual)

    scenario = online.online_scenario(problem.NAME, actual, state, 2)

    assert_service_roles(scenario, expected_mean=3.0)


def test_distribution_parameters_do_not_estimate_from_realized_services():
    actual = {"service_times": torch.tensor([[0.0, 120.0, 240.0]])}
    mean, std = service_time_parameters(actual)
    assert mean == 3.0
    assert std == pytest.approx(math.sqrt(5.0))


@pytest.mark.parametrize(("mean", "std"), [(0.0, 0.0), (6.5, 0.0)])
def test_zero_variance_service_sampling_is_deterministic(mean, std):
    actual = actual_input(mean, std)

    values = sample_service_times(actual, 7)

    np.testing.assert_array_equal(values, np.full(7, mean))


def test_lognormal_service_samples_follow_configured_mean_and_standard_deviation():
    actual = actual_input(mean=5.0, std=2.0)
    np.random.seed(914)

    values = sample_service_times(actual, 50000)

    assert np.all(values > 0)
    assert np.mean(values) == pytest.approx(5.0, abs=0.04)
    assert np.std(values) == pytest.approx(2.0, abs=0.07)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("ortec", [False, True], ids=["synthetic", "ortec"])
def test_msa_service_draws_ignore_actual_and_future_realized_durations(
    problem, ortec, monkeypatch
):
    actual = actual_input()
    sampled = future_sample()
    original_actual_services = actual["service_times"].clone()
    original_future_services = sampled["service_times"].clone()
    np.random.seed(2026)
    first = make_scenario(problem, actual, sampled, monkeypatch, ortec=ortec)
    changed_actual = deepcopy(actual)
    changed_actual["service_times"][0, 1:] += 1000.0
    changed_future = deepcopy(sampled)
    changed_future["service_times"][1:] += 2000.0
    np.random.seed(2026)
    second = make_scenario(problem, changed_actual, changed_future, monkeypatch, ortec=ortec)

    assert first["mapping"] == second["mapping"]
    np.testing.assert_array_equal(first["service"], second["service"])
    values = first["service"][customer_indices(first)]
    assert len(values) == 4  # Two revealed customers and two sampled future customers.
    assert np.all(values > 0)
    assert len(np.unique(values)) > 1
    assert_service_roles(first)
    torch.testing.assert_close(actual["service_times"], original_actual_services)
    torch.testing.assert_close(sampled["service_times"], original_future_services)


def test_msa_draws_are_reproducible_and_independent_between_scenarios(monkeypatch):
    actual = actual_input()
    sampled = future_sample()
    np.random.seed(392)
    first = make_scenario(PDTRP, actual, sampled, monkeypatch)
    second = make_scenario(PDTRP, actual, sampled, monkeypatch)
    np.random.seed(392)
    repeated = make_scenario(PDTRP, actual, sampled, monkeypatch)

    np.testing.assert_array_equal(first["service"], repeated["service"])
    assert not np.array_equal(first["service"], second["service"])


@pytest.mark.parametrize("problem", PROBLEMS)
def test_msa_zero_variance_replaces_both_known_and_future_services(problem, monkeypatch):
    actual = actual_input(mean=6.5, std=0.0)

    scenario = make_scenario(problem, actual, future_sample(), monkeypatch)

    assert_service_roles(scenario, expected_mean=6.5)


def test_prepared_scenario_worker_does_not_resample_services(monkeypatch):
    scenario = {"service": np.array([0.0, 1.25, 7.5])}
    original = scenario["service"].copy()

    def solve(data, *args, **kwargs):
        np.testing.assert_array_equal(data["service"], original)
        return {"route": [1, 2], "elapsed_seconds": 0.0}

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared scenario worker sampled service times")

    monkeypatch.setattr(msa, "solve_scenario", solve)
    monkeypatch.setattr(np.random, "lognormal", forbidden)

    details = msa.solve_prepared_scenario((scenario, "pdtrp", 10, 5))

    assert details["route"] == [1, 2]
    np.testing.assert_array_equal(scenario["service"], original)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_realized_service_times_still_advance_execution_clock(problem):
    actual = actual_input(mean=100.0, std=0.0)
    state = problem.make_state(actual)

    advanced = state.update(torch.tensor([1]))

    assert advanced.cur_times.item() == pytest.approx(13.0)  # One travel + twelve service.
    times, _ = problem.get_times(actual, torch.tensor([[0, 1, 2]]))
    assert times[0, 2].item() == pytest.approx(14.4142, abs=1e-4)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_clairvoyant_plans_with_realized_service_times(problem, monkeypatch):
    actual = actual_input(mean=100.0, std=0.0)
    observed = []

    def solve(scenario, *args, **kwargs):
        observed.append(scenario)
        route = []
        reloads = iter(sorted(scenario["reload_nodes"]))
        load = 0.0
        for index, node in enumerate(scenario["mapping"]):
            if node in (None, 0):
                continue
            if len(scenario["demand"]):
                demand = scenario["demand"][index]
                if load + demand > scenario["capacity"]:
                    route.append(next(reloads))
                    load = 0.0
                load += demand
            route.append(index)
        return {
            "route": route,
            "elapsed_seconds": 0.0, "solver_objective": 0.0, "solver_status": 1,
            "termination_reason": "test", "improvement_count": 0,
        }

    monkeypatch.setattr(clairvoyant, "solve_scenario", solve)
    args = SimpleNamespace(
        problem=problem.NAME, max_trips=2, static_time_ms=10,
        static_no_improvement_ms=5,
    )

    clairvoyant.evaluate_static(args, problem, actual)

    for scenario in observed:
        for index, node in enumerate(scenario["mapping"]):
            if node is not None:
                assert scenario["service"][index] == actual["service_times"][0, node].item()
