"""Chapter 7 openings, narrow/wide presets, and scenario-sampling consistency."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import evaluate_msa_ortools as msa
from options import get_options
from problems import PDCTRPTW, PDTRPTW
from problems.time_windows import generate_time_windows, resolve_time_window_bounds
from utils.data_utils import dataset_to_input


def make_dataset(problem, **overrides):
    kwargs = dict(
        min_total=8, max_total=8, min_dod=0.5, max_dod=0.5,
        n_subregions=1, batch_size=2, num_samples=2,
        neighbors=2, knn_strat="fixed",
    )
    if problem is PDCTRPTW:
        kwargs.update(vehicle_capacity=1, min_trips_required_lb=1, min_trips_required_ub=2)
    kwargs.update(overrides)
    return problem.make_dataset(**kwargs)


@pytest.mark.parametrize("problem", [PDTRPTW, PDCTRPTW])
@pytest.mark.parametrize("scenario,minimum,maximum", [("narrow", 30, 90), ("wide", 60, 180)])
@pytest.mark.parametrize("dod", [0.0, 0.5, 1.0])
def test_generated_windows_follow_chapter7_openings_and_selected_widths(problem, scenario, minimum, maximum, dod):
    dataset = make_dataset(problem, time_window_scenario=scenario, min_dod=dod, max_dod=dod)
    for item in dataset:
        calls = item["arrival_times"][1:]
        travel = item["distance_matrix"][0, 1:] / item["speed"].item()
        starts, ends = item["window_starts"][1:], item["window_ends"][1:]
        dynamic = calls > 0
        torch.testing.assert_close(starts[dynamic], (calls + travel)[dynamic], atol=0.0011, rtol=0)
        assert torch.all(starts[~dynamic] >= travel[~dynamic] - 0.0011)
        assert torch.all(ends - starts >= minimum - 0.0011)
        assert torch.all(ends - starts <= maximum + 0.0011)
        assert torch.all(ends <= item["time_horizon"] + item["latest_end"] + 0.0011)
        assert item["min_time_window"].item() == minimum
        assert item["max_time_window"].item() == maximum
        assert item["window_starts"][0] == item["window_ends"][0] == 0


@pytest.mark.parametrize("problem", [PDTRPTW, PDCTRPTW])
def test_synthetic_windows_do_not_use_legacy_reaction_time(problem):
    np.random.seed(75)
    first = make_dataset(problem, reaction_time=0)[0]
    np.random.seed(75)
    second = make_dataset(problem, reaction_time=600)[0]
    for key in ("arrival_times", "window_starts", "window_ends", "all_nodes"):
        torch.testing.assert_close(first[key], second[key])


def test_late_call_caps_closing_time_without_reducing_minimum_width():
    starts, ends = generate_time_windows([480], [10], 60, 180, 600)
    assert starts.tolist() == [490]
    assert 550 <= ends[0] <= 600


def test_impossible_window_fails_instead_of_sampling_reversed_bounds():
    with pytest.raises(ValueError, match="Cannot fit"):
        generate_time_windows([480], [70], 60, 180, 600)


@pytest.mark.parametrize("make_rng", [np.random.RandomState, np.random.default_rng])
def test_window_draws_preserve_customers_and_random_stream_across_dynamism(make_rng):
    releases = np.arange(1, 7) * 30.0
    travel = np.arange(1, 7, dtype=float)
    previous = None
    following_draws = None
    for dynamic_count in (0, 2, 4, 6):
        rng = make_rng(20)
        calls = releases.copy()
        calls[dynamic_count:] = 0
        starts, ends = generate_time_windows(calls, travel, 60, 180, 600, rng=rng)
        tail = rng.random(5)
        if previous is not None:
            previous_calls, previous_starts, previous_ends = previous
            unchanged = calls == previous_calls
            np.testing.assert_array_equal(starts[unchanged], previous_starts[unchanged])
            np.testing.assert_array_equal(ends[unchanged], previous_ends[unchanged])
            np.testing.assert_array_equal(tail, following_draws)
        previous = calls, starts, ends
        following_draws = tail


@pytest.mark.parametrize("minimum,maximum", [(-1, 90), (100, 30), (30, float("nan"))])
def test_invalid_width_bounds_are_rejected(minimum, maximum):
    with pytest.raises(ValueError, match="bounds"):
        resolve_time_window_bounds("wide", minimum, maximum)


@pytest.mark.parametrize("scenario,expected", [("narrow", (30, 90)), ("wide", (60, 180))])
def test_training_and_msa_cli_resolve_matching_presets(scenario, expected):
    training = get_options(["--time_window_scenario", scenario])
    evaluation = msa.parse_args([
        "--problem", "pdtrptw", "--dataset", "data.txt", "--output", "out.json",
        "--dod", "0.5", "--time-window-scenario", scenario,
    ])
    assert (training.min_time_window, training.max_time_window) == expected
    assert (evaluation.min_time_window, evaluation.max_time_window) == expected


def test_explicit_widths_override_preset_defaults():
    assert resolve_time_window_bounds("wide", 60, 100) == (60, 100)


@pytest.mark.parametrize("problem", [PDTRPTW, PDCTRPTW])
def test_msa_future_windows_use_observed_width_prior_and_depot_opening_rule(problem):
    item = make_dataset(problem, batch_size=1, num_samples=1, time_window_scenario="narrow")[0]
    # Exercise metadata propagation through the batched evaluation path too.
    actual = dataset_to_input(next(iter(DataLoader([item], batch_size=1))), problem.NAME, "cpu")
    assert actual["min_time_window"].item() == 30
    assert actual["max_time_window"].item() == 90
    state = problem.make_state(actual)
    scenario = msa.make_scenario(problem, problem.NAME, actual, state, 8, 0.5, 1, 2)
    future = [i for i, node in enumerate(scenario["mapping"]) if i and node is None]
    assert future
    for node in future:
        opening, closing = scenario["windows"][node]
        call = scenario["arrival_times"][node]
        depot_travel = scenario["distance"][0, node] / scenario["speed"]
        assert opening == pytest.approx(call + depot_travel, abs=0.002)
        assert 30 - 0.002 <= closing - opening <= 90 + 0.002
