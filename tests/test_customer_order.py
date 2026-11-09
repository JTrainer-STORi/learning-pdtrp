"""Customer indices stay attached to their sampled attributes across dynamism."""

import importlib

import numpy as np
import pytest
import torch

from utils import load_problem


PROBLEMS = ("pdtrp", "pdctrp", "pdtrptw", "pdctrptw")


def problem_module(name):
    directory = {"pdtrptw": "pdtrp_tw", "pdctrptw": "pdctrp_tw"}.get(name, name)
    return importlib.import_module(f"problems.{directory}.problem_{name}")


@pytest.mark.parametrize("name", PROBLEMS)
def test_synthetic_generation_keeps_original_customer_order(monkeypatch, name):
    module = problem_module(name)
    coords = np.array([[0.2, 0.4], [0.9, 0.1], [0.6, 0.7]])
    calls = np.array([30.0, 1.0, 10.0])
    service = np.array([[3.0, 1.0, 2.0]])
    demand = np.array([[0.2, 0.3, 0.1]])
    monkeypatch.setattr(module, "time_and_subregion_generation",
                        lambda **kwargs: (coords.copy(), calls.copy()))
    monkeypatch.setattr(module, "sample_lognorm", lambda *args, **kwargs: service.copy())
    if "c" in name:
        monkeypatch.setattr(module, "generate_scaled_demands",
                            lambda *args, **kwargs: demand.copy())

    dataset = load_problem(name).make_dataset(
        min_customers=3, max_customers=3, min_dod=1, max_dod=1,
        batch_size=1, num_samples=1, time_horizon=10,
        min_trips_required_lb=1, min_trips_required_ub=1,
    )
    sample = dataset[0]

    torch.testing.assert_close(sample["all_nodes"][1:], torch.tensor(coords, dtype=torch.float))
    torch.testing.assert_close(sample["arrival_times"][1:], torch.tensor(calls, dtype=torch.float))
    torch.testing.assert_close(sample["service_times"][1:], torch.tensor(service[0], dtype=torch.float))
    if "demand" in sample:
        torch.testing.assert_close(sample["demand"][1:], torch.tensor(demand[0], dtype=torch.float))
    if "window_starts" in sample:
        expected = calls + np.linalg.norm(coords - 0.5, axis=1) / (4 / 60)
        np.testing.assert_allclose(sample["window_starts"][1:], expected, atol=0.001)


@pytest.mark.parametrize("name", PROBLEMS)
def test_ortec_generation_preserves_sampled_indices_and_matrix(monkeypatch, name):
    coords = np.array([[0.5, 0.5], [0.2, 0.4], [0.9, 0.1], [0.6, 0.7]])
    calls = np.array([0.0, 30.0, 0.0, 10.0])
    service = np.array([0.0, 3.0, 1.0, 2.0])
    distance = np.array([[0, 2, 3, 4], [5, 0, 6, 7], [8, 9, 0, 10], [11, 12, 13, 0]]) / 100
    starts = np.array([0.0, 31.0, 15.0, 11.0])
    ends = np.array([0.0, 100.0, 90.0, 95.0])
    demand = np.array([0.0, 0.2, 0.3, 0.1])
    sampled = (coords, service, calls, distance, starts, ends, demand)
    monkeypatch.setattr(problem_module(name), "subsample_ortec",
                        lambda **kwargs: tuple(value.copy() for value in sampled))

    dataset = load_problem(name).make_dataset(
        min_customers=3, max_customers=3, min_dod=0.5, max_dod=0.5,
        batch_size=1, num_samples=1, use_ortec="unused.vrp",
    )
    sample = dataset[0]

    expected = {"all_nodes": coords, "service_times": service,
                "arrival_times": calls, "distance_matrix": distance}
    if "demand" in sample:
        expected["demand"] = demand
    if "window_starts" in sample:
        expected.update(window_starts=starts, window_ends=ends)
    for key, value in expected.items():
        torch.testing.assert_close(sample[key], torch.tensor(value, dtype=torch.float))


@pytest.mark.parametrize("name", PROBLEMS)
@pytest.mark.parametrize("seed", [20, 21])
@pytest.mark.parametrize("customers", [50, 100])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_same_seed_dynamism_variants_keep_indices_and_nested_calls(name, seed, customers, batch_size):
    datasets = []
    levels = (0.0, 0.2, 0.5, 0.8, 1.0)
    for dod in levels:
        np.random.seed(seed)
        torch.manual_seed(seed)
        datasets.append(load_problem(name).make_dataset(
            min_customers=customers, max_customers=customers, min_dod=dod, max_dod=dod,
            batch_size=batch_size, num_samples=4, n_subregions=1,
            arrival_weights=[1], arrival_skews=["uniform"], time_horizon=10,
        ))

    for index in range(4):
        reference = datasets[0][index]
        previous_calls = reference["arrival_times"]
        previous = reference
        for dod, dataset in zip(levels, datasets):
            sample = dataset[index]
            for key in ("all_nodes", "service_times", "distance_matrix", "demand"):
                if key in reference:
                    torch.testing.assert_close(sample[key], reference[key], rtol=0, atol=0)
            calls = sample["arrival_times"]
            assert int((calls > 0).sum()) == int(customers * dod)
            retained = previous_calls > 0
            assert (calls[retained] > 0).all()
            torch.testing.assert_close(calls[retained], previous_calls[retained], rtol=0, atol=0)
            if "window_starts" in sample:
                unchanged = calls == previous_calls
                for key in ("window_starts", "window_ends"):
                    torch.testing.assert_close(sample[key][unchanged], previous[key][unchanged], rtol=0, atol=0)
            previous_calls = calls
            previous = sample
