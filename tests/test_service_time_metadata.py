"""Keep service-time priors available without deriving them from realized times."""

import pytest
import torch
from torch.utils.data import DataLoader

from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from utils.data_utils import dataset_to_input


PROBLEMS = (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW)


def make_dataset(tmp_path, problem, loaded=False):
    options = {
        "batch_size": 1,
        "num_samples": 2,
        "service_times_mean": 7.0,
        "service_times_var": 4.0,
    }
    if loaded:
        path = tmp_path / "data.txt"
        row = "0 0 1 0 0 1 arrival_times 0 0 10 service_times 0 99 123 "
        if problem in (PDTRPTW, PDCTRPTW):
            row += "window_starts 0 0 10 window_ends 0 1000 1000 "
        if problem in (PDCTRP, PDCTRPTW):
            row += "demands 0 0.4 0.4 "
        row += "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n"
        path.write_text(row * 2)
        options["filename"] = str(path)
    else:
        options.update(
            min_total=5, max_total=5, min_dod=0.4, max_dod=0.4,
            n_subregions=1, neighbors=2, knn_strat="fixed",
        )
        if problem in (PDCTRP, PDCTRPTW):
            options.update(
                vehicle_capacity=1.0,
                min_trips_required_lb=2,
                min_trips_required_ub=2,
            )
    return problem.make_dataset(**options)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("pomo", [1, 3])
def test_configured_prior_survives_loading_batching_and_pomo(
    tmp_path, problem, loaded, batched, pomo
):
    dataset = make_dataset(tmp_path, problem, loaded)
    sample = next(iter(DataLoader(dataset, batch_size=2))) if batched else dataset[0]
    # Deliberately unrealistic latent durations must not alter the known prior.
    sample["service_times"] = torch.full_like(sample["service_times"], 999.0)

    inputs = dataset_to_input(sample, problem.NAME, torch.device("cpu"), pomo_batch_size=pomo)

    expected_batch = (2 if batched else 1) * pomo
    assert inputs["all_nodes"].shape[0] == expected_batch
    for key, expected in (("service_times_mean", 7.0), ("service_times_std", 2.0)):
        assert inputs[key].shape[0] == expected_batch
        torch.testing.assert_close(inputs[key], torch.full_like(inputs[key], expected))
    assert torch.all(inputs["service_times"] == 999.0)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("batched", [False, True])
def test_handcrafted_samples_can_omit_prior(tmp_path, problem, batched):
    dataset = make_dataset(tmp_path, problem)
    sample = next(iter(DataLoader(dataset, batch_size=2))) if batched else dataset[0]
    del sample["service_times_mean"]
    del sample["service_times_std"]

    inputs = dataset_to_input(sample, problem.NAME, torch.device("cpu"), pomo_batch_size=3)

    assert "service_times_mean" not in inputs
    assert "service_times_std" not in inputs
    assert inputs["all_nodes"].shape[0] == (2 if batched else 1) * 3


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("scalar", [False, True])
def test_global_batch_prior_broadcasts_before_pomo(tmp_path, problem, scalar):
    dataset = make_dataset(tmp_path, problem)
    sample = next(iter(DataLoader(dataset, batch_size=2)))
    sample["service_times_mean"] = torch.tensor(7.0) if scalar else torch.tensor([7.0])
    sample["service_times_std"] = torch.tensor(2.0) if scalar else torch.tensor([2.0])

    inputs = dataset_to_input(sample, problem.NAME, torch.device("cpu"), pomo_batch_size=3)

    torch.testing.assert_close(inputs["service_times_mean"], torch.full((6,), 7.0))
    torch.testing.assert_close(inputs["service_times_std"], torch.full((6,), 2.0))
