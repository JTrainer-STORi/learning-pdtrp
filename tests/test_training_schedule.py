from types import SimpleNamespace

import pytest
import torch

from options import get_options
from problems import PDCTRP, PDCTRPTW
from reinforce_baselines import RolloutBaseline
from train import rollout_groundtruth, should_run_periodic


def test_periodic_work_runs_on_interval_and_final_epoch():
    due = [
        epoch
        for epoch in range(100)
        if should_run_periodic(epoch, every=5, epoch_start=0, n_epochs=100)
    ]

    assert due == list(range(4, 100, 5))


def test_periodic_work_runs_on_final_partial_interval():
    due = [
        epoch
        for epoch in range(7, 10)
        if should_run_periodic(epoch, every=5, epoch_start=7, n_epochs=3)
    ]

    assert due == [9]


@pytest.mark.parametrize(
    "flag", ["--val_every", "--baseline_update_every"]
)
def test_training_intervals_must_be_positive(flag):
    with pytest.raises(SystemExit):
        get_options([flag, "0"])


def test_new_training_options_preserve_existing_defaults():
    opts = get_options([])

    assert opts.val_every == 1
    assert opts.baseline_update_every == 1


def test_rollout_baseline_skips_challenge_between_intervals(monkeypatch):
    baseline = RolloutBaseline.__new__(RolloutBaseline)
    baseline.opts = SimpleNamespace(
        baseline_update_every=5,
        epoch_start=0,
        n_epochs=100,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("rollout should have been skipped")

    monkeypatch.setattr("reinforce_baselines.rollout_baseline", fail_if_called)

    baseline.epoch_callback(model=object(), epoch=2)


@pytest.mark.parametrize(
    ("problem_name", "problem", "uses_time_windows"),
    (
        ("pdctrp", PDCTRP, False),
        ("pdcvrp", PDCTRP, False),
        ("pdctrptw", PDCTRPTW, True),
        ("pdcvrptw", PDCTRPTW, True),
    ),
)
def test_capacity_groundtruth_supports_canonical_and_legacy_names(
    problem_name, problem, uses_time_windows, monkeypatch
):
    sample = {
        "all_nodes": torch.zeros(3, 2),
        "distance_matrix": torch.zeros(3, 3),
        "demand": torch.tensor([0.0, 0.2, 0.2]),
        "vehicle_capacity": torch.tensor([1.0]),
        "tour_nodes": torch.tensor([0, 1, 2]),
    }
    if uses_time_windows:
        sample.update(
            window_ends=torch.ones(3),
            visit_times=torch.zeros(3),
            gamma=torch.tensor([1.0]),
            theta=torch.tensor([1.0]),
        )

    calls = []

    def fake_get_costs(*args):
        calls.append(args)
        return torch.tensor([1.0]), {}

    monkeypatch.setattr(problem, "get_costs", fake_get_costs)
    opts = SimpleNamespace(
        problem=problem_name,
        val_batch_size=1,
        num_workers=0,
        device=torch.device("cpu"),
    )

    costs, _ = rollout_groundtruth(problem, [sample], opts)

    torch.testing.assert_close(costs, torch.tensor([1.0]))
    assert len(calls) == 1
