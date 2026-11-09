"""Scaled training workloads retain batching and reach every data source."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from options import get_options
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from problems.utils import transform_and_subsample_rl_instance
from reinforce_baselines import RolloutBaseline
from run import run_rl
from utils import load_model, torch_load_cpu
from utils.data_utils import dataset_to_input
from utils.training_data import make_validation_datasets


PROBLEMS = (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW)


def settings():
    return dict(batch_size=2, num_samples=4, n_subregions=1,
                arrival_weights=[1], arrival_skews=["uniform"],
                min_dod=.5, max_dod=.5, speed=4, neighbors=-1, knn_strat="fixed")


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize(("customers", "hours"), [(20, 4), (50, 10), (100, 20)])
def test_scaled_batches_match_explicit_fixed_horizon_generation(problem, customers, hours):
    np.random.seed(17)
    scaled = problem.make_dataset(min_customers=customers, max_customers=customers,
                                  time_horizon_per_customer=.2, **settings())
    np.random.seed(17)
    fixed = problem.make_dataset(min_customers=customers, max_customers=customers,
                                 time_horizon=hours, **settings())
    for i in range(len(scaled)):
        sample = scaled[i]
        assert sample["all_nodes"].shape == (customers + 1, 2)
        assert sample["time_horizon"].item() == hours * 60
        assert sample["arrival_times"].max() <= hours * 60
        assert int((sample["arrival_times"] > 0).sum()) == customers // 2
        # Includes calls, windows, demands and distances, so a reported horizon
        # alone cannot conceal generation with the wrong fixed horizon.
        for key in sample:
            torch.testing.assert_close(sample[key], fixed[i][key], rtol=0, atol=0)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_variable_customer_counts_keep_each_batch_homogeneous(problem):
    np.random.seed(5)
    options = settings()
    options["num_samples"] = 16
    dataset = problem.make_dataset(min_customers=20, max_customers=50,
                                   time_horizon_per_customer=.2, **options)
    counts = []
    for batch in DataLoader(dataset, batch_size=2, shuffle=False):
        n = batch["all_nodes"].shape[1] - 1
        counts.append(n)
        torch.testing.assert_close(batch["time_horizon"], torch.full((2, 1), n * 12.0))
        assert (batch["arrival_times"] <= batch["time_horizon"]).all()
    assert len(set(counts)) > 1


@pytest.mark.parametrize("problem", PROBLEMS)
def test_legacy_ortec_generation_receives_customer_scaled_horizon(monkeypatch, problem):
    module = importlib.import_module(problem.__module__)
    horizons = []

    def subsample(**kwargs):
        n = kwargs["n_total"]
        horizons.append(kwargs["time_horizon"])
        coords = np.column_stack((np.linspace(0, 1, n), np.zeros(n)))
        matrix = np.abs(coords[:, 0, None] - coords[None, :, 0])
        return coords, np.zeros(n), np.zeros(n), matrix, np.zeros(n), np.full(n, 240), np.zeros(n)

    monkeypatch.setattr(module, "subsample_ortec", subsample)
    dataset = problem.make_dataset(min_customers=20, max_customers=20,
                                   use_ortec="source.txt", time_horizon_per_customer=.2,
                                   **settings())
    assert horizons == [240] * 4
    assert dataset[0]["all_nodes"].shape == (21, 2)
    assert dataset[0]["time_horizon"].item() == 240


@pytest.mark.parametrize("problem", PROBLEMS)
def test_generated_validation_is_shared_and_does_not_consume_training_rng(problem):
    opts = get_options(["--problem", problem.NAME, "--min-customers", "20", "--max-customers", "30",
                        "--time-horizon-per-customer", "0.2", "--val_size", "2"])
    np.random.seed(19)
    expected_next = np.random.random()
    np.random.seed(19)
    first = make_validation_datasets(problem, opts)
    assert np.random.random() == expected_next
    np.random.seed(91)
    opts.seed = 91
    second = make_validation_datasets(problem, opts)
    assert len(first) == 3
    for a, b, dod in zip(first, second, (.2, .5, .8)):
        assert a[0]["time_horizon"].item() == 360
        assert int((a[0]["arrival_times"] > 0).sum()) == int(30 * dod)
        for key in a[0]:
            torch.testing.assert_close(a[0][key], b[0][key])


def test_rollout_baseline_generates_scaled_data(monkeypatch):
    opts = get_options(["--problem", "pdtrp", "--min-customers", "20", "--max-customers", "30",
                        "--time-horizon-per-customer", "0.2", "--batch_size", "2",
                        "--epoch_size", "4", "--rollout_batch_size", "2", "--rollout_size", "4"])
    inspected = []

    def rollout(model, dataset, args, mode):
        for item in dataset:
            assert item["time_horizon"].item() == (len(item["all_nodes"]) - 1) * 12
        inspected.append(True)
        return torch.ones(len(dataset)), []

    monkeypatch.setattr("reinforce_baselines.rollout_baseline", rollout)
    model = torch.nn.Linear(1, 1)
    baseline = RolloutBaseline(model, PDTRP, opts)
    assert inspected == [True]
    saved_dataset = baseline.dataset
    baseline._update_model(model, 1, dataset=saved_dataset)
    assert baseline.dataset is saved_dataset


@pytest.mark.parametrize("flag,value", [
    ("--time-horizon-per-customer", "0"), ("--time-horizon-per-customer", "-1"),
    ("--time-horizon-per-customer", "nan"), ("--time-horizon-per-customer", "inf"),
    ("--time_horizon", "0"), ("--validation-seed", "-1"),
    ("--validation-seed", "4294967296"), ("--epoch_size", "3"),
])
def test_training_rejects_invalid_workload_configuration(flag, value):
    with pytest.raises(SystemExit):
        get_options([flag, value])


def test_training_rejects_competing_horizons_and_incompatible_baseline_batches():
    with pytest.raises(SystemExit):
        get_options(["--time_horizon", "10", "--time-horizon-per-customer", ".2"])
    with pytest.raises(SystemExit):
        get_options(["--batch_size", "16", "--rollout_batch_size", "32"])
    get_options(["--baseline", "exponential", "--batch_size", "16", "--rollout_batch_size", "32"])


def test_resume_inherits_scaling_and_validation_seed_unless_overridden(tmp_path):
    (tmp_path / "args.json").write_text(json.dumps({
        "min_customers": 20, "max_customers": 100, "time_horizon": 8,
        "time_horizon_per_customer": .2, "validation_seed": 777,
    }))
    path = str(tmp_path / "epoch-0.pt")
    opts = get_options(["--resume", path])
    assert opts.time_horizon_per_customer == .2 and opts.validation_seed == 777
    fixed = get_options(["--resume", path, "--time_horizon", "7.5"])
    assert fixed.time_horizon == 7.5 and fixed.time_horizon_per_customer is None


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("lstm", [False, True])
@pytest.mark.parametrize("edge_features", ["adjacency", "distance"])
def test_scaled_training_checkpoint_and_recursive_evaluation(tmp_path, problem, lstm, edge_features):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        argv = ["--problem", problem.NAME, "--min-customers", "10", "--max-customers", "12",
                "--time-horizon-per-customer", ".2", "--n_epochs", "2", "--epoch_size", "4",
                "--batch_size", "2", "--rollout_size", "4", "--rollout_batch_size", "2",
                "--val_size", "1", "--val_batch_size", "1", "--embedding_dim", "16",
                "--hidden_dim", "16", "--n_heads", "2", "--n_encode_layers", "1",
                "--neighbors", ".2", "--knn_strat", "percentage", "--gated", "--learn_norm",
                "--edge_features", edge_features,
                "--normalization", "batch", "--seed", "0", "--checkpoint_epochs", "1",
                "--output_dir", str(tmp_path / "models"), "--no_cuda", "--no_wandb",
                "--log_dir", str(tmp_path / "logs"), "--no_videos", "--no_progress_bar"]
        tensorboard_enabled = problem is PDTRP and not lstm and edge_features == "adjacency"
        if not tensorboard_enabled:
            argv += ["--no_tensorboard"]
        if problem.NAME.endswith("tw"):
            argv += ["--use_time_feature", "--scale_times", "--use_arrival_times"]
        if lstm:
            argv += ["--use_arrival_lstm"]
        opts = get_options(argv)
        run_rl(opts)
        checkpoint = Path(opts.save_dir) / "epoch-1.pt"
        saved = torch_load_cpu(checkpoint)
        assert "numpy_rng_state" in saved
        assert all(torch.isfinite(value).all() for value in saved["model"].values())
        assert any(state["exp_avg"].abs().sum() > 0 for state in saved["optimizer"]["state"].values())
        if tensorboard_enabled:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events_file, = (tmp_path / "logs").rglob("events.out.tfevents.*")
            tags = EventAccumulator(str(events_file)).Reload().Tags()["scalars"]
            assert {"actor_loss", "grad_norm", "val1/avg_reward"} <= set(tags)
        model, args = load_model(str(checkpoint))
        assert args["time_horizon_per_customer"] == .2
        assert args["use_arrival_lstm"] == lstm
        model.eval()
        model.set_decode_type("greedy")
        sample = make_validation_datasets(problem, opts)[0][0]
        with torch.no_grad():
            cost, _, _ = model(dataset_to_input(sample, problem.NAME, "cpu"), mode="recursive")
        assert torch.isfinite(cost).all()

        if problem is PDCTRPTW and lstm and edge_features == "adjacency":
            # Resume past epoch zero with periodic checkpointing disabled:
            # the last additional epoch must still be saved.
            resumed_opts = get_options(argv + [
                "--resume", str(checkpoint), "--n_epochs", "1", "--checkpoint_epochs", "0",
                "--output_dir", str(tmp_path / "continued"),
            ])
            run_rl(resumed_opts)
            assert (Path(resumed_opts.save_dir) / "epoch-2.pt").is_file()
    finally:
        torch.set_num_threads(previous_threads)


def test_native_ortec_windows_honor_the_requested_horizon():
    source = {
        "time_windows": np.array([[0, 0], [120, 240], [240, 480]], dtype=float),
        "duration_matrix": np.array([[0, 60, 120], [60, 0, 60], [120, 60, 0]], dtype=float),
        "coords": np.array([[50, 50], [10, 10], [100, 100]], dtype=float),
        "service_times": np.array([0, 180, 180], dtype=float),
    }
    np.random.seed(3)
    sample = transform_and_subsample_rl_instance(
        source, "pdtrptw", 3, 240, 4 / 60, latest_end=120, n_imm=2, reaction_time=60,
    )
    assert sample[5].max() == 360  # T + latest_end, rather than the old 480 + latest_end.
    np.testing.assert_allclose(sample[2][1:], sample[4][1:] - 60)
    np.testing.assert_array_equal(sample[1], [0, 3, 3])
