"""Check worker loading and sparse validation used by the HEC LSTM study."""

import csv
from pathlib import Path

import pytest
from options import get_options
from run import run_rl


@pytest.mark.parametrize("problem", ["pdtrptw", "pdctrptw"])
def test_paired_training_with_two_workers_keeps_final_validation_and_phase_timings(tmp_path, problem):
    epoch_seeds = []
    for lstm in (False, True):
        argv = [
            "--problem", problem, "--min-customers", "10", "--max-customers", "12",
            "--time-horizon-per-customer", ".2", "--n_epochs", "3",
            "--epoch_size", "4", "--batch_size", "2", "--rollout_size", "4",
            "--rollout_batch_size", "2", "--val_size", "1", "--val_batch_size", "1",
            "--num_workers", "2", "--val_every", "2", "--baseline_update_every", "1",
            "--training-data-seed", "17", "--embedding_dim", "16", "--hidden_dim", "16",
            "--model", "dattention", "--gated", "--learn_norm", "--normalization", "batch",
            "--neighbors", ".2", "--knn_strat", "percentage",
            "--n_heads", "2", "--n_encode_layers", "1", "--use_time_feature",
            "--scale_times", "--use_arrival_times", "--no_cuda", "--no_wandb",
            "--no_tensorboard", "--no_videos", "--no_progress_bar",
            "--output_dir", str(tmp_path / f"lstm-{lstm}"),
        ]
        if lstm:
            argv.append("--use_arrival_lstm")
        opts = get_options(argv)
        run_rl(opts)
        with (Path(opts.save_dir) / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        assert [int(row["epoch"]) for row in history] == [0, 1, 2]
        assert history[0]["val1_objective"] == ""
        assert all(float(row["val1_objective"]) > 0 for row in history[1:])
        assert (Path(opts.save_dir) / "epoch-2.pt").is_file()
        for row in history:
            phases = [float(row[key]) for key in (
                "data_generation_seconds", "baseline_precompute_seconds", "optimization_seconds",
                "checkpoint_seconds", "validation_seconds", "baseline_update_seconds",
            )]
            assert all(value >= 0 for value in phases)
            assert float(row["train_seconds"]) + .001 >= sum(phases[:3])
            assert float(row["epoch_seconds"]) + .001 >= sum(phases)
        epoch_seeds.append([row["epoch_data_seed"] for row in history])
    assert epoch_seeds[0] == epoch_seeds[1]
