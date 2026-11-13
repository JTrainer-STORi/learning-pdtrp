"""Keep benchmark selection on validation data and preserve older epoch loading."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from options import get_options
from run import run_rl
from train import save_best_validation
from utils import load_model, torch_load_cpu


def test_best_checkpoint_uses_all_validation_sets_and_keeps_earliest_tie(tmp_path):
    opts = SimpleNamespace(save_dir=str(tmp_path), validation_seed=12345)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    assert save_best_validation(model, 1, {'val1_objective': 10., 'val2_objective': 30.}, opts)
    initial = (tmp_path / 'best-validation.pth').read_bytes()
    with torch.no_grad():
        model.weight.fill_(2.)
    # One DoD improves, but the mean gets worse; do not select on that DoD alone.
    assert not save_best_validation(model, 3, {'val1_objective': 5., 'val2_objective': 50.}, opts)
    assert not save_best_validation(model, 5, {'val1_objective': 20., 'val2_objective': 20.}, opts)
    assert (tmp_path / 'best-validation.pth').read_bytes() == initial
    assert save_best_validation(model, 7, {'val1_objective': 8., 'val2_objective': 12.}, opts)
    selection = json.loads((tmp_path / 'best-validation.json').read_text())
    assert selection['epoch'] == 7
    assert selection['mean_validation_objective'] == 10.
    assert selection['validation_seed'] == 12345
    saved = torch_load_cpu(tmp_path / 'best-validation.pth')
    torch.testing.assert_close(saved['model']['weight'], torch.tensor([[2.]]))
    assert saved['validation'] == selection


@pytest.mark.parametrize('missing', [None, float('nan'), float('inf')])
def test_incomplete_or_nonfinite_validation_is_not_selected(tmp_path, missing):
    model = torch.nn.Linear(1, 1)
    opts = SimpleNamespace(save_dir=str(tmp_path), validation_seed=12345)
    assert not save_best_validation(model, 0, {'val1_objective': 2., 'val2_objective': missing}, opts)
    assert not (tmp_path / 'best-validation.pth').exists()


@pytest.mark.parametrize('problem', ['pdtrp', 'pdctrp', 'pdtrptw', 'pdctrptw'])
def test_best_checkpoint_loads_and_matches_history_without_changing_directory_default(tmp_path, problem):
    argv = [
        '--problem', problem, '--min-customers', '10', '--max-customers', '12',
        '--time-horizon-per-customer', '.2', '--n_epochs', '3',
        '--epoch_size', '4', '--batch_size', '2', '--rollout_size', '4',
        '--rollout_batch_size', '2', '--val_size', '2', '--val_batch_size', '1',
        '--val_every', '2', '--save-best-validation', '--training-data-seed', '17',
        '--embedding_dim', '16', '--hidden_dim', '16', '--n_heads', '2',
        '--n_encode_layers', '1', '--model', 'dattention', '--gated', '--learn_norm',
        '--normalization', 'batch', '--neighbors', '.2', '--knn_strat', 'percentage',
        '--no_cuda', '--no_wandb', '--no_tensorboard', '--no_videos', '--no_progress_bar',
        '--output_dir', str(tmp_path),
    ]
    if problem.endswith('tw'):
        argv += ['--use_time_feature', '--scale_times', '--use_arrival_times']
    opts = get_options(argv)
    assert not opts.use_arrival_lstm
    run_rl(opts)
    folder = Path(opts.save_dir)
    selection = json.loads((folder / 'best-validation.json').read_text())
    with (folder / 'history.csv').open() as handle:
        history = list(csv.DictReader(handle))
    scores = [(int(row['epoch']), sum(float(row[f'val{i}_objective']) for i in (1, 2, 3)) / 3)
              for row in history if row['val1_objective']]
    expected_epoch, expected_score = min(scores, key=lambda entry: entry[1])
    assert selection['epoch'] == expected_epoch
    assert selection['mean_validation_objective'] == pytest.approx(expected_score)
    best, _ = load_model(str(folder / 'best-validation.pth'))
    expected = torch_load_cpu(folder / f'epoch-{expected_epoch}.pt')['model']
    for key, value in best.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    # Existing directory loading must continue to choose the latest epoch file.
    latest, _ = load_model(str(folder))
    final = torch_load_cpu(folder / 'epoch-2.pt')['model']
    for key, value in latest.state_dict().items():
        torch.testing.assert_close(value, final[key], rtol=0, atol=0)


def test_best_validation_saving_is_opt_in():
    assert not get_options([]).save_best_validation
