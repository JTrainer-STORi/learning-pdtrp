"""Specialists keep training and checkpoint selection at the requested N/DoD."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from options import get_options
from problems import PDCTRPTW, PDTRPTW
from run import run_rl
from utils import load_model
from utils.data_utils import dataset_to_input
from utils.training_data import make_training_dataset, make_validation_datasets


@pytest.mark.parametrize('problem', [PDTRPTW, PDCTRPTW])
@pytest.mark.parametrize('dod', [.2, .5, .8])
def test_fixed_training_and_validation_have_exact_customer_and_dynamic_counts(problem, dod):
    opts = get_options([
        '--problem', problem.NAME, '--min-customers', '50', '--max-customers', '50',
        '--min_dod', str(dod), '--max_dod', str(dod), '--validation-dods', str(dod),
        '--time-horizon-per-customer', '.2', '--training-data-seed', '0',
        '--epoch_size', '4', '--batch_size', '2', '--rollout_size', '4',
        '--rollout_batch_size', '2', '--val_size', '4', '--val_batch_size', '1',
    ])
    training = make_training_dataset(problem, opts, 0)
    np.random.seed(51)
    expected_next = np.random.random()
    np.random.seed(51)
    validation = make_validation_datasets(problem, opts)
    assert np.random.random() == expected_next
    assert len(validation) == 1
    for dataset in (training, validation[0]):
        for sample in dataset:
            assert sample['all_nodes'].shape == (51, 2)
            assert sample['time_horizon'].item() == 600
            assert int((sample['arrival_times'] > 0).sum()) == int(50 * dod)
    opts.seed = 1
    other_seed_validation = make_validation_datasets(problem, opts)[0]
    for i in range(len(validation[0])):
        for key in validation[0][i]:
            torch.testing.assert_close(validation[0][i][key], other_seed_validation[i][key], rtol=0, atol=0)


def test_validation_dod_selection_does_not_change_epoch_training_samples():
    argv = ['--min-customers', '20', '--max-customers', '30', '--training-data-seed', '9',
            '--epoch_size', '4', '--batch_size', '2', '--rollout_size', '4', '--rollout_batch_size', '2']
    default = make_training_dataset(PDTRPTW, get_options(argv), 0)
    specialist = make_training_dataset(PDTRPTW, get_options(argv + ['--validation-dods', '.5']), 0)
    for i in range(len(default)):
        for key in default[i]:
            torch.testing.assert_close(default[i][key], specialist[i][key], rtol=0, atol=0)


@pytest.mark.parametrize('value', ['-0.1', '1.1', 'nan', 'inf'])
def test_invalid_validation_dods_are_rejected(value):
    with pytest.raises(SystemExit):
        get_options(['--validation-dods', value])


def test_validation_sources_are_exclusive_and_resume_keeps_specialist_dod(tmp_path):
    with pytest.raises(SystemExit):
        get_options(['--validation-dods', '.5', '--val_datasets', 'holdout.txt'])
    (tmp_path / 'args.json').write_text(json.dumps({'validation_dods': [.5]}))
    resume = ['--resume', str(tmp_path / 'epoch-0.pt')]
    assert get_options(resume).validation_dods == [.5]
    assert get_options(resume + ['--validation-dods', '.2', '.8']).validation_dods == [.2, .8]
    opts = get_options(resume + ['--val_datasets', 'holdout.txt'])
    assert opts.validation_dods is None and opts.val_datasets == ['holdout.txt']
    assert get_options([]).validation_dods is None


@pytest.mark.parametrize('problem', [PDTRPTW, PDCTRPTW])
def test_specialist_training_selects_and_loads_checkpoint_using_one_validation_dod(tmp_path, problem):
    opts = get_options([
        '--problem', problem.NAME, '--min-customers', '10', '--max-customers', '10',
        '--min_dod', '.5', '--max_dod', '.5', '--validation-dods', '.5',
        '--time-horizon-per-customer', '.2', '--training-data-seed', '0',
        '--n_epochs', '2', '--epoch_size', '4', '--batch_size', '2',
        '--rollout_size', '4', '--rollout_batch_size', '2', '--val_size', '2',
        '--val_batch_size', '1', '--val_every', '1', '--checkpoint_epochs', '1',
        '--save-best-validation', '--embedding_dim', '16', '--hidden_dim', '16',
        '--n_heads', '2', '--n_encode_layers', '1', '--model', 'dattention',
        '--gated', '--learn_norm', '--normalization', 'batch', '--neighbors', '.2',
        '--knn_strat', 'percentage', '--use_time_feature', '--scale_times', '--use_arrival_times',
        '--no_cuda', '--no_wandb', '--no_tensorboard', '--no_videos', '--no_progress_bar',
        '--output_dir', str(tmp_path),
    ])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run_rl(opts)
        folder = Path(opts.save_dir)
        selected = json.loads((folder / 'best-validation.json').read_text())
        history = list(csv.DictReader((folder / 'history.csv').open()))
        assert all('val2_objective' not in row for row in history)
        expected = min(history, key=lambda row: float(row['val1_objective']))
        assert selected['epoch'] == int(expected['epoch'])
        assert selected['mean_validation_objective'] == pytest.approx(float(expected['val1_objective']))
        assert list(selected['validation_objectives']) == ['val1_objective']
        model, args = load_model(str(folder / 'best-validation.pth'))
        assert args['validation_dods'] == [.5] and not args['use_arrival_lstm']
        model.eval()
        model.set_decode_type('greedy')
        samples = make_validation_datasets(problem, opts)[0]
        costs = []
        with torch.inference_mode():
            for sample in samples:
                cost, _, _ = model(dataset_to_input(sample, problem.NAME, 'cpu'), mode='recursive')
                costs.append(float(cost.item()))
        assert np.mean(costs) == pytest.approx(selected['mean_validation_objective'], rel=1e-5)
    finally:
        torch.set_num_threads(previous_threads)
