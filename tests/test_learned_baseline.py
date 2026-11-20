import csv
from pathlib import Path

import pytest
import torch

from options import get_options
from reinforce_baselines import LearnedBaseline
from run import run_rl
from utils import load_model, torch_load_cpu


def test_learned_baseline_has_separate_gradients_and_initialization_rng():
    rng = torch.get_rng_state()
    baseline = LearnedBaseline(8, torch.device('cpu'), warmup_epochs=1, exp_beta=0.8)
    assert torch.equal(rng, torch.get_rng_state())
    features = torch.randn(2, 8, requires_grad=True)
    cost = torch.tensor([20., 40.], requires_grad=True)
    value, loss = baseline.eval(features, cost, customer_count=10)
    assert value.item() == 30.0 and not value.requires_grad
    loss.backward()
    assert features.grad is None and cost.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in baseline.get_learnable_parameters())

    with torch.no_grad():
        for parameter in baseline.get_learnable_parameters():
            parameter.zero_()
        baseline.predictor[-1].bias.fill_(2.)
    baseline.epoch_callback(None, 0)
    value, loss = baseline.eval(features, cost, customer_count=10)
    torch.testing.assert_close(value, torch.tensor([20., 20.]))
    assert loss.item() == 2.0
    assert baseline.warmup.v.item() == 30.0


@pytest.mark.parametrize('algorithm,encoder', [('reinforce', 'gnn'), ('ppo', 'transformer')])
def test_learned_baseline_trains_and_resumes_without_greedy_rollout(tmp_path, monkeypatch, algorithm, encoder):
    def unexpected_rollout(*args, **kwargs):
        pytest.fail('A learned baseline does not run the greedy baseline simulator')

    monkeypatch.setattr('reinforce_baselines.rollout_baseline', unexpected_rollout)
    common = [
        '--problem', 'pdctrptw', '--encoder', encoder, '--use-time-window-slack',
        '--use_time_feature', '--use_arrival_times', '--scale_times',
        '--min-customers', '5', '--max-customers', '8', '--time-horizon-per-customer', '.2',
        '--epoch_size', '4', '--batch_size', '2', '--val_size', '2', '--val_batch_size', '1',
        '--validation-dods', '.5', '--embedding_dim', '16', '--n_encode_layers', '1', '--n_heads', '2',
        '--no_cuda', '--no_wandb', '--no_tensorboard', '--no_progress_bar', '--save-best-validation',
        '--gated', '--output_dir', str(tmp_path),
    ]
    opts = get_options(common + ['--n_epochs', '2', '--training-algorithm', algorithm, '--baseline', 'learned'])
    run_rl(opts)
    root = Path(opts.save_dir)
    first = torch_load_cpu(root / 'epoch-0.pt')
    final = torch_load_cpu(root / 'epoch-1.pt')
    assert len(final['optimizer']['param_groups']) == 2
    initial_head, trained_head = [ckpt['baseline']['predictor'] for ckpt in (first, final)]
    assert any(not torch.equal(initial_head[name], trained_head[name]) for name in initial_head)
    with (root / 'history.csv').open() as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == 2 and all(row['val1_objective'] for row in history)
    if algorithm == 'ppo':
        assert all(float(row['learned_baseline_loss']) > 0 for row in history)

    model, args = load_model(str(root / 'best-validation.pth'))
    assert args['baseline'] == 'learned' and not any('predictor' in key for key in model.state_dict())
    resumed = get_options(common + ['--n_epochs', '1', '--run_name', 'resumed', '--resume', str(root / 'epoch-1.pt')])
    assert resumed.baseline == 'learned' and resumed.training_algorithm == algorithm
    run_rl(resumed)
    checkpoint = torch_load_cpu(Path(resumed.save_dir) / 'epoch-2.pt')
    assert any(not torch.equal(trained_head[name], checkpoint['baseline']['predictor'][name])
               for name in trained_head)
