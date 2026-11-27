import csv
from pathlib import Path

import pytest
import torch

from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from nets.encoders.transformer_encoder import TransformerEncoder
from options import get_options
from problems import PDCTRPTW, PDTRPTW
from run import run_rl
from tests.test_time_window_slack import inputs
from train import ppo_policy_loss
from utils import load_model


def test_ppo_clips_in_the_advantage_direction_and_ignores_padding():
    ratios = torch.tensor([[1.4, 0.7, 2.0], [0.7, 1.4, 2.0]])
    new = ratios.log().requires_grad_()
    active = torch.tensor([[True, True, False], [True, True, False]])
    loss, diagnostics = ppo_policy_loss(new, torch.zeros_like(new), torch.tensor([2., -3.]), active, 0.2)
    expected = -(2 * (1.2 + .7) - 3 * (.8 + 1.4)) / 2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    torch.testing.assert_close(new.grad, torch.tensor([[0., -.7, 0.], [0., 2.1, 0.]]))
    assert diagnostics[1:].tolist() == [4., 4.]


@pytest.mark.parametrize(('problem', 'lookahead'), [
    (PDTRPTW, 'off'), (PDTRPTW, 'direct'),
    (PDCTRPTW, 'off'), (PDCTRPTW, 'direct'), (PDCTRPTW, 'capacity-aware'),
])
@pytest.mark.parametrize('encoder', [GNNEncoder, TransformerEncoder])
@pytest.mark.parametrize('slack', [False, True])
def test_replaying_decisions_matches_rollout_and_reinforce_gradient(problem, encoder, slack, lookahead, monkeypatch):
    data = inputs(problem, batch_size=2)
    model = DynamicAttentionModel(
        problem, 16, encoder, 1, n_heads=2, normalization='batch' if encoder is GNNEncoder else 'layer',
        use_time_feature=True, scale_times=True, use_time_window_slack=slack,
        use_action_lookahead=lookahead == 'direct',
        use_capacity_aware_lookahead=lookahead == 'capacity-aware',
    )
    model.set_decode_type('greedy')

    cost, likelihood, _, tour = model(data, return_pi=True)
    advantage = -cost.detach()
    parameters = tuple(model.parameters())
    reinforce_grads = torch.autograd.grad(-(advantage * likelihood).mean(), parameters, allow_unused=True)
    with torch.no_grad():
        rollout_cost, old_log_p, trace = model(data, record_decisions=True)
    torch.testing.assert_close(rollout_cost, cost)
    assert not trace.active[0, -1] and trace.active[1, -1]
    # The first delayed choice at the depot is followed by reconsideration
    # at the depot and is recorded even though it is absent from the scored tour.
    assert trace.current_node[1, :2].tolist() == [0, 0]
    assert trace.active[1, :2].all() and trace.time[1, 0] < trace.time[1, 1]
    assert trace.active[1].sum() > (tour[1] != 0).sum()

    def unexpected_simulation(*_):
        pytest.fail('PPO replay should not initialize or advance the simulator')

    monkeypatch.setattr(problem, 'make_state', unexpected_simulation)
    new_log_p = model.score_decisions(data, trace)
    torch.testing.assert_close(new_log_p, old_log_p, atol=5e-6, rtol=1e-5)
    loss, _ = ppo_policy_loss(new_log_p, old_log_p, advantage, trace.active, .2)
    ppo_grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    for actual, expected in zip(ppo_grads, reinforce_grads):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=3e-4)


@pytest.mark.parametrize(('lookahead_flag', 'encoding', 'clipping'), [
    (None, 'full', 10), ('--use-action-lookahead', 'full', 10),
    ('--use-capacity-aware-lookahead', 'full', 10), ('--use-capacity-aware-lookahead', 'revealed', 10),
    ('--use-capacity-aware-lookahead', 'revealed', 0),
])
def test_ppo_trains_multiple_passes_and_saves_a_loadable_checkpoint(tmp_path, lookahead_flag, encoding, clipping):
    opts = get_options([
        '--problem', 'pdctrptw', '--encoder', 'transformer', '--use-time-window-slack',
        '--use_time_feature', '--use_arrival_times', '--scale_times',
        '--training-algorithm', 'ppo', '--ppo-update-epochs', '2', '--ppo-rollout-batches', '2',
        '--min-customers', '5', '--max-customers', '8', '--time-horizon-per-customer', '.2',
        '--n_epochs', '2', '--epoch_size', '6', '--batch_size', '2',
        '--rollout_size', '2', '--rollout_batch_size', '2', '--val_size', '2', '--val_batch_size', '1',
        '--embedding_dim', '16', '--n_encode_layers', '1', '--n_heads', '2',
        '--no_cuda', '--no_wandb', '--no_tensorboard', '--no_progress_bar', '--save-best-validation',
        '--output_dir', str(tmp_path), '--ppo-encoding', encoding,
    ] + ([lookahead_flag] if lookahead_flag else []) + (['--tanh_clipping', '0'] if clipping == 0 else []))
    run_rl(opts)
    save_dir = Path(opts.save_dir)
    with (save_dir / 'history.csv').open() as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == 2
    for row in history:
        assert int(row['training_instances']) == 6
        assert int(row['optimizer_updates']) == 6
        assert float(row['ppo_first_replay_max_abs_error']) < 2e-5
        if int(row['ppo_entropy_decisions']):
            assert float(row['ppo_sampled_relative_entropy']) >= 0
        else:
            assert row['ppo_sampled_relative_entropy'] == ''
        assert float(row['ppo_collection_seconds']) > 0 and float(row['ppo_update_seconds']) > 0
        assert float(row['ppo_approx_kl']) > 0
        assert 0 <= float(row['ppo_clip_fraction']) <= 1
    model, args = load_model(str(save_dir / 'best-validation.pth'))
    assert args['tanh_clipping'] == model.tanh_clipping == clipping
    assert args['training_algorithm'] == 'ppo' and args['ppo_update_epochs'] == 2
    assert model.use_action_lookahead == bool(lookahead_flag)
    assert model.use_capacity_aware_lookahead == (lookahead_flag == '--use-capacity-aware-lookahead')
    model.eval()
    model.set_decode_type('greedy')
    with torch.no_grad():
        cost, _, _ = model(inputs(PDCTRPTW), mode='recursive')
    assert torch.isfinite(cost).all()

    resumed = get_options(['--resume', str(save_dir / 'epoch-1.pt'), '--problem', 'pdctrptw'])
    assert resumed.ppo_encoding == encoding
    assert resumed.tanh_clipping == clipping
    assert resumed.training_algorithm == 'ppo' and resumed.ppo_rollout_batches == 2
    assert resumed.use_action_lookahead == model.use_action_lookahead
    assert resumed.use_capacity_aware_lookahead == model.use_capacity_aware_lookahead
