"""Global set attention, routing integration, and checkpoint compatibility."""

import json
from pathlib import Path

import pytest
import torch

from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.factory import get_encoder_class
from nets.encoders.gnn_encoder import GNNEncoder
from nets.encoders.transformer_encoder import TransformerEncoder
from options import get_options
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from run import run_rl
from utils import load_args, load_model


def routing_inputs(problem, batch_size=1):
    coords = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 2.], [2., 2.]]])
    data = dict(
        all_nodes=coords, distance_matrix=torch.cdist(coords, coords),
        graph=(~torch.eye(5, dtype=torch.bool))[None], speed=torch.ones(1, 1),
        time_horizon=torch.tensor([[30.]]), latest_end=torch.tensor([[10.]]),
        arrival_times=torch.tensor([[0., 0., 1.5, 0., 2.5]]),
        service_times=torch.tensor([[0., 2., 0., 1., 0.]]),
        window_starts=torch.tensor([[0., 0., 3., 0., 4.]]),
        window_ends=torch.tensor([[100., 12., 16., 18., 22.]]),
        gamma=torch.ones(1, 1), theta=torch.ones(1, 1),
    )
    if problem in (PDCTRP, PDCTRPTW):
        data.update(demand=torch.tensor([[0., .65, .45, .5, .5]]),
                    vehicle_capacity=torch.ones(1, 1))
    data = {k: v.repeat(batch_size, *([1] * (v.ndim - 1))) for k, v in data.items()}
    if batch_size == 2:
        data['arrival_times'][0].zero_()
        data['window_starts'][0].zero_()
        data['arrival_times'][1] = torch.tensor([0., 1., 5., 15., 25.])
        data['window_starts'][1] = torch.tensor([0., 20., 20., 30., 40.])
        data['window_ends'][1] = torch.tensor([100., 40., 40., 50., 60.])
    return data


def routing_model(problem, slack=False, lstm=False):
    return DynamicAttentionModel(
        problem, embedding_dim=16, encoder_class=TransformerEncoder,
        n_encode_layers=2, n_heads=4, normalization='layer',
        use_time_feature=True, use_arrival_times=True, scale_times=True,
        use_time_window_slack=slack, neighbors=-1, knn_strat='fixed',
        use_arrival_lstm=lstm,
        transformer_resblock_hidden=24,
    )


def test_encoder_is_permutation_equivariant_and_ignores_knn_edges():
    encoder = TransformerEncoder(hidden_dim=16, transformer_resblock_hidden=24).eval()
    x = torch.randn(2, 6, 16)
    permutation = torch.tensor([4, 0, 5, 2, 3, 1])
    a = encoder(x, graph=torch.zeros(2, 6, 6))
    b = encoder(x[:, permutation], graph=torch.ones(2, 6, 6), e=torch.randn(2, 6, 6, 16))
    torch.testing.assert_close(b, a[:, permutation])
    changed = x.clone()
    changed[:, -1] += torch.arange(16) * .5
    # Even an isolated token can affect another token through global attention.
    assert not torch.allclose(encoder(changed)[:, 0], a[:, 0])
    assert len(encoder.pre_blocks) == len(encoder.post_blocks) == 2
    assert len(encoder.attn_blocks) == 3
    assert encoder.attn_blocks[0].attn.num_heads == 2


def test_padding_cannot_change_valid_outputs_or_gradients():
    encoder = TransformerEncoder(hidden_dim=16, n_layers=2)
    x = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    output, attention = encoder(x, node_mask=valid, return_attention=True)
    assert output[~valid].eq(0).all()
    for row, count in enumerate((3, 2)):
        torch.testing.assert_close(output[row, :count], encoder(x[row:row+1, :count])[0])
    output.square().sum().backward()
    assert x.grad[~valid].eq(0).all()
    assert x.grad[valid].abs().sum() > 0
    for layer in attention.values():
        assert layer['weights'].masked_select(~valid[:, None, None, :]).eq(0).all()
    changed = x.detach().clone()
    changed[~valid] = 1000
    torch.testing.assert_close(encoder(changed, node_mask=valid), output.detach())


@pytest.mark.parametrize('bad_mask', [torch.zeros(2, 3, dtype=torch.bool), torch.ones(2, 2)])
def test_encoder_rejects_missing_valid_tokens_and_bad_masks(bad_mask):
    with pytest.raises(ValueError, match='nonpadding'):
        TransformerEncoder(hidden_dim=16)(torch.randn(2, 3, 16), node_mask=bad_mask)


def test_encoder_rejects_empty_graph():
    with pytest.raises(ValueError, match='nonempty'):
        TransformerEncoder(hidden_dim=16)(torch.empty(1, 0, 16))


@pytest.mark.parametrize('problem', [PDTRP, PDCTRP, PDTRPTW, PDCTRPTW])
@pytest.mark.parametrize('mode', ['masked', 'recursive', 'recursive_plus_removal'])
@pytest.mark.parametrize('lstm', [False, True])
def test_transformer_routing_visits_all_customers(problem, mode, lstm):
    model = routing_model(problem, slack=problem in (PDTRPTW, PDCTRPTW), lstm=lstm).eval()
    assert not hasattr(model, 'edge_embed')
    model.set_decode_type('greedy')
    with torch.no_grad():
        cost, _, _, tour = model(routing_inputs(problem), mode=mode, return_pi=True)
    assert torch.isfinite(cost).all()
    assert sorted(n for n in tour[0].tolist() if n) == [1, 2, 3, 4]


@pytest.mark.parametrize('problem', [PDTRPTW, PDCTRPTW])
@pytest.mark.parametrize('slack', [False, True])
def test_transformer_masked_training_keeps_cached_encoder_and_padding_gradients(problem, slack):
    model = routing_model(problem, slack=slack).train()
    model.set_decode_type('greedy')
    calls, padding = [], []
    handle = model.embedder.register_forward_hook(lambda *args: calls.append(True))
    original = model._get_log_p

    def inspect(fixed, state, mask=None, **kwargs):
        logp, action_mask = original(fixed, state, mask, **kwargs)
        finished = state.visited_.all(dim=-1).squeeze(1)
        if finished.any():
            padding.append(logp[finished, 0, 0].sum())
            assert logp[finished, 0, 0].eq(0).all()
        return logp, action_mask

    model._get_log_p = inspect
    cost, likelihood, _ = model(routing_inputs(problem, batch_size=2))
    assert calls == [True] and padding
    weights = [model.node_embed.weight, model.embedder.attn_blocks[0].attn.in_proj_weight]
    if slack:
        weights.append(model.project_time_window_slack.weight)
    padding_gradients = torch.autograd.grad(torch.stack(padding).sum(), weights, retain_graph=True)
    assert all(grad.eq(0).all() for grad in padding_gradients)
    (cost.detach() * likelihood).mean().backward()
    handle.remove()
    for weight in weights:
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0


@pytest.mark.parametrize('extra', [
    ['--transformer-heads', '3'], ['--transformer-heads', '0'],
    ['--transformer-resblocks-before', '-1'], ['--transformer-resblock-hidden', '0'],
    ['--n_encode_layers', '0'], ['--normalization', 'batch'], ['--use_incremental_encoder'],
])
def test_transformer_options_reject_invalid_configurations(extra):
    with pytest.raises(SystemExit):
        get_options(['--encoder', 'transformer'] + extra)


def test_factory_and_legacy_args_keep_gnn_default(tmp_path):
    assert get_options([]).encoder == 'gnn'
    assert get_encoder_class() is GNNEncoder
    with pytest.raises(ValueError, match='incremental'):
        get_encoder_class('transformer', True)
    args = tmp_path / 'args.json'
    args.write_text('{}')
    assert load_args(args)['encoder'] == 'gnn'


@pytest.mark.parametrize('problem', ['pdtrptw', 'pdctrptw'])
@pytest.mark.parametrize('slack', [False, True])
def test_transformer_training_load_and_resume(tmp_path, problem, slack):
    common = [
        '--problem', problem, '--min-customers', '6', '--max-customers', '6',
        '--min_dod', '.5', '--max_dod', '.5', '--validation-dods', '.5',
        '--use_time_feature', '--scale_times', '--use_arrival_times',
        '--time-horizon-per-customer', '.2', '--training-data-seed', '0',
        '--n_epochs', '1', '--epoch_size', '4', '--batch_size', '2',
        '--rollout_size', '4', '--rollout_batch_size', '2', '--bl_warmup_epochs', '0',
        '--val_size', '2', '--val_batch_size', '1', '--save-best-validation',
        '--checkpoint_epochs', '1', '--embedding_dim', '16', '--hidden_dim', '16',
        '--n_heads', '2', '--model', 'dattention', '--neighbors', '-1', '--knn_strat', 'fixed',
        '--no_cuda', '--no_wandb', '--no_tensorboard', '--no_videos', '--no_progress_bar',
    ]
    architecture = [
        '--encoder', 'transformer', '--n_encode_layers', '2', '--transformer-heads', '4',
        '--transformer-resblocks-before', '1', '--transformer-resblocks-after', '3',
        '--transformer-resblock-hidden', '24',
    ]
    if slack:
        architecture.append('--use-time-window-slack')
    opts = get_options(common + architecture + ['--output_dir', str(tmp_path / 'first')])
    run_rl(opts)
    model, args = load_model(str(Path(opts.save_dir) / 'best-validation.pth'))
    assert isinstance(model.embedder, TransformerEncoder)
    assert args['encoder'] == 'transformer' and args['use_time_window_slack'] == slack
    assert len(model.embedder.pre_blocks) == 1 and len(model.embedder.post_blocks) == 3
    assert model.embedder.attn_blocks[0].attn.num_heads == 4
    assert model.embedder.pre_blocks[0].ff[0].out_features == 24
    model.set_decode_type('greedy')
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(routing_inputs(PDTRPTW if problem == 'pdtrptw' else PDCTRPTW),
                                          mode='recursive')[0]).all()
    checkpoint = str(Path(opts.save_dir) / 'epoch-0.pt')
    for flag in ('--resume', '--load_path'):
        restored = get_options(common + [flag, checkpoint, '--output_dir', str(tmp_path / flag[2:])])
        for name in ('encoder', 'normalization', 'n_encode_layers', 'transformer_heads',
                     'transformer_resblocks_before', 'transformer_resblocks_after',
                     'transformer_resblock_hidden', 'use_time_window_slack'):
            assert getattr(restored, name) == getattr(opts, name)
    resumed = get_options(common + ['--resume', checkpoint, '--output_dir', str(tmp_path / 'resumed')])
    run_rl(resumed)
    loaded, _ = load_model(str(Path(resumed.save_dir) / 'epoch-1.pt'))
    assert isinstance(loaded.embedder, TransformerEncoder)
