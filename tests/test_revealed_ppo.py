from unittest.mock import patch

import pytest
import torch

from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.transformer_encoder import TransformerEncoder
from problems import PDCTRPTW
from tests.test_time_window_slack import inputs


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_revealed_collection_replay_and_online_decisions(device):
    data = {k: v.to(device) for k, v in inputs(PDCTRPTW, full=True, batch_size=2).items()}
    model = DynamicAttentionModel(
        PDCTRPTW, 16, TransformerEncoder, 2, n_heads=2, neighbors=-1, knn_strat="fixed",
        use_time_feature=True, use_arrival_times=True, scale_times=True, use_time_window_slack=True,
        use_capacity_aware_lookahead=True,
    ).to(device)
    model.set_decode_type("greedy")

    with torch.no_grad():
        cost, old, trace = model(data, mode="revealed", record_decisions=True)
    assert trace.visible[0].all() and not trace.visible[1, 0].all()
    assert not trace.active[0, -1] and trace.active[1, -1]
    assert trace.visible[1, :, 2].unique().numel() == 2
    replay = model.score_decisions(data, trace)
    torch.testing.assert_close(replay, old, atol=2e-5, rtol=1e-5)
    assert replay[~trace.active].eq(0).all()

    # Compare gradients to direct differentiable collection on the same actions.
    _, likelihood, _ = model(data, mode="revealed")
    parameters = tuple(model.parameters())
    reference = torch.autograd.grad(likelihood.sum(), parameters)
    actual = torch.autograd.grad(replay.sum(), parameters)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=1e-4)

    with torch.no_grad():
        model.eval()
        for i in range(2):
            row = {k: v[i:i+1] for k, v in data.items()}
            with patch.object(model, "_select_node", wraps=model._select_node) as select:
                model(row, mode="revealed")
                batched_choices = select.call_args_list
            with patch.object(model, "_select_node", wraps=model._select_node) as select:
                online, _, _ = model(row, mode="recursive")
                recursive_choices = select.call_args_list
            torch.testing.assert_close(online, cost[i:i+1])
            assert len(recursive_choices) == len(batched_choices) == trace.active[i].sum()
            for t, (batched, recursive) in enumerate(zip(batched_choices, recursive_choices)):
                visible = trace.visible[i, t]
                torch.testing.assert_close(batched.args[0][:, visible], recursive.args[0],
                                           atol=2e-5, rtol=1e-5)

        # Two legal first actions; change only an unrevealed request.
        row = {k: v.to(device) for k, v in inputs(PDCTRPTW, full=True).items()}
        altered = {k: v.clone() for k, v in row.items()}
        altered["demand"][0, 2] = .99
        altered["window_ends"][0, 2] += 100
        _, original, first = model(row, mode="revealed", record_decisions=True)
        _, changed, second = model(altered, mode="revealed", record_decisions=True)
        assert not first.visible[0, 0, 2]
        assert (~first.mask[0, 0]).sum() >= 2
        torch.testing.assert_close(original[:, 0], changed[:, 0])
        assert torch.equal(first.actions[:, 0], second.actions[:, 0])


def test_replay_spans_multiple_encoder_chunks_and_long_static_segments():
    # 129 independent graphs exceed one replay chunk. A repeated recorded
    # decision exercises long segments without requiring a large routing fixture.
    data = inputs(PDCTRPTW, full=True)
    data["arrival_times"].zero_()
    model = DynamicAttentionModel(PDCTRPTW, 16, TransformerEncoder, 1, n_heads=2)
    model.set_decode_type("greedy")
    with torch.no_grad():
        _, old, trace = model(data, mode="revealed", record_decisions=True)
    data = {k: v.expand(129, *v.shape[1:]) for k, v in data.items()}
    trace = type(trace)(*(v[:, :1].expand(129, 11, *v.shape[2:]) if v is not None else None for v in trace))
    expected = old[:, :1].expand(129, 11)
    scores = model.score_decisions(data, trace)
    torch.testing.assert_close(scores, expected, atol=2e-5, rtol=1e-5)
    scores.mean().backward()
    assert model.init_embed_depot.weight.grad.isfinite().all()
