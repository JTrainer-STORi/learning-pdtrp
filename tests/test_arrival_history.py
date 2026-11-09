"""Arrival memory follows request revelation, independently of vehicle capacity."""

import pytest
import torch

from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from problems import PDTRP, PDCTRP, PDTRPTW, PDCTRPTW


def capacity_example():
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                            [0.0, 2.0], [2.0, 2.0]]])
    return {
        "all_nodes": coords,
        "distance_matrix": torch.cdist(coords, coords),
        "graph": (~torch.eye(5, dtype=torch.bool))[None],
        "speed": torch.tensor([[1.0]]),
        "time_horizon": torch.tensor([[480.0]]),
        "latest_end": torch.tensor([[120.0]]),
        "arrival_times": torch.tensor([[0.0, 0.0, 1.5, 0.0, 2.5]]),
        "service_times": torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]),
        "window_starts": torch.zeros(1, 5),
        "window_ends": torch.full((1, 5), 20.0),
        "demand": torch.tensor([[0.0, 0.7, 0.6, 0.6, 0.2]]),
        "vehicle_capacity": torch.tensor([[1.0]]),
        "gamma": torch.tensor([1.0]),
        "theta": torch.tensor([1.0]),
    }


def make_model(problem, scale_times=False):
    model = DynamicAttentionModel(
        problem, embedding_dim=16, encoder_class=GNNEncoder, n_encode_layers=1,
        n_heads=4, edge_features="distance", use_arrival_lstm=True,
        scale_times=scale_times, knn_strat="fixed", neighbors=-1, tanh_clipping=0,
    )
    model.set_decode_type("greedy")
    return model


def record_arrivals(monkeypatch, model):
    events = []
    original_update = model.update_lstm_with_arrivals_packed

    def record(sequences):
        events.append([sequence[:, -1].detach().tolist() for sequence in sequences])
        original_update(sequences)

    monkeypatch.setattr(model, "update_lstm_with_arrivals_packed", record)
    return events


def force_actions(monkeypatch, model, actions):
    actions = iter(actions)

    def select(probs, mask):
        selected = torch.tensor(next(actions), device=probs.device).reshape(-1)
        assert not mask.gather(1, selected[:, None]).any()
        return selected

    monkeypatch.setattr(model, "_select_node", select)


@pytest.mark.parametrize("problem", [PDCTRP, PDCTRPTW])
@pytest.mark.parametrize("mode", ["masked", "recursive"])
@pytest.mark.parametrize("scale_times", [False, True])
def test_arrivals_include_capacity_blocked_requests_and_exclude_reloaded_customers(
    monkeypatch, problem, mode, scale_times
):
    inputs = capacity_example()
    state = problem.make_state(inputs).update(torch.tensor([1]))
    # Customer 2 calls during service but cannot fit in the remaining capacity.
    assert not state.not_arrived_[0, 0, 2]
    assert state.get_mask()[0, 0, 2]
    # Reloading reveals customer 4 and makes already-known customers feasible.
    reloaded = state.update(torch.tensor([0]))
    assert not reloaded.get_mask()[0, 0, 2:4].any()
    assert not reloaded.not_arrived_[0, 0, 4]

    model = make_model(problem, scale_times)
    model.train(mode == "masked")
    events = record_arrivals(monkeypatch, model)
    force_actions(monkeypatch, model, [1, 0, 2, 0, 3, 4])
    cost, likelihood, _ = model(inputs, mode=mode)

    scale = 600.0 if scale_times else 1.0
    assert len(events) == (6 if mode == "masked" else 2)
    assert events[0][0] == pytest.approx([1.5 / scale])
    assert events[1][0] == pytest.approx([2.5 / scale])
    assert all(event == [[]] for event in events[2:])
    if mode == "masked":
        (cost.detach() * likelihood).mean().backward()
        assert torch.isfinite(model.lstm.weight_ih_l0.grad).all()
        assert model.lstm.weight_ih_l0.grad.abs().sum() > 0


@pytest.mark.parametrize("problem", [PDCTRP, PDCTRPTW])
def test_each_pomo_rollout_receives_its_own_arrival_history(monkeypatch, problem):
    inputs = {
        key: value.repeat_interleave(2, dim=0)
        for key, value in capacity_example().items()
    }
    # The same call falls in different transitions along the two POMO routes.
    inputs["arrival_times"][:, 4] = 3.5
    model = make_model(problem)
    events = record_arrivals(monkeypatch, model)
    # POMO chooses customers 1 and 3 initially; subsequent routes reload as needed.
    force_actions(monkeypatch, model, [[0, 0], [2, 1], [0, 0], [3, 2], [4, 4]])
    cost, likelihood, _ = model(inputs, pomo_batch_size=2)

    assert events[0] == [[1.5], [1.5]]
    assert events[1] == [[], [3.5]]
    assert events[2] == [[3.5], []]
    assert all(event == [[], []] for event in events[3:])
    (cost.detach() * likelihood).mean().backward()
    assert torch.isfinite(model.lstm.weight_ih_l0.grad).all()


@pytest.mark.parametrize("problem", [PDTRP, PDCTRP, PDTRPTW, PDCTRPTW])
@pytest.mark.parametrize("mode,remove_visited", [
    ("masked", False), ("recursive", False), ("recursive", True),
])
@pytest.mark.parametrize("scale_times", [False, True])
def test_arrival_history_is_chronological_when_customer_indices_are_not(
    monkeypatch, problem, mode, remove_visited, scale_times
):
    inputs = capacity_example()
    inputs["demand"][:, 1:] = 0.1
    inputs["service_times"][:, 1] = 3.0
    # Both calls happen during the first visit, in reverse node-index order.
    inputs["arrival_times"][:, 2] = 2.5
    inputs["arrival_times"][:, 4] = 1.5
    model = make_model(problem, scale_times)
    model.recursively_remove_visited_nodes = remove_visited
    model.train(mode == "masked")
    events = record_arrivals(monkeypatch, model)
    monkeypatch.setattr(model, "_select_node", lambda probs, mask: (~mask).long().argmax(dim=1))

    cost, likelihood, _ = model(inputs, mode=mode)

    scale = 600.0 if scale_times else 1.0
    assert events[0][0] == pytest.approx([1.5 / scale, 2.5 / scale])
    assert all(event == [[]] for event in events[1:])
    assert torch.isfinite(cost).all()
    if mode == "masked":
        (cost.detach() * likelihood).mean().backward()
        assert torch.isfinite(model.lstm.weight_ih_l0.grad).all()
