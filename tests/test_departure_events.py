"""Departure commits travel/service; calls during a prior wait allow replanning."""

from types import SimpleNamespace

import pytest
import torch

import evaluate_msa_ortools as msa
import evaluate_online_ortools as online
from evaluate_model_free import first_come_first_served, nearest_neighbour
from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from problems import PDCTRPTW, PDTRPTW


TW_PROBLEMS = (PDTRPTW, PDCTRPTW)


def example():
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.1, 0.0]]])
    return {
        "all_nodes": coords,
        "distance_matrix": torch.cdist(coords, coords),
        "graph": (~torch.eye(3, dtype=torch.bool))[None],
        "speed": torch.tensor([1.0]),
        "time_horizon": torch.tensor([480.0]),
        "latest_end": torch.tensor([120.0]),
        "arrival_times": torch.tensor([[0.0, 0.0, 10.0]]),
        "service_times": torch.zeros(1, 3),
        "window_starts": torch.tensor([[0.0, 100.0, 10.0]]),
        "window_ends": torch.tensor([[0.0, 200.0, 12.0]]),
        "demand": torch.tensor([[0.0, 0.1, 0.1]]),
        "vehicle_capacity": torch.tensor([[1.0]]),
        "gamma": torch.tensor([1.0]),
        "theta": torch.tensor([1.0]),
    }


@pytest.mark.parametrize("problem", TW_PROBLEMS)
def test_call_before_departure_keeps_vehicle_at_origin_and_allows_new_choice(problem):
    actual = example()
    initial = problem.make_state(actual)
    if problem is PDCTRPTW:
        initial = initial._replace(used_cap=torch.tensor([[0.4]]))
    waiting = initial.update(torch.tensor([1]))

    assert waiting.cur_times.item() == 10
    assert waiting.prev_a.item() == 0
    torch.testing.assert_close(waiting.cur_coords, initial.cur_coords)
    assert waiting.lengths.item() == 0
    assert not waiting.last_action_executed.item()
    assert not waiting.visited_[0, 0, 1]
    assert not waiting.get_mask()[0, 0, 2]
    assert waiting.arrival_occured.item()
    assert waiting.routes == [[0]]
    assert waiting.visit_times == [[0.0]]
    assert initial.visit_times == [[0.0]]
    if problem is PDCTRPTW:
        assert waiting.used_cap.item() == pytest.approx(0.4)

    served_b = waiting.update(torch.tensor([2]))
    assert served_b.last_action_executed.item()
    assert served_b.cur_times.item() == pytest.approx(10.1)
    if problem is PDCTRPTW:
        assert served_b.used_cap.item() == pytest.approx(0.5)
    finished = served_b.update(torch.tensor([1]))
    assert finished.all_finished()
    tour = finished.construct_solutions(None)
    assert tour.tolist() == [[2, 1]]
    times = torch.tensor(finished.get_visit_times())
    torch.testing.assert_close(times, torch.tensor([[0.0, 10.1, 100.0, 101.0]]))
    cost, info = problem.get_costs(actual, torch.tensor([[0, 2, 1]]), times,
                                   actual["gamma"], actual["theta"])
    assert cost.item() == pytest.approx(2.0)
    assert info["lateness_penalty"].item() == 0


@pytest.mark.parametrize("problem", TW_PROBLEMS)
@pytest.mark.parametrize("call,interrupted", [(8.0, True), (9.0, True), (9.5, False), (12.0, False)])
def test_departure_ties_and_calls_during_travel_or_service(problem, call, interrupted):
    actual = example()
    actual["window_starts"][0, 1] = 10
    actual["service_times"][0, 1] = 5
    actual["arrival_times"][0, 2] = call
    state = problem.make_state(actual).update(torch.tensor([1]))

    assert state.last_action_executed.item() is not interrupted
    assert state.cur_times.item() == (call if interrupted else 15)
    assert state.prev_a.item() == (0 if interrupted else 1)
    assert not state.not_arrived_[0, 0, 2]
    assert state.routes == ([[0]] if interrupted else [[0, 1]])


@pytest.mark.parametrize("problem", TW_PROBLEMS)
def test_revealed_call_does_not_interrupt_same_wait_twice(problem):
    actual = example()
    actual = {key: value.clone() for key, value in actual.items()}
    actual["arrival_times"][0, 1] = 0
    state = problem.make_state(actual).update(torch.tensor([1]))
    assert state.cur_times.item() == 10
    # The already revealed call cannot interrupt the same wait again.
    state = state.update(torch.tensor([1]))
    assert state.cur_times.item() == 100
    assert state.routes == [[0, 1]]
    assert not state.arrival_occured.item()


@pytest.mark.parametrize("problem", TW_PROBLEMS)
@pytest.mark.parametrize("calls", [(10.0, 20.0), (10.0, 10.0)])
def test_successive_and_simultaneous_calls_during_same_wait(problem, calls):
    actual = example()
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.1, 0.0], [0.2, 0.0]]])
    actual.update(
        all_nodes=coords, distance_matrix=torch.cdist(coords, coords),
        graph=(~torch.eye(4, dtype=torch.bool))[None],
        arrival_times=torch.tensor([[0.0, 0.0, *calls]]), service_times=torch.zeros(1, 4),
        window_starts=torch.tensor([[0.0, 100.0, 0.0, 0.0]]),
        window_ends=torch.full((1, 4), 200.0), demand=torch.tensor([[0.0, 0.1, 0.1, 0.1]]),
    )
    state = problem.make_state(actual)
    for call in sorted(set(calls)):
        state = state.update(torch.tensor([1]))
        assert state.cur_times.item() == call
        assert not state.last_action_executed.item()
        assert state.routes == [[0]]
    assert not state.not_arrived_.any()
    state = state.update(torch.tensor([1]))
    assert state.cur_times.item() == 100
    assert state.routes == [[0, 1]]


@pytest.mark.parametrize("problem", TW_PROBLEMS)
def test_initial_empty_queue_advances_to_first_call_without_serving_depot(problem):
    actual = example()
    actual["arrival_times"] = torch.tensor([[0.0, 10.0, 20.0]])
    state = problem.make_state(actual)
    assert state.cur_times.item() == 10
    assert state.prev_a.item() == 0
    assert state.routes == [[0]]
    assert not state.get_mask()[0, 0, 1]
    assert state.get_mask()[0, 0, 2]


@pytest.mark.parametrize("problem", TW_PROBLEMS)
def test_model_free_tours_exclude_canceled_choices(problem):
    actual = example()
    tour, times = nearest_neighbour(problem, actual)
    assert tour.tolist() == [[0, 2, 1]]
    assert times[0, 1].item() == pytest.approx(10.1)
    fcfs_tour, _ = first_come_first_served(problem, actual)
    assert fcfs_tour.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize("problem", TW_PROBLEMS)
@pytest.mark.parametrize("method", ["online", "msa"])
def test_online_baselines_score_actual_wait_history(monkeypatch, problem, method):
    actual = example()
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.1, 0.0], [0.2, 0.0]]])
    actual.update(
        all_nodes=coords, distance_matrix=torch.cdist(coords, coords),
        graph=(~torch.eye(4, dtype=torch.bool))[None],
        arrival_times=torch.tensor([[0.0, 0.0, 0.0, 10.0]]),
        service_times=torch.zeros(1, 4),
        window_starts=torch.tensor([[0.0, 100.0, 0.0, 10.0]]),
        window_ends=torch.tensor([[0.0, 200.0, 5.0, 200.0]]),
        demand=torch.tensor([[0.0, 0.1, 0.1, 0.1]]),
    )

    def chosen_node(known):
        # First propose A and wait; the new call changes the policy's choice
        # to already-known B. B's lateness must include the canceled wait.
        if 3 not in known and 2 in known:
            return 1
        return 2 if 2 in known else (3 if 3 in known else 1)

    def solve(scenario, *args, **kwargs):
        node = chosen_node(scenario["mapping"])
        return {"route": [scenario["mapping"].index(node)], "elapsed_seconds": 0.01,
                "termination_reason": "no_improvement", "improvement_count": 0}

    args = SimpleNamespace(
        problem=problem.NAME, max_trips=9, min_trips=1,
        online_time_ms=10, online_no_improvement_ms=5,
        solver_time_ms=10, solver_no_improvement_ms=5,
        scenarios=1, workers=1, dod=0.5,
    )
    if method == "online":
        monkeypatch.setattr(online, "solve_scenario", solve)
        objective, _, tour = online.evaluate_online(args, problem, actual)
        assert tour == [0, 2, 3, 1]
    else:
        def scenario(problem, name, actual, state, *args):
            known = torch.nonzero((~state.visited_[0, 0]) & (~state.not_arrived_[0, 0])).flatten().tolist()
            chosen = chosen_node(known)
            return {"mapping": [None, chosen, *[node for node in known if node != chosen]]}

        monkeypatch.setattr(msa, "dataset_to_input", lambda *args: actual)
        monkeypatch.setattr(msa, "make_scenario", scenario)
        monkeypatch.setattr(msa, "solve_prepared_scenario", lambda job: solve(job[0]))
        objective, _ = msa.evaluate_instance(args, problem, {"all_nodes": coords[0]})
    assert objective == pytest.approx(7.1)


@pytest.mark.parametrize("problem", TW_PROBLEMS)
@pytest.mark.parametrize("mode", ["masked", "recursive", "recursive_plus_removal"])
@pytest.mark.parametrize("use_lstm", [False, True])
def test_neural_rollout_reconsiders_then_returns_only_executed_visits(monkeypatch, problem, mode, use_lstm):
    model = DynamicAttentionModel(problem, embedding_dim=32, encoder_class=GNNEncoder,
                                  n_encode_layers=1, n_heads=4, use_arrival_lstm=use_lstm).eval()
    choices = iter([1, 2, 1])
    monkeypatch.setattr(model, "_select_node", lambda probabilities, mask: torch.tensor([next(choices)]))
    cost, _, _, tour, times = model(example(), mode=mode, return_pi=True, return_times=True)
    assert tour.tolist() == [[0, 2, 1]]
    assert cost.item() == pytest.approx(2.0)
    torch.testing.assert_close(times, torch.tensor([[0.0, 10.1, 100.0, 101.0]]))


@pytest.mark.parametrize("problem", TW_PROBLEMS)
def test_batched_training_handles_different_events_and_finished_padding(monkeypatch, problem):
    actual = {key: value.repeat((2,) + (1,) * (value.ndim - 1)) for key, value in example().items()}
    actual["arrival_times"][1].zero_()
    actual["window_starts"][1].zero_()
    actual["window_ends"][1].fill_(200)
    model = DynamicAttentionModel(problem, embedding_dim=32, encoder_class=GNNEncoder,
                                  n_encode_layers=1, n_heads=4)
    choices = iter([[1, 1], [2, 2], [1, 0]])
    monkeypatch.setattr(model, "_select_node", lambda probabilities, mask: torch.tensor(next(choices)))
    captured = []
    original_likelihood = model._calc_log_likelihood

    def capture_likelihood(log_probabilities, mask):
        captured.append(log_probabilities)
        return original_likelihood(log_probabilities, mask)

    monkeypatch.setattr(model, "_calc_log_likelihood", capture_likelihood)
    cost, likelihood, _, tour, times = model(actual, return_pi=True, return_times=True)
    assert tour.tolist() == [[0, 2, 1], [0, 1, 2]]
    assert times.shape == (2, 4)
    assert torch.isfinite(likelihood).all()
    assert captured[0].shape == (2, 3)
    # Row 1 finished after two real decisions; its third choice is padding.
    assert captured[0][1, 2].item() == 0.0
    torch.testing.assert_close(likelihood[1], captured[0][1, :2].sum())
    padding_gradients = torch.autograd.grad(
        captured[0][1, 2], tuple(model.parameters()), retain_graph=True, allow_unused=True
    )
    assert all(gradient is None or torch.count_nonzero(gradient) == 0
               for gradient in padding_gradients)
    (cost.detach() * likelihood).mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_batched_capacity_reloads_and_finished_padding_keep_times_aligned(monkeypatch):
    actual = {key: value.repeat((2,) + (1,) * (value.ndim - 1)) for key, value in example().items()}
    actual["vehicle_capacity"] = torch.tensor([[0.15], [1.0]])
    actual["arrival_times"][1].zero_()
    actual["window_starts"][1].zero_()
    actual["window_ends"][1].fill_(200)
    model = DynamicAttentionModel(PDCTRPTW, embedding_dim=32, encoder_class=GNNEncoder,
                                  n_encode_layers=1, n_heads=4)
    choices = iter([[1, 1], [2, 2], [0, 0], [1, 0]])
    monkeypatch.setattr(model, "_select_node", lambda probabilities, mask: torch.tensor(next(choices)))
    cost, likelihood, _, tour, times = model(actual, return_pi=True, return_times=True)
    assert tour.tolist() == [[0, 2, 0, 1], [0, 1, 2, 0]]
    torch.testing.assert_close(times, torch.tensor([[0.0, 10.1, 10.2, 100.0, 101.0],
                                                   [0.0, 1.0, 1.9, 2.0, 2.0]]))
    torch.testing.assert_close(cost, torch.tensor([2.2, 2.0]))
    (cost.detach() * likelihood).mean().backward()
