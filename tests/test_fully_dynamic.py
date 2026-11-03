"""Exercise zero-advance-request instances through generation and evaluation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import evaluate_msa_ortools as msa
from evaluate_model_free import first_come_first_served, nearest_neighbour
from evaluate_online_ortools import evaluate_online
from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from utils.data_utils import dataset_to_input


PROBLEMS = (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW)


def generated_dataset(problem, total=5, batch_size=1, **overrides):
    options = dict(
        min_total=total, max_total=total, min_dod=1.0, max_dod=1.0,
        n_subregions=1, batch_size=batch_size, num_samples=batch_size,
        neighbors=2, knn_strat="fixed",
    )
    if problem in (PDCTRP, PDCTRPTW):
        options.update(vehicle_capacity=1.0, min_trips_required_lb=1, min_trips_required_ub=1)
    options.update(overrides)
    return problem.make_dataset(**options)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("total", [2, 5])
def test_generated_instances_have_only_dynamic_customers(problem, total):
    dataset = generated_dataset(problem, total, batch_size=2)
    for item in dataset:
        assert item["all_nodes"].shape == (total, 2)
        assert item["arrival_times"][0] == 0
        assert torch.all(item["arrival_times"][1:] > 0)
        assert torch.all(item["arrival_times"][1:] <= item["time_horizon"])
        if problem in (PDTRPTW, PDCTRPTW):
            assert item["window_starts"].shape == item["window_ends"].shape == (total,)
            assert torch.all(item["window_starts"][1:] >= item["arrival_times"][1:])
            assert torch.all(item["window_ends"][1:] >= item["window_starts"][1:])


@pytest.mark.parametrize("problem", PROBLEMS)
def test_initial_wait_advances_each_empty_batch_row_without_a_depot_visit(problem):
    dataset = generated_dataset(problem, batch_size=2)
    actual = dataset_to_input(next(iter(DataLoader(dataset, batch_size=2))), problem.NAME, "cpu")
    actual["arrival_times"] = torch.tensor([[0.0, 20.0, 10.0, 10.0, 30.0],
                                             [0.0, 0.0, 12.0, 22.0, 32.0]])
    state = problem.make_state(actual)

    torch.testing.assert_close(state.cur_times[:, 0], torch.tensor([10.0, 0.0]))
    assert state.prev_a.eq(0).all()
    assert state.lengths.eq(0).all()
    assert state.i.item() == 0
    assert state.visit_times == [[0.0], [0.0]]
    assert state.visited_[:, :, 0].all() and not state.visited_[:, :, 1:].any()
    assert not state.not_arrived_[0, 0, 2:4].any()
    assert state.not_arrived_[0, 0, 1]
    assert state.not_arrived_[0, 0, 4]
    assert not state.get_mask()[0, 0, 2:4].any()
    if problem in (PDCTRP, PDCTRPTW):
        assert state.used_cap.eq(0).all()


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("heuristic", [nearest_neighbour, first_come_first_served])
def test_heuristics_complete_fully_dynamic_instances(problem, heuristic):
    actual = dataset_to_input(generated_dataset(problem)[0], problem.NAME, "cpu")
    tour, times = heuristic(problem, actual)
    assert sorted(tour[0][tour[0] != 0].tolist()) == [1, 2, 3, 4]
    assert times[0, 1] >= actual["arrival_times"][0, tour[0, 1]]


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("mode", ["masked", "recursive", "recursive_plus_removal"])
def test_neural_rollout_completes_fully_dynamic_instances(problem, mode):
    actual = dataset_to_input(generated_dataset(problem)[0], problem.NAME, "cpu")
    model = DynamicAttentionModel(
        problem, embedding_dim=32, encoder_class=GNNEncoder, n_encode_layers=1, n_heads=4,
    ).eval()
    model.set_decode_type("greedy")
    with torch.inference_mode():
        cost, _, _, tour, times = model(actual, mode=mode, return_pi=True, return_times=True)
    assert torch.isfinite(cost).all()
    assert sorted(tour[0][tour[0] != 0].tolist()) == [1, 2, 3, 4]
    assert torch.all(times[0, :tour.size(1)] >= actual["arrival_times"][0, tour[0]])


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("dod", [0.0, 1.0])
@pytest.mark.parametrize("use_lstm", [False, True])
def test_pomo_training_uses_only_revealed_start_nodes(monkeypatch, problem, dod, use_lstm):
    dataset = generated_dataset(problem, min_dod=dod, max_dod=dod)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    actual = dataset_to_input(batch, problem.NAME, "cpu", pomo_batch_size=3)
    state_class = type(problem.make_state(actual))
    original_update = state_class.update
    starts = []

    def checked_update(state, selected):
        masked = state.get_mask().gather(-1, selected[:, None, None])
        assert not masked.any(), "A training rollout selected an unavailable request"
        if state.i.item() == 0:
            starts.extend(selected.tolist())
        return original_update(state, selected)

    monkeypatch.setattr(state_class, "update", checked_update)
    model = DynamicAttentionModel(
        problem, embedding_dim=32, encoder_class=GNNEncoder, n_encode_layers=1, n_heads=4,
        use_arrival_lstm=use_lstm,
    )
    model.set_decode_type("sampling")
    costs, likelihood, _ = model(actual, pomo_batch_size=3)
    assert torch.isfinite(costs).all() and torch.isfinite(likelihood).all()
    assert starts == ([1, 2, 3] if dod == 0 else [1, 1, 1])
    (costs.detach() * likelihood).mean().backward()


@pytest.fixture
def ortec_source():
    coords = np.array([[0.5, 0.5], [0.0, 0.1], [0.2, 0.2], [0.3, 0.4], [0.6, 0.7], [0.9, 0.8]])
    return {
        "coords": coords,
        "duration_matrix": np.linalg.norm(coords[:, None] - coords[None, :], axis=-1) * 1000,
        "service_times": np.array([0, 180, 180, 180, 180, 180]),
        "time_windows": np.array([[0, 0], [0, 1000], [200, 1000], [300, 1000], [400, 1000], [500, 1000]]),
        "demands": np.array([0, 1, 1, 1, 1, 1]),
    }


@pytest.mark.parametrize("problem", PROBLEMS)
def test_ortec_generation_supports_full_dynamism(monkeypatch, problem, ortec_source):
    monkeypatch.setattr("problems.utils.read_vrplib", lambda path: ortec_source)
    dataset = generated_dataset(problem, use_ortec="source.vrp")
    item = dataset[0]
    assert item["all_nodes"].shape == (5, 2)
    assert item["arrival_times"][0] == 0
    assert torch.all(item["arrival_times"][1:] > 0)
    state = problem.make_state(dataset_to_input(item, problem.NAME, "cpu"))
    assert state.cur_times.item() == item["arrival_times"][1:].min().item()


@pytest.mark.parametrize("problem", PROBLEMS)
def test_ortec_msa_scenarios_support_full_dynamism(problem, ortec_source):
    context = {**ortec_source, "distance": ortec_source["duration_matrix"] / 1000}
    sample = msa.sample_ortec_scenario(context, problem.NAME, 5, 1.0, 1, 1)
    assert sample["source_ids"][0] == 0
    assert sample["arrival_times"][0] == 0
    assert torch.all(sample["arrival_times"][1:] > 0)
    assert len(set(sample["source_ids"])) == 5


def test_ortec_full_dynamism_requires_enough_positive_call_times(monkeypatch, ortec_source):
    monkeypatch.setattr("problems.utils.read_vrplib", lambda path: ortec_source)
    with pytest.raises(ValueError, match="positive call times"):
        generated_dataset(PDTRP, total=6, use_ortec="source.vrp")


@pytest.mark.parametrize("problem", PROBLEMS)
def test_online_and_msa_solve_fully_dynamic_instances(problem):
    item = generated_dataset(problem)[0]
    actual = dataset_to_input(item, problem.NAME, "cpu")
    options = SimpleNamespace(
        problem=problem.NAME, max_trips=9, min_trips=1,
        online_time_ms=100, online_no_improvement_ms=5,
        solver_time_ms=100, solver_no_improvement_ms=5,
        scenarios=1, workers=1, dod=1.0,
    )
    objective, _, tour = evaluate_online(options, problem, actual)
    assert np.isfinite(objective)
    assert sorted(node for node in tour if node) == [1, 2, 3, 4]
    objective, _ = msa.evaluate_instance(options, problem, item)
    assert np.isfinite(objective)
