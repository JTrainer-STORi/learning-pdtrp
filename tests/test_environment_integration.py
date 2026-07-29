import contextlib
import io

import pytest
import torch

from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from utils.data_utils import dataset_to_input


PROBLEMS = (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW)


def make_small_input(problem):
    kwargs = {
        "min_total": 5,
        "max_total": 5,
        "min_dod": 0.4,
        "max_dod": 0.4,
        "n_subregions": 1,
        "batch_size": 1,
        "num_samples": 1,
        "neighbors": 2,
        "knn_strat": "fixed",
    }
    if problem in (PDCTRP, PDCTRPTW):
        kwargs.update(
            vehicle_capacity=1.0,
            min_trips_required_lb=2,
            min_trips_required_ub=2,
        )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dataset = problem.make_dataset(**kwargs)
    return dataset_to_input(dataset[0], problem.NAME, torch.device("cpu"))


@pytest.mark.parametrize("problem", PROBLEMS)
def test_model_classifies_canonical_problem_names(problem):
    model = DynamicAttentionModel(
        problem,
        embedding_dim=32,
        encoder_class=GNNEncoder,
        n_encode_layers=1,
        n_heads=4,
    )

    assert model.is_pdcvrp is (problem is PDCTRP)
    assert model.is_pdtrptw is (problem is PDTRPTW)
    assert model.is_pdcvrptw is (problem is PDCTRPTW)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("mode", ("masked", "recursive"))
def test_every_problem_completes_model_forward(problem, mode):
    model_input = make_small_input(problem)
    model = DynamicAttentionModel(
        problem,
        embedding_dim=32,
        encoder_class=GNNEncoder,
        n_encode_layers=1,
        n_heads=4,
        edge_features="adjacency",
    )
    model.set_decode_type("greedy")
    model.eval()

    with torch.no_grad():
        cost, log_likelihood, _ = model(model_input, mode=mode)

    assert cost.shape == (1,)
    assert torch.isfinite(cost).all()
    assert log_likelihood is None


@pytest.mark.parametrize("problem", PROBLEMS)
def test_arrival_is_reported_only_once(problem):
    model_input = state_input(problem)
    model_input["arrival_times"] = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    state = problem.make_state(model_input)

    # Reaching static customer 1 at t=1 reveals customer 2.
    state = state.update(torch.tensor([1]))
    assert state.arrival_occured.item() is True
    assert state.not_arrived_[0, 0, 2].item() is False

    # A zero-distance next transition starts exactly at t=1. Customer 2 must
    # not be reported as newly arrived a second time.
    state = state.update(torch.tensor([2]))
    assert state.arrival_occured.item() is False


@pytest.mark.parametrize("problem", PROBLEMS)
def test_final_return_time_uses_supplied_distance_matrix(problem):
    model_input = state_input(problem)
    model_input["arrival_times"].zero_()
    model_input["distance_matrix"] = torch.tensor(
        [
            [
                [0.0, 10.0, 20.0, 30.0],
                [10.0, 0.0, 10.0, 20.0],
                [20.0, 10.0, 0.0, 10.0],
                [99.0, 20.0, 10.0, 0.0],
            ]
        ]
    )
    state = problem.make_state(model_input)

    for customer in (1, 2, 3):
        state = state.update(torch.tensor([customer]))

    # Three ten-unit legs plus the asymmetric 99-unit return.
    assert state.visit_times[0][-1] == pytest.approx(129.0)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_graph_access_and_mask_use_state_graph(problem):
    model_input = state_input(problem)
    state = problem.make_state(model_input)

    torch.testing.assert_close(state.get_graph(), model_input["graph"])
    assert not state.get_graph_mask().any()

    state = state.update(torch.tensor([1]))
    expected_non_neighbors = ~model_input["graph"][:, 1:2].bool()
    torch.testing.assert_close(state.get_graph_mask(), expected_non_neighbors)


def state_input(problem):
    coordinates = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]
    )
    data = {
        "all_nodes": coordinates,
        "service_times": torch.zeros(1, 4),
        "arrival_times": torch.tensor([[0.0, 0.0, 1.0, 100.0]]),
        "speed": torch.tensor([1.0]),
        "time_horizon": torch.tensor([100.0]),
        "distance_matrix": torch.cdist(coordinates, coordinates),
        "graph": torch.tensor(
            [
                [
                    [False, True, False, False],
                    [True, False, True, False],
                    [False, True, False, True],
                    [False, False, True, False],
                ]
            ]
        ),
    }
    if problem in (PDTRPTW, PDCTRPTW):
        data.update(
            latest_end=torch.tensor([20.0]),
            window_starts=torch.zeros(1, 4),
            window_ends=torch.full((1, 4), 200.0),
        )
    if problem in (PDCTRP, PDCTRPTW):
        data.update(
            demand=torch.tensor([[0.0, 0.1, 0.1, 0.1]]),
            vehicle_capacity=torch.tensor([[1.0]]),
        )
    return data
