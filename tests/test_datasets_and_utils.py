import numpy as np
import pytest
import torch

from options import get_options
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from problems.utils import generate_scaled_demands, nearest_neighbor_graph, sample_lognorm
from utils import load_problem
from utils.data_utils import BatchedRandomSampler, dataset_to_input


PROBLEMS = [PDTRP, PDCTRP, PDTRPTW, PDCTRPTW]


@pytest.mark.parametrize("problem", PROBLEMS)
def test_problem_loader_uses_canonical_names(problem):
    assert load_problem(problem.NAME) is problem


def test_problem_loader_rejects_unknown_name():
    with pytest.raises(AssertionError, match="unsupported"):
        load_problem("not-a-problem")


def test_problem_loader_supports_legacy_checkpoint_names():
    assert load_problem("pdcvrp") is PDCTRP
    assert load_problem("pdcvrptw") is PDCTRPTW

    from problems.pdcvrp.problem_pdcvrp import PDCVRP, PDCVRPDataset
    from problems.pdcvrp_tw.problem_pdcvrptw import PDCVRPTW, PDCVRPTWDataset
    from problems.pdctrp.problem_pdctrp import PDCTRPDataset
    from problems.pdctrp_tw.problem_pdctrptw import PDCTRPTWDataset

    assert PDCVRP is PDCTRP
    assert PDCVRPDataset is PDCTRPDataset
    assert PDCVRPTW is PDCTRPTW
    assert PDCVRPTWDataset is PDCTRPTWDataset


@pytest.mark.parametrize("problem", PROBLEMS)
def test_generated_dataset_has_consistent_shapes_and_model_input(problem):
    kwargs = {
        "min_total": 5,
        "max_total": 5,
        "min_dod": 0.4,
        "max_dod": 0.4,
        "n_subregions": 1,
        "batch_size": 2,
        "num_samples": 2,
        "neighbors": 2,
        "knn_strat": "fixed",
    }
    if problem in (PDCTRP, PDCTRPTW):
        kwargs.update(
            vehicle_capacity=1.0,
            min_trips_required_lb=2,
            min_trips_required_ub=2,
        )

    dataset = problem.make_dataset(**kwargs)
    item = dataset[0]
    model_input = dataset_to_input(item, problem.NAME, torch.device("cpu"))

    assert len(dataset) == 2
    assert item["all_nodes"].shape == (5, 2)
    assert item["distance_matrix"].shape == (5, 5)
    assert item["graph"].shape == (5, 5)
    assert torch.allclose(item["distance_matrix"], item["distance_matrix"].T)
    assert torch.all(item["distance_matrix"].diag() == 0)
    assert model_input["all_nodes"].shape == (1, 5, 2)
    if problem in (PDCTRP, PDCTRPTW):
        assert model_input["demand"].shape == (1, 5)
    if problem in (PDTRPTW, PDCTRPTW):
        assert model_input["window_starts"].shape == (1, 5)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_problem_generation_honors_dimension_scale(problem):
    kwargs = {
        "min_total": 5,
        "max_total": 5,
        "min_dod": 0.4,
        "max_dod": 0.4,
        "n_subregions": 1,
        "batch_size": 1,
        "num_samples": 1,
        "dimension_scale": 2.0,
    }
    if problem in (PDCTRP, PDCTRPTW):
        kwargs.update(
            vehicle_capacity=1.0,
            min_trips_required_lb=2,
            min_trips_required_ub=2,
        )

    nodes = problem.make_dataset(**kwargs)[0]["all_nodes"]

    torch.testing.assert_close(nodes[0], torch.tensor([1.0, 1.0]))
    assert torch.all((nodes >= 0) & (nodes <= 2.0))


def test_nearest_neighbor_graph_returns_distances_and_no_self_edges():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

    graph, distances = nearest_neighbor_graph(nodes, neighbors=1, knn_strat="fixed")

    assert graph.shape == distances.shape == (3, 3)
    np.testing.assert_allclose(np.diag(distances), 0)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_array_equal(np.diag(graph), np.ones(3))


def test_fixed_neighbor_graph_accepts_cli_float_count():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

    graph, _ = nearest_neighbor_graph(nodes, neighbors=1.0, knn_strat="fixed")

    assert graph.shape == (3, 3)


def test_full_neighbor_graph_preserves_supplied_distance_matrix():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    road_distances = np.array([[0.0, 7.0], [11.0, 0.0]])

    _, distances = nearest_neighbor_graph(
        nodes,
        neighbors=-1,
        knn_strat="fixed",
        distance_matrix=road_distances,
    )

    np.testing.assert_array_equal(distances, road_distances)


def test_sample_lognorm_handles_zero_distribution():
    np.testing.assert_array_equal(sample_lognorm(0, 0, size=4), np.zeros(4))


def test_scaled_demands_stay_within_capacity():
    demands = generate_scaled_demands(
        batch_size=3,
        n_total=8,
        vehicle_capacity=1.0,
        min_trips_required_lb=2,
        min_trips_required_ub=3,
    )

    assert demands.shape == (3, 8)
    assert np.all((demands >= 0.001) & (demands <= 1.0))


def test_batched_random_sampler_preserves_order_inside_batches():
    sampler = BatchedRandomSampler(range(12), batch_size=3)
    sampled = list(sampler)

    assert sorted(sampled) == list(range(12))
    for start in range(0, len(sampled), 3):
        assert sampled[start : start + 3] == list(
            range(sampled[start], sampled[start] + 3)
        )


def test_options_parse_cpu_safe_minimal_configuration():
    opts = get_options(
        [
            "--problem",
            "pdctrp",
            "--no_cuda",
            "--no_wandb",
            "--no_tensorboard",
            "--n_epochs",
            "1",
        ]
    )

    assert opts.problem == "pdctrp"
    assert opts.use_cuda is False
    assert opts.n_epochs == 1
    assert opts.dimension_scale == 1.0
    assert opts.save_dir.startswith("outputs/pdctrp_")


@pytest.mark.parametrize("problem", PROBLEMS)
def test_dataset_rejects_silently_truncated_sample_count(problem):
    with pytest.raises(ValueError, match="divisible"):
        problem.make_dataset(batch_size=4, num_samples=5)
