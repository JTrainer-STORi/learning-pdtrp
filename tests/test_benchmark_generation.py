"""Uniform-uniform N=50 benchmarks survive serialization and remain reproducible."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import generate_data
from evaluate_model_free import nearest_neighbour
from utils import load_problem
from utils.data_utils import dataset_to_input


PROBLEMS = ("pdtrp", "pdctrp", "pdtrptw", "pdctrptw")


def generate(tmp_path, problem, dod=0.5, count=2, seed=20, name="data.txt"):
    args = generate_data.parse_args([
        "--problem", problem, "--num-customers", "50", "--num-samples", str(count),
        "--dod", str(dod), "--seed", str(seed), "--output", str(tmp_path / name),
    ])
    result = generate_data.generate(args)
    return result, json.loads(Path(result["metadata"]).read_text())


@pytest.mark.parametrize("problem_name", PROBLEMS)
@pytest.mark.parametrize("dod,dynamic_count", [(0.2, 10), (0.5, 25), (0.8, 40)])
def test_uniform_benchmark_round_trip_preserves_instances_and_reference_routes(
    tmp_path, problem_name, dod, dynamic_count
):
    result, metadata = generate(tmp_path, problem_name, dod)
    problem = load_problem(problem_name)
    options = metadata["dataset_options"]
    loaded = problem.make_dataset(filename=result["dataset"], **options)
    np.random.seed(20)
    torch.manual_seed(20)
    expected = problem.make_dataset(**options)

    assert len(loaded) == len(expected) == 2
    assert metadata["distribution"] == "uniform-uniform"
    assert metadata["num_nodes_including_depot"] == 51
    assert metadata["num_customers"] == 50
    assert metadata["size_convention"] == "customers_excluding_depot"
    assert metadata["dynamic_customers"] == dynamic_count
    assert metadata["realized_dod"] == dynamic_count / 50
    assert metadata["reference_method"] == "nearest_neighbour"
    assert metadata["reference_is_optimal"] is False
    payload = Path(result["dataset"]).read_bytes()
    assert len(payload.splitlines()) == 2
    assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
    assert len(metadata["source_sha256"]) >= 8

    for source, actual in zip(expected, loaded):
        for key, value in source.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        nodes, calls = actual["all_nodes"], actual["arrival_times"]
        assert nodes.shape == (51, 2)
        torch.testing.assert_close(nodes[0], torch.tensor([0.5, 0.5]))
        assert torch.all((nodes >= 0) & (nodes <= 1))
        assert calls[0] == 0
        assert (calls[1:] > 0).sum() == dynamic_count
        assert torch.all(calls[1:] >= calls[:-1])
        assert torch.all((calls[calls > 0] >= 1) & (calls[calls > 0] <= 480))
        assert actual["service_times"][0] == 0
        assert actual["service_times"].ge(0).all()

        model_input = dataset_to_input(actual, problem_name, "cpu")
        tour, times = nearest_neighbour(problem, model_input)
        torch.testing.assert_close(tour[0], actual["tour_nodes"])
        torch.testing.assert_close(times[0], actual["visit_times"])
        if problem_name in ("pdtrptw", "pdctrptw"):
            cost, _ = problem.get_costs(model_input, tour, times, model_input["gamma"], model_input["theta"])
            dynamic = calls[1:] > 0
            expected_opening = calls[1:] + actual["distance_matrix"][0, 1:] / actual["speed"]
            torch.testing.assert_close(actual["window_starts"][1:][dynamic],
                                       expected_opening[dynamic], rtol=0, atol=0.0011)
            widths = actual["window_ends"][1:] - actual["window_starts"][1:]
            assert widths.ge(60 - 0.0011).all() and widths.le(180 + 0.0011).all()
        else:
            cost, _ = problem.get_costs(model_input, tour)
        assert torch.isfinite(cost).all()


@pytest.mark.parametrize("problem", PROBLEMS)
def test_seed_and_sample_count_produce_stable_benchmark_prefixes(tmp_path, problem):
    first, _ = generate(tmp_path, problem, count=2, name="first.txt")
    again, _ = generate(tmp_path, problem, count=2, name="again.txt")
    longer, _ = generate(tmp_path, problem, count=3, name="longer.txt")
    other, _ = generate(tmp_path, problem, count=2, seed=21, name="other.txt")
    rows = lambda result: Path(result["dataset"]).read_bytes().splitlines()
    assert rows(first) == rows(again) == rows(longer)[:2]
    assert rows(first) != rows(other)


def test_generator_refuses_to_overwrite_existing_benchmarks(tmp_path):
    result, _ = generate(tmp_path, "pdtrp")
    original = Path(result["dataset"]).read_bytes()
    with pytest.raises(FileExistsError):
        generate(tmp_path, "pdtrp")
    assert Path(result["dataset"]).read_bytes() == original


def test_generator_cli_writes_machine_readable_manifest(tmp_path):
    command = [sys.executable, str(Path(generate_data.__file__)), "--problem", "pdtrptw",
               "--num-samples", "1", "--output", str(tmp_path / "data.txt")]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)
    assert result["samples"] == 1
    assert Path(result["dataset"]).is_file()
    assert Path(result["metadata"]).is_file()
