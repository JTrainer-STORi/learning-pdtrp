import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import generate_larsen_data
from problems import PDTRP
from problems.larsen_generation import (
    DYNAMISM_LEVELS, dynamism_level, generate_balanced_instances, sample_instance,
)
from utils.data_utils import dataset_to_input


@pytest.mark.parametrize(
    "advance,dynamic,expected",
    [(20, 0, 0), (0, 19, 100), (19, 1, 5), (10, 10, 50),
     (26, 11, None), (1, 3, 75), (0, 0, None)],
)
def test_dynamism_selection_uses_exact_ratios(advance, dynamic, expected):
    assert dynamism_level(advance, dynamic) == expected


def test_quota_selection_and_per_group_prefixes_are_reproducible():
    first, statistics = generate_balanced_instances(20, instances_per_dod=2, seed=20)
    longer, _ = generate_balanced_instances(20, instances_per_dod=3, seed=20)
    assert statistics["candidates"] == 42 + sum(statistics["rejected"].values())
    assert tuple(first) == DYNAMISM_LEVELS
    assert len({item["customers"] for group in first.values() for item in group}) > 1
    for level in first:
        assert len(first[level]) == 2
        for index, item in enumerate(first[level]):
            assert item["advance_customers"] + item["dynamic_customers"] == item["customers"]
            assert 100 * item["dynamic_customers"] == level * item["customers"]
            assert item["arrival_rate_per_hour"] == (20 - item["advance_customers"]) / 8
            other = longer[level][index]
            assert item["candidate_index"] == other["candidate_index"]
            for key, value in item["sample"].items():
                np.testing.assert_array_equal(value, other["sample"][key])
            calls = item["sample"]["arrival_times"]
            assert np.count_nonzero(calls) == item["dynamic_customers"]
            assert np.all(np.diff(calls) >= 0)
            assert np.all((calls[calls > 0] > 0) & (calls[calls > 0] <= 480))


def test_poisson_calls_can_arrive_before_one_minute():
    class FeatureRng:
        def uniform(self, low, high, size):
            return np.full(size, 0.25)

        def lognormal(self, mean, sigma, size):
            return np.full(size, 3.0)

        def random(self, size):
            assert size == 3
            return np.array([0.5, 0.999, 0.1])

    sample = sample_instance(2, 3, FeatureRng())
    np.testing.assert_allclose(sample["arrival_times"], [0, 0, 0, 0.48, 240, 432])
    assert sample["all_nodes"].shape == (6, 2)
    np.testing.assert_allclose(sample["all_nodes"][0], [0.5, 0.5])


def test_larsen_export_loads_variable_size_rows_and_valid_reference_routes(tmp_path):
    args = generate_larsen_data.parse_args([
        "--scenario", "A", "--instances-per-dod", "2", "--seed", "20",
        "--output-dir", str(tmp_path / "scenario-a"),
    ])
    result = generate_larsen_data.generate(args)
    output = Path(result["output_dir"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert result["samples"] == manifest["num_samples"] == 42
    assert manifest["nominal_expected_customers"] == 20
    assert manifest["reference_is_optimal"] is False
    for name, digest in manifest["dataset_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    dataset = PDTRP.make_dataset(
        filename=str(output / "instances.txt"), num_samples=42, batch_size=1,
        neighbors=-1, knn_strat="fixed", n_subregions=1,
    )
    assert len(dataset) == 42
    assert len({len(sample["all_nodes"]) for sample in dataset}) > 1
    for index, sample in enumerate(dataset):
        actual = dataset_to_input(sample, "pdtrp", "cpu")
        calls = sample["arrival_times"]
        assert 100 * calls.gt(0).sum() == DYNAMISM_LEVELS[index // 2] * (len(calls) - 1)
        tour = sample["tour_nodes"][None]
        cost, _ = PDTRP.get_costs(actual, tour)
        times, _ = PDTRP.get_times(actual, tour)
        assert torch.isfinite(cost).all()
        torch.testing.assert_close(times[0], sample["visit_times"])
    assert (output / "overview.png").stat().st_size > 0
    assert (output / "overview.pdf").stat().st_size > 0
    with pytest.raises(FileExistsError):
        generate_larsen_data.generate(args)


def test_candidate_budget_does_not_silently_return_incomplete_groups(tmp_path):
    args = generate_larsen_data.parse_args([
        "--instances-per-dod", "2", "--max-candidates", "1",
        "--output-dir", str(tmp_path / "incomplete"),
    ])
    with pytest.raises(RuntimeError, match="Candidate limit"):
        generate_larsen_data.generate(args)
    assert not args.output_dir.exists()
