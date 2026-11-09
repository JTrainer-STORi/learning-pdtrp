"""Coordinate-only ORTEC benchmarks share synthetic dynamics and travel units."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import evaluate_clairvoyant_ortools as oracle
import evaluate_msa_ortools as msa
import evaluate_online_ortools as online
import generate_data
import comparison_jobs as jobs
from problems.ortec_locations import load_ortec_locations
from utils import load_problem
from utils.data_utils import dataset_to_input


PROBLEMS = ("pdtrp", "pdctrp", "pdtrptw", "pdctrptw")


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.vrp"
    size = 13
    coords = np.column_stack((20 + 10 * np.arange(size), -5 + 3 * (np.arange(size) % 4)))
    lines = ["NAME : fixture", "TYPE : VRPTW", f"DIMENSION : {size}", "CAPACITY : 1000",
             "EDGE_WEIGHT_TYPE : EXPLICIT", "EDGE_WEIGHT_FORMAT : FULL_MATRIX", "EDGE_WEIGHT_SECTION"]
    # Asymmetric road costs and implausible source services/windows must be ignored.
    lines += [" ".join(str(0 if i == j else i + 2 * j + 100) for j in range(size))
              for i in range(size)]
    lines += ["NODE_COORD_SECTION", *[f"{i+1} {x} {y}" for i, (x, y) in enumerate(coords)],
              "DEMAND_SECTION", *[f"{i+1} {0 if i == 0 else 999}" for i in range(size)],
              "DEPOT_SECTION", "1", "-1", "SERVICE_TIME_SECTION",
              *[f"{i+1} {0 if i == 0 else 36000}" for i in range(size)],
              "TIME_WINDOW_SECTION", *[f"{i+1} 0 1" for i in range(size)], "EOF"]
    path.write_text("\n".join(lines) + "\n")
    return path


def generate(tmp_path, source, problem="pdtrp", dod=0.5, count=2, name="data.txt", horizon=10):
    args = generate_data.parse_args([
        "--problem", problem, "--ortec-locations", str(source),
        "--num-customers", "6", "--num-samples", str(count), "--dod", str(dod),
        "--time-horizon", str(horizon), "--seed", "20",
        "--min-trips-required", "1", "--max-trips-required", "3",
        "--output", str(tmp_path / name),
    ])
    result = generate_data.generate(args)
    metadata = json.loads(Path(result["metadata"]).read_text())
    dataset = load_problem(problem).make_dataset(filename=result["dataset"], **metadata["dataset_options"])
    return result, metadata, dataset


def test_source_map_is_normalized_once_with_one_scale_factor(source):
    pool = load_ortec_locations(source)
    coords = pool["coords"]

    assert coords.min() >= 0 and coords.max() <= 1
    assert coords[0, 0] == 0  # The source depot is retained, not moved to the centre.
    np.testing.assert_allclose(np.diff(coords[:, 0]), np.full(12, 1/12))
    np.testing.assert_allclose(coords[1, 1] - coords[0, 1], 3/120)
    larger = load_ortec_locations(source, 2)
    np.testing.assert_allclose(larger["coords"], 2 * coords)
    np.testing.assert_allclose(larger["distance"], 2 * pool["distance"])


@pytest.mark.parametrize("problem", PROBLEMS)
def test_location_benchmark_round_trip_uses_synthetic_features(tmp_path, source, problem):
    _, metadata, dataset = generate(tmp_path, source, problem, horizon=20)
    pool = load_ortec_locations(source)

    assert metadata["distribution"] == "ortec-locations-uniform"
    assert metadata["distance_model"] == "euclidean"
    assert metadata["location_source"]["sha256"] == pool["sha256"]
    assert metadata["coordinate_rounding_decimals"] is None
    assert metadata["dataset_options"].get("use_ortec") is None
    for index, sample in enumerate(dataset):
        ids = metadata["source_ids"][index]
        assert ids[0] == 0 and len(set(ids)) == 7
        np.testing.assert_allclose(sample["all_nodes"], pool["coords"][ids], atol=1e-7)
        torch.testing.assert_close(sample["distance_matrix"], torch.cdist(sample["all_nodes"], sample["all_nodes"]))
        assert sample["time_horizon"].item() == 1200
        calls = sample["arrival_times"]
        assert int((calls > 0).sum()) == 3
        assert (calls[calls > 0] >= 1).all() and (calls <= 1200).all()
        assert sample["service_times"][1:].max() < 100  # Native services were 600 minutes.
        if "demand" in sample:
            assert (sample["demand"][1:] > 0).all() and (sample["demand"] <= 1).all()
        if "window_starts" in sample:
            dynamic = calls > 0
            expected = calls + sample["distance_matrix"][0] / sample["speed"]
            torch.testing.assert_close(sample["window_starts"][dynamic], expected[dynamic], atol=0.0011, rtol=0)
            assert sample["window_ends"][1:].min() > 1


@pytest.mark.parametrize("problem", PROBLEMS)
def test_customer_ids_services_demands_and_calls_are_paired_across_dod(tmp_path, source, problem):
    previous = None
    reference = None
    for dod in (0, 0.5, 1):
        _, metadata, dataset = generate(tmp_path, source, problem, dod, count=3, name=f"dod-{dod}.txt")
        if reference is None:
            reference = (metadata, dataset)
        assert metadata["source_ids"] == reference[0]["source_ids"]
        for index, sample in enumerate(dataset):
            for key in ("all_nodes", "service_times", "distance_matrix", "demand"):
                if key in sample:
                    torch.testing.assert_close(sample[key], reference[1][index][key], atol=0, rtol=0)
            if previous is not None:
                old_calls = previous[index]["arrival_times"]
                retained = old_calls > 0
                torch.testing.assert_close(sample["arrival_times"][retained], old_calls[retained], atol=0, rtol=0)
        previous = dataset


def test_source_sampling_prefixes_are_reproducible_and_population_is_checked(tmp_path, source):
    first, first_metadata, _ = generate(tmp_path, source)
    longer, longer_metadata, _ = generate(tmp_path, source, count=3, name="longer.txt")
    assert Path(first["dataset"]).read_text().splitlines() == Path(longer["dataset"]).read_text().splitlines()[:2]
    assert first_metadata["source_ids"] == longer_metadata["source_ids"][:2]
    args = generate_data.parse_args([
        "--ortec-locations", str(source), "--num-customers", "13", "--num-samples", "1",
        "--output", str(tmp_path / "too-large.txt"),
    ])
    with pytest.raises(ValueError, match="only 12 customers"):
        generate_data.generate(args)
    assert not args.output.exists()


def test_common_customers_keep_coordinates_when_instance_size_changes(tmp_path, source):
    _, small_metadata, small = generate(tmp_path, source, count=1)
    args = generate_data.parse_args([
        "--ortec-locations", str(source), "--num-customers", "10", "--num-samples", "1",
        "--time-horizon", "10", "--seed", "20", "--output", str(tmp_path / "larger.txt"),
    ])
    result = generate_data.generate(args)
    metadata = json.loads(Path(result["metadata"]).read_text())
    larger = load_problem("pdtrp").make_dataset(filename=result["dataset"], **metadata["dataset_options"])
    assert metadata["source_ids"][0][:7] == small_metadata["source_ids"][0]
    torch.testing.assert_close(larger[0]["all_nodes"][:7], small[0]["all_nodes"], rtol=0, atol=0)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_native_operational_attributes_do_not_change_coordinate_only_instances(tmp_path, source, problem):
    first, _, _ = generate(tmp_path, source, problem, count=1)
    changed = tmp_path / "other-source.vrp"
    changed.write_text(source.read_text().replace(" 999\n", " 1\n")
                       .replace(" 36000\n", " 180\n").replace(" 0 1\n", " 3600 7200\n"))
    second, _, _ = generate(tmp_path, changed, problem, count=1, name="other-data.txt")
    assert Path(first["dataset"]).read_bytes() == Path(second["dataset"]).read_bytes()


@pytest.mark.parametrize("problem_name", PROBLEMS)
def test_msa_coordinate_scenarios_use_requested_horizon_and_source_pool(tmp_path, source, problem_name):
    _, metadata, dataset = generate(tmp_path, source, problem_name, horizon=20)
    pool = load_ortec_locations(source)
    problem = load_problem(problem_name)
    item = dataset[0]
    actual = dataset_to_input(item, problem_name, "cpu")
    assert msa.map_ortec_nodes(item, pool) == metadata["source_ids"][0]

    np.random.seed(123)
    sampled = msa.sample_ortec_location_scenario(pool, problem, actual, 7, 1.0, 1, 3)

    assert sampled["time_horizon"].item() == 1200
    assert sampled["arrival_times"][1:].min() >= 1
    assert 480 < sampled["arrival_times"].max() <= 1200
    np.testing.assert_allclose(sampled["all_nodes"], pool["coords"][sampled["source_ids"]], atol=1e-7)
    state = problem.make_state(actual)
    scenario = msa.make_scenario(problem, problem_name, actual, state, 7, 0.5, 1, 3,
                                 ortec=pool, actual_source_ids=metadata["source_ids"][0])
    np.testing.assert_allclose(scenario["distance"], scenario["distance"].T, atol=1e-7)
    assert scenario["first_nodes"]


@pytest.mark.parametrize("module", [online, oracle])
def test_direct_ortools_evaluation_needs_no_legacy_ortec_flag(tmp_path, source, module):
    result, _, _ = generate(tmp_path, source, count=1)
    output = tmp_path / module.__name__
    module.main([
        "--problem", "pdtrp", "--dataset", result["dataset"], "--num-samples", "1",
        "--time-horizon", "10", "--solve-seconds", "0.01", "--output", str(output),
    ])
    summary = json.loads((output / "solve-0.01s" / "summary.json").read_text())
    assert summary["samples"] == 1


def test_parallel_msa_cli_evaluates_coordinate_only_files(tmp_path, source):
    result, _, _ = generate(tmp_path, source, count=1)
    output = tmp_path / "msa.json"
    msa.main([
        "--problem", "pdtrp", "--dataset", result["dataset"], "--num-samples", "1",
        "--time-horizon", "10", "--ortec-locations", str(source), "--dod", "0.5",
        "--min-trips", "1", "--max-trips", "3", "--scenarios", "2", "--workers", "2",
        "--solve-seconds", "0.01", "--output", str(output),
    ])
    summary = json.loads((tmp_path / "msa.solve-0.01s.json").read_text())
    assert summary["distribution"] == "ortec-locations-uniform"
    assert summary["ortec_source"] is None
    assert summary["ortec_location_source"]["sha256"] == load_ortec_locations(source)["sha256"]
    assert summary["workers"] == 2 and len(summary["samples"]) == 1


def test_msa_rejects_mixing_coordinate_and_road_modes():
    with pytest.raises(SystemExit):
        msa.parse_args([
            "--problem", "pdtrp", "--dataset", "data.txt", "--output", "out.json", "--dod", "0.5",
            "--ortec-locations", "source.vrp", "--use-ortec", "source.vrp",
        ])


def test_msa_mapping_checks_coordinates_as_well_as_distances(tmp_path, source):
    _, _, dataset = generate(tmp_path, source, count=1)
    pool = load_ortec_locations(source)
    # Translating the pool leaves every distance unchanged but gives the wrong
    # coordinate features for a future sampled customer.
    pool["coords"] = pool["coords"] + 0.01
    with pytest.raises(ValueError, match="coordinates do not match"):
        msa.map_ortec_nodes(dataset[0], pool)


def test_msa_mapping_tolerance_tracks_the_requested_coordinate_scale(tmp_path, source):
    args = generate_data.parse_args([
        "--ortec-locations", str(source), "--num-customers", "6", "--num-samples", "1",
        "--dimension-scale", "100", "--seed", "20", "--output", str(tmp_path / "large-scale.txt"),
    ])
    result = generate_data.generate(args)
    metadata = json.loads(Path(result["metadata"]).read_text())
    dataset = load_problem("pdtrp").make_dataset(filename=result["dataset"], **metadata["dataset_options"])
    assert msa.map_ortec_nodes(dataset[0], load_ortec_locations(source, 100)) == metadata["source_ids"][0]


def test_comparison_plan_freezes_location_source_and_only_passes_it_to_msa(tmp_path, source):
    generated, _, _ = generate(tmp_path, source, count=1)
    cases = tmp_path / "cases.json"
    case = {"problem": "pdtrp", "dataset": generated["dataset"],
            "ortec_locations": source.name, "dod": 0.5, "time_horizon": 10}
    cases.write_text(json.dumps([case]))
    plan_path = tmp_path / "plan" / "plan.json"
    jobs.main([
        "plan", "--cases", str(cases), "--output", str(plan_path.parent),
        "--num-samples", "1", "--seeds", "0", "--scenarios", "1", "--msa-workers", "1",
    ])
    plan = json.loads(plan_path.read_text())
    for task in plan["tasks"]:
        assert task["ortec"] is None
        assert task["ortec_locations"]["path"] == str(source)
        assert task["ortec_locations"]["sha256"]
        command = jobs.evaluation_command(plan, task, tmp_path / "result")
        assert "--use-ortec" not in command
        assert ("--ortec-locations" in command) == (task["method"] == "msa")

    # A source changed after planning must be caught even when the text data is unchanged.
    source.write_text(source.read_text() + "\n")
    task = plan["tasks"][0]
    array = next(array for array in plan["arrays"] if task["id"] in array["task_ids"])
    with pytest.raises(ValueError, match="changed"):
        jobs.run_task(plan_path, array["name"], array["task_ids"].index(task["id"]))
