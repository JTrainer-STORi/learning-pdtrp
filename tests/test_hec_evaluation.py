"""Check independent sample evaluation for restartable comparison jobs."""

import builtins
import csv
import json

import numpy as np
import pytest
import torch

import evaluate_msa_ortools as msa
import evaluate_ortools_baselines as baselines
import evaluate_online_ortools as online
import evaluate_clairvoyant_ortools as clairvoyant
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW


PROBLEMS = (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW)


def dataset_row(problem, marker=1):
    row = (
        f"0 0 {marker} 0 0 1 arrival_times 0 0 0 service_times 0 0 0 "
    )
    if problem in (PDTRPTW, PDCTRPTW):
        row += "window_starts 0 0 0 window_ends 0 1000 1000 "
    if problem in (PDCTRP, PDCTRPTW):
        row += "demands 0 0.4 0.4 "
    return row + "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n"


@pytest.mark.parametrize("problem", PROBLEMS)
def test_dataset_offset_selects_rows_and_truncates_nonempty_tail(tmp_path, problem):
    path = tmp_path / "data.txt"
    path.write_text("".join(dataset_row(problem, marker) for marker in (1, 2, 3)))

    dataset = problem.make_dataset(filename=str(path), offset=1, num_samples=8, batch_size=1)

    assert len(dataset) == 2
    assert dataset[0]["all_nodes"][1, 0].item() == 2
    assert dataset[1]["all_nodes"][1, 0].item() == 3


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize(("offset", "num_samples"), [(-1, 1), (0, 0), (0, -1), (1, 1)])
def test_dataset_rejects_invalid_or_empty_selection(tmp_path, problem, offset, num_samples):
    path = tmp_path / "data.txt"
    path.write_text(dataset_row(problem))

    with pytest.raises(ValueError):
        problem.make_dataset(
            filename=str(path), offset=offset, num_samples=num_samples, batch_size=1
        )


@pytest.mark.parametrize("problem", PROBLEMS)
def test_dataset_file_selection_streams_without_reading_past_slice(tmp_path, monkeypatch, problem):
    path = tmp_path / "data.txt"
    path.touch()
    original_open = builtins.open

    class GuardedRows:
        def __init__(self):
            self.rows = iter(["skipped malformed row\n", dataset_row(problem, 2)])
            self.count = 0
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def __iter__(self):
            return self

        def __next__(self):
            self.count += 1
            if self.count > 2:
                raise AssertionError("Loader read past the requested dataset slice")
            return next(self.rows)

        def readlines(self):
            raise AssertionError("Loader must not read the whole dataset")

    source = GuardedRows()

    def guarded_open(filename, *args, **kwargs):
        if str(filename) == str(path):
            return source
        return original_open(filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    dataset = problem.make_dataset(filename=str(path), offset=1, num_samples=1, batch_size=1)

    assert len(dataset) == 1
    assert dataset[0]["all_nodes"][1, 0].item() == 2
    assert source.count == 2
    assert source.closed is True


def solver_details():
    return {
        "solver_objective": 3.0, "solver_status": 1,
        "termination_reason": "no_improvement", "improvement_count": 0,
        "solve_count": 2, "termination_counts": {"no_improvement": 2},
    }


@pytest.mark.parametrize(("module", "method"), [
    (baselines, "online"), (baselines, "oracle"), (baselines, "both"),
    (online, "online"), (clairvoyant, "oracle"),
])
def test_selected_ortools_method_uses_global_sample_id(tmp_path, monkeypatch, module, method):
    dataset = tmp_path / "data.txt"
    dataset.write_text(dataset_row(PDTRP, 1) + dataset_row(PDTRP, 2))
    output = tmp_path / method
    calls = []

    def evaluate_online(args, problem, actual, *, return_details=False):
        assert method != "oracle", "Oracle-only evaluation called the online solver"
        assert actual["all_nodes"][0, 1, 0].item() == 2
        calls.append("online")
        details = solver_details()
        details["termination_reasons"] = {"no_improvement": 2}
        result = (3.0, 0.01, [0, 1, 2])
        return (*result, details) if return_details else result

    def evaluate_static(args, problem, actual, *, return_details=False):
        assert method != "online", "Online-only evaluation called the oracle solver"
        assert actual["all_nodes"][0, 1, 0].item() == 2
        calls.append("oracle")
        result = (3.0, 0.01, [0, 1, 2])
        return (*result, solver_details()) if return_details else result

    if method != "oracle":
        monkeypatch.setattr(module, "evaluate_online", evaluate_online)
    if method != "online":
        monkeypatch.setattr(module, "evaluate_static", evaluate_static)
    argv = [
        "--problem", "pdtrp", "--dataset", str(dataset),
        "--output", str(output), "--offset", "1", "--num-samples", "1",
    ]
    if module is baselines:
        argv += ["--method", method]

    module.main(argv)

    summary = json.loads((output / "summary.json").read_text())
    with (output / "samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["offset"] == 1
    assert len(rows) == 1
    assert int(rows[0]["sample"]) == 1
    assert calls == (["online", "oracle"] if method == "both" else [method])
    assert ("online_objective" in rows[0]) is (method != "oracle")
    assert ("static_objective" in rows[0]) is (method != "online")
    assert ("gap_percent" in rows[0]) is (method == "both")


def test_msa_chunk_has_same_instance_rng_as_full_evaluation(tmp_path, monkeypatch):
    dataset = tmp_path / "data.txt"
    dataset.write_text(dataset_row(PDTRP, 1) + dataset_row(PDTRP, 2))
    draws = []

    def evaluate(args, problem, item, ortec=None, executor=None, return_details=False):
        draws.append((item["all_nodes"][1, 0].item(), np.random.random(3).tolist(), torch.rand(3).tolist()))
        details = {
            "wall_time_seconds": 0.01, "solver_wall_time_seconds": 0.01,
            "aggregate_solver_seconds": 0.02, "scenario_count": 2, "workers": 1,
            "termination_counts": {"no_improvement": 2}, "improvement_count": 0,
            "solver_no_improvement_ms": args.solver_no_improvement_ms,
            "solver_time_ms": args.solver_time_ms,
        }
        return (3.0, 0.01, details) if return_details else (3.0, 0.01)

    monkeypatch.setattr(msa, "evaluate_instance", evaluate)
    outputs = []
    for offset, count in [(0, 2), (1, 1)]:
        output = tmp_path / f"msa-{offset}.json"
        msa.main([
            "--problem", "pdtrp", "--dataset", str(dataset), "--output", str(output),
            "--dod", "0.5", "--seed", "97", "--offset", str(offset),
            "--num-samples", str(count), "--solver-no-improvement-ms", "5",
        ])
        outputs.append(json.loads(output.read_text()))

    assert draws[1] == draws[2]
    assert outputs[1]["offset"] == 1
    assert outputs[0]["samples"][1]["sample"] == outputs[1]["samples"][0]["sample"] == 1
    expected = msa.instance_seed(97, 1)
    assert outputs[0]["samples"][1]["instance_seed"] == expected
    assert outputs[1]["samples"][0]["instance_seed"] == expected


@pytest.mark.parametrize(("base_seed", "sample"), [(0, 0), (97, 1), (2**40, 50000)])
def test_instance_seed_is_stable_for_global_sample_identity(base_seed, sample):
    expected = int(np.random.SeedSequence([base_seed, sample]).generate_state(1)[0])
    assert msa.instance_seed(base_seed, sample) == expected


@pytest.mark.parametrize("module", [baselines, online, clairvoyant, msa])
def test_cli_rejects_negative_offset(module):
    argv = ["--problem", "pdtrp", "--dataset", "input.txt", "--output", "output", "--offset", "-1"]
    if module is msa:
        argv += ["--dod", "0.5"]
    with pytest.raises(SystemExit):
        module.parse_args(argv)
