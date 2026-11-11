import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import evaluate_msa_ortools
import evaluate_ortools_baselines
import evaluate_online_ortools
import evaluate_clairvoyant_ortools
from utils.ortools_reloads import DEPOT_RETURN_POLICY


def test_evaluation_writes_oracle_stopping_configuration_and_results(tmp_path, monkeypatch):
    dataset = tmp_path / "tiny.txt"
    dataset.write_text(
        "0 0 1 0 0 1 arrival_times 0 0 0 service_times 0 0 0 "
        "window_starts 0 0 0 window_ends 0 0 0 "
        "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n"
    )
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ortools_baselines.py",
        "--problem", "pdtrptw", "--dataset", str(dataset),
        "--output", str(output), "--num-samples", "1",
        "--online-no-improvement-ms", "20", "--static-no-improvement-ms", "20",
    ])

    evaluate_ortools_baselines.main()

    summary = json.loads((output / "summary.json").read_text())
    with (output / "samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["static_time_ms"] is None
    assert summary["online_time_ms"] is None
    assert summary["online_no_improvement_ms"] == 20
    assert summary["online_stopping_rule"] == "no_improvement"
    assert sum(summary["online_termination_reasons"].values()) == 2
    assert summary["static_no_improvement_ms"] == 20
    assert summary["static_stopping_rule"] == "no_improvement"
    assert sum(summary["static_termination_reasons"].values()) == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["static_termination_reason"] in {"no_improvement", "search_completed"}
    assert int(row["static_solver_status"]) in {1, 2, 7}
    assert float(row["static_solver_objective"]) == pytest.approx(
        float(row["static_objective"]), abs=0.002
    )
    assert sorted(json.loads(row["static_tour"])) == [0, 1, 2]
    assert float(row["static_runtime_seconds"]) > 0
    assert int(row["online_solve_count"]) == 2
    assert sum(json.loads(row["online_termination_reasons"]).values()) == 2


@pytest.mark.parametrize(("module", "method", "prefix"), [
    (evaluate_online_ortools, "online", "online"),
    (evaluate_clairvoyant_ortools, "oracle", "static"),
])
def test_separate_ortools_cli_evaluates_only_its_method_at_global_offset(tmp_path, module, method, prefix):
    dataset = tmp_path / "tiny.txt"
    dataset.write_text(
        "this row must be skipped\n"
        "0 0 1 0 0 1 arrival_times 0 0 0 service_times 0 0 0 "
        "window_starts 0 0 0 window_ends 0 0 0 "
        "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n"
    )
    output = tmp_path / "results"

    subprocess.run([
        sys.executable, str(Path(module.__file__).resolve()),
        "--problem", "pdtrptw", "--dataset", str(dataset),
        "--output", str(output), "--offset", "1", "--num-samples", "1",
        f"--{prefix}-no-improvement-ms", "20",
    ], check=True, capture_output=True, text=True)

    summary = json.loads((output / "summary.json").read_text())
    with (output / "samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["method"] == method
    assert summary["depot_return_policy"] == "not_applicable"
    assert summary["initial_route_policy"] == "best_of_customer_order_and_earliest_deadline"
    assert summary["offset"] == 1
    assert summary[f"{prefix}_time_ms"] is None
    assert summary[f"{prefix}_no_improvement_ms"] == 20
    assert summary[f"{prefix}_stopping_rule"] == "no_improvement"
    assert len(rows) == 1
    row = rows[0]
    assert int(row["sample"]) == 1
    assert float(row[f"{prefix}_objective"]) > 0
    assert float(row[f"{prefix}_runtime_seconds"]) > 0
    assert "gap_percent" not in row
    other = "static" if method == "online" else "online"
    assert not any(key.startswith(other + "_") for key in row)
    assert not any(key.startswith(other + "_") for key in summary)
    if method == "online":
        assert int(row["online_solve_count"]) == 2
        assert sum(json.loads(row["online_termination_reasons"]).values()) == 2
    else:
        assert row["static_termination_reason"] in {"no_improvement", "search_completed"}
        assert sorted(json.loads(row["static_tour"])) == [0, 1, 2]
        assert float(row["static_solver_objective"]) == pytest.approx(
            float(row["static_objective"]), abs=0.002
        )


@pytest.mark.parametrize(("module", "flag", "value"), [
    (evaluate_online_ortools, "--static-no-improvement-ms", "20"),
    (evaluate_online_ortools, "--static-time-ms", "20"),
    (evaluate_online_ortools, "--method", "oracle"),
    (evaluate_clairvoyant_ortools, "--online-no-improvement-ms", "20"),
    (evaluate_clairvoyant_ortools, "--online-time-ms", "20"),
    (evaluate_clairvoyant_ortools, "--method", "online"),
])
def test_separate_ortools_cli_rejects_flags_for_other_methods(module, flag, value):
    with pytest.raises(SystemExit):
        module.parse_args([
            "--problem", "pdtrp", "--dataset", "input.txt", "--output", "output",
            flag, value,
        ])


def test_parallel_msa_cli_records_uncapped_search_and_elapsed_time(tmp_path):
    dataset = tmp_path / "tiny.txt"
    dataset.write_text(
        "0 0 1 0 0 1 arrival_times 0 0 0 service_times 0 0 0 "
        "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n" * 2
    )
    output = tmp_path / "msa.json"

    evaluate_msa_ortools.main([
        "--problem", "pdtrp", "--dataset", str(dataset), "--output", str(output),
        "--dod", "0.5", "--num-samples", "2", "--scenarios", "2",
        "--workers", "2", "--solver-no-improvement-ms", "20",
    ])

    result = json.loads(output.read_text())
    assert result["workers"] == result["workers_requested"] == 2
    assert result["solver_time_ms"] is None
    assert result["solver_no_improvement_ms"] == 20
    assert result["scenario_count"] == 8
    assert sum(result["termination_counts"].values()) == 8
    assert result["wall_time_seconds"] >= result["solver_wall_time_seconds"] > 0
    assert result["aggregate_solver_seconds"] > 0
    assert len(result["samples"]) == 2
    for sample in result["samples"]:
        assert sample["objective"] == pytest.approx(2 + 2**0.5, abs=1e-4)
        assert sample["runtime"] == sample["wall_time_seconds"]
        assert sample["wall_time_seconds"] >= sample["solver_wall_time_seconds"] > 0
        assert sample["scenario_count"] == 4


@pytest.mark.parametrize(("module", "prefix"), [
    (evaluate_clairvoyant_ortools, "static"),
    (evaluate_online_ortools, "online"),
    (evaluate_msa_ortools, "solver"),
])
@pytest.mark.parametrize("problem", ["pdctrp", "pdctrptw", "pdtrptw"])
def test_results_record_depot_return_and_initial_route_policies(tmp_path, module, prefix, problem):
    dataset = tmp_path / "tiny.txt"
    windows = "window_starts 0 0 0 window_ends 0 1000 1000 " if problem.endswith("tw") else ""
    demands = "" if problem == "pdtrptw" else "demands 0 0.4 0.4 "
    dataset.write_text(
        "0 0 1 0 0 1 arrival_times 0 0 0 service_times 0 0 0 "
        f"{windows}{demands}"
        "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n"
    )
    output = tmp_path / ("msa.json" if prefix == "solver" else "results")
    argv = ["--problem", problem, "--dataset", str(dataset), "--output", str(output),
            "--num-samples", "1", f"--{prefix}-time-ms", "10",
            f"--{prefix}-no-improvement-ms", "5"]
    if prefix == "solver":
        argv += ["--dod", "0.5", "--scenarios", "1", "--workers", "1"]
    module.main(argv)

    result = json.loads((output if prefix == "solver" else output / "summary.json").read_text())
    assert result["depot_return_policy"] == (
        "not_applicable" if problem == "pdtrptw" else DEPOT_RETURN_POLICY
    )
    assert result["initial_route_policy"] == {
        "pdctrp": "demand_packing",
        "pdctrptw": "best_of_demand_packing_and_earliest_deadline",
        "pdtrptw": "best_of_customer_order_and_earliest_deadline",
    }[problem]


@pytest.mark.parametrize("total_limit", [None, "1000"])
@pytest.mark.parametrize("method", ["ortools", "online", "oracle", "msa"])
def test_wrapper_does_not_add_a_total_deadline_unless_requested(tmp_path, total_limit, method):
    dataset = tmp_path / "data.txt"
    dataset.touch()
    arguments_file = tmp_path / "arguments.json"
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['PDTRP_ARGUMENTS_FILE'], 'w') as handle:\n"
        "    json.dump(sys.argv[1:], handle)\n"
    )
    python.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = str(tmp_path) + os.pathsep + environment["PATH"]
    environment["PDTRP_ARGUMENTS_FILE"] = str(arguments_file)
    environment["STATIC_NO_IMPROVEMENT_MS"] = "25"
    environment["ONLINE_NO_IMPROVEMENT_MS"] = "25"
    environment["SOLVER_NO_IMPROVEMENT_MS"] = "25"
    environment["MSA_WORKERS"] = "2"
    environment["SCENARIOS"] = "4"
    environment["DOD"] = "0.5"
    for key in ("STATIC_TIME_MS", "ONLINE_TIME_MS", "SOLVER_TIME_MS"):
        environment.pop(key, None)
        if total_limit is not None:
            environment[key] = total_limit
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate-baseline.sh"

    subprocess.run([
        "bash", str(script), method, "pdtrp", str(dataset), str(tmp_path / "output")
    ], env=environment, check=True)

    arguments = json.loads(arguments_file.read_text())
    prefixes = {
        "ortools": ("static", "online"), "online": ("online",),
        "oracle": ("static",), "msa": ("solver",),
    }[method]
    script_name = {
        "ortools": "evaluate_ortools_baselines.py", "online": "evaluate_online_ortools.py",
        "oracle": "evaluate_clairvoyant_ortools.py", "msa": "evaluate_msa_ortools.py",
    }[method]
    script_argument = arguments[1] if arguments[0] == "-u" else arguments[0]
    assert Path(script_argument).name == script_name
    for prefix in prefixes:
        assert arguments[arguments.index(f"--{prefix}-no-improvement-ms") + 1] == "25"
        if total_limit is None:
            assert f"--{prefix}-time-ms" not in arguments
        else:
            assert arguments[arguments.index(f"--{prefix}-time-ms") + 1] == total_limit
    if method == "msa":
        assert arguments[arguments.index("--workers") + 1] == "2"
        assert arguments[arguments.index("--scenarios") + 1] == "4"
    if method in {"online", "oracle"}:
        assert "--method" not in arguments
        other = "static" if method == "online" else "online"
        assert f"--{other}-no-improvement-ms" not in arguments
        assert f"--{other}-time-ms" not in arguments
