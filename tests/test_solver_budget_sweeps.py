"""Fixed-budget comparisons must isolate outputs, stopping rules and RNGs."""

import csv
import json

import numpy as np
import pytest
import torch

import evaluate_clairvoyant_ortools as clairvoyant
import evaluate_msa_ortools as msa
import evaluate_online_ortools as online
import evaluate_ortools_baselines as paired


MODULES = (online, clairvoyant, paired, msa)


def cli_arguments(module, dataset="input.txt", output="output"):
    arguments = [
        "--problem", "pdtrp", "--dataset", str(dataset), "--output", str(output),
    ]
    return arguments + (["--dod", "0.5"] if module is msa else [])


def dataset_row(marker=1):
    return (
        f"0 0 {marker} 0 0 1 arrival_times 0 0 0 service_times 0 0 0 "
        "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n"
    )


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "0.0001", "0.0015", "1e30"])
def test_budget_cli_rejects_invalid_or_unrepresentable_seconds(module, value):
    with pytest.raises(SystemExit):
        module.parse_args([*cli_arguments(module), "--solve-seconds", value])


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("values", [[], ["1", "1.0"]])
def test_budget_cli_requires_nonempty_unique_budgets(module, values):
    with pytest.raises(SystemExit):
        module.parse_args([*cli_arguments(module), "--solve-seconds", *values])


@pytest.mark.parametrize(("module", "prefix"), [
    (online, "online"), (clairvoyant, "static"), (paired, "online"),
    (paired, "static"), (msa, "solver"),
])
@pytest.mark.parametrize("suffix", ["time-ms", "no-improvement-ms"])
def test_budget_cli_rejects_competing_stopping_options(module, prefix, suffix):
    with pytest.raises(SystemExit):
        module.parse_args([
            *cli_arguments(module), "--solve-seconds", "1", "5",
            f"--{prefix}-{suffix}", "10",
        ])


@pytest.mark.parametrize("module", MODULES)
def test_single_seconds_budget_selects_time_only_stopping(module):
    args = module.parse_args([*cli_arguments(module), "--solve-seconds", "0.125"])

    assert args.solve_seconds == [0.125]
    for prefix in ("online", "static", "solver"):
        if hasattr(args, f"{prefix}_no_improvement_ms"):
            assert getattr(args, f"{prefix}_no_improvement_ms") is None


@pytest.mark.parametrize("module", MODULES)
def test_budget_sweep_writes_each_result_and_comparison_at_selected_offset(tmp_path, module):
    dataset = tmp_path / "tiny.txt"
    dataset.write_text("skip this malformed first row\n" + dataset_row())
    output = tmp_path / ("msa.json" if module is msa else "results")
    arguments = [
        *cli_arguments(module, dataset, output),
        "--num-samples", "1", "--offset", "1", "--speed", "4", "--time-horizon", "10",
        "--solve-seconds", "0.01", "0.02",
    ]
    if module is msa:
        arguments += ["--scenarios", "2", "--workers", "2", "--seed", "97"]

    module.main(arguments)

    manifest_path = tmp_path / "msa.sweep.json" if module is msa else output / "sweep.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["solve_seconds"] == [0.01, 0.02]
    assert len(manifest["runs"]) == 2
    for milliseconds in (10, 20):
        label = f"solve-{milliseconds / 1000:g}s"
        if module is msa:
            result = json.loads((tmp_path / f"msa.{label}.json").read_text())
            assert result["solver_time_ms"] == milliseconds
            assert result["solver_no_improvement_ms"] is None
            assert result["solver_stopping_rule"] == "time_limit"
            assert result["workers"] == 2
            rows = result["samples"]
            assert rows[0]["instance_seed"] == msa.instance_seed(97, 1)
        else:
            result = json.loads((output / label / "summary.json").read_text())
            with (output / label / "samples.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            prefixes = ("online", "static") if module is paired else (
                "online" if module is online else "static",
            )
            for prefix in prefixes:
                assert result[f"{prefix}_time_ms"] == milliseconds
                assert result[f"{prefix}_no_improvement_ms"] is None
                assert result[f"{prefix}_stopping_rule"] == "time_limit"
                assert float(rows[0][f"{prefix}_objective"]) > 0
        assert result["solve_seconds"] == milliseconds / 1000
        assert result["dataset_parameters"]["time_horizon"] == 10
        assert result["offset"] == 1
        assert len(rows) == 1
        assert int(rows[0]["sample"]) == 1

    csv_path = tmp_path / "msa.budget-summary.csv" if module is msa else output / "budget-summary.csv"
    with csv_path.open(newline="") as handle:
        comparison = list(csv.DictReader(handle))
    assert [float(row["solve_seconds"]) for row in comparison] == [0.01, 0.02]
    assert all(int(row["samples"]) == 1 for row in comparison)
    if module is msa:
        assert not output.exists()
    else:
        assert not (output / "summary.json").exists()


def test_msa_restarts_instance_rng_for_each_budget(tmp_path, monkeypatch):
    dataset = tmp_path / "data.txt"
    dataset.write_text(dataset_row(1) + dataset_row(2))
    draws = []

    def evaluate(args, problem, item, ortec=None, executor=None, return_details=False):
        assert args.solver_no_improvement_ms is None
        draws.append((args.solver_time_ms, np.random.random(3).tolist(), torch.rand(3).tolist()))
        details = {
            "wall_time_seconds": 0.01, "solver_wall_time_seconds": 0.01,
            "aggregate_solver_seconds": 0.01, "scenario_count": 1,
            "termination_counts": {"time_limit": 1}, "improvement_count": 0,
        }
        return float(args.solver_time_ms), 0.01, details

    monkeypatch.setattr(msa, "evaluate_instance", evaluate)
    msa.main([
        *cli_arguments(msa, dataset, tmp_path / "msa.json"), "--seed", "97",
        "--num-samples", "2", "--solve-seconds", "0.01", "0.02",
    ])

    assert [draw[0] for draw in draws] == [10, 10, 20, 20]
    assert draws[0][1:] == draws[2][1:]
    assert draws[1][1:] == draws[3][1:]
    first = json.loads((tmp_path / "msa.solve-0.01s.json").read_text())
    second = json.loads((tmp_path / "msa.solve-0.02s.json").read_text())
    assert first["objective_mean"] == 10
    assert second["objective_mean"] == 20
