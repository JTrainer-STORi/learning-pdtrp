"""Scaled call horizons reach every evaluator and frozen HEC task configuration."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import comparison_jobs as jobs
import evaluate_clairvoyant_ortools as oracle
import evaluate_model_free as model_free
import evaluate_msa_ortools as msa
import evaluate_online_ortools as online
import evaluate_ortools_baselines as paired
import evaluate_trained_policy as policy
import generate_data
from problems import PDTRP
from utils.data_utils import dataset_to_input


MODULES = (oracle, model_free, msa, online, paired, policy)


def cli_arguments(module):
    if module is policy:
        return ["--problem", "pdtrp", "--checkpoint", "model.pt", "--datasets", "data.txt",
                "--output-dir", "output", "--device", "cpu"]
    return ["--problem", "pdtrp", "--dataset", "data.txt", "--output", "output"] + (
        ["--dod", "0.5"] if module is msa else []
    )


@pytest.mark.parametrize("module", MODULES)
def test_evaluation_clis_accept_fractional_horizons(module):
    args = module.parse_args(cli_arguments(module) + ["--time-horizon", "10.5"])
    assert args.time_horizon == 10.5
    assert args.speed == 4.0
    assert args.latest_end == 2.0


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("flags", [["--time-horizon", "0"], ["--speed", "-1"], ["--latest-end", "nan"]])
def test_evaluation_clis_reject_invalid_time_settings(module, flags):
    with pytest.raises(SystemExit):
        module.parse_args(cli_arguments(module) + flags)


@pytest.fixture
def scaled_dataset(tmp_path):
    output = tmp_path / "pdtrp50-th10.txt"
    generate_data.generate(generate_data.parse_args([
        "--problem", "pdtrp", "--num-customers", "50", "--dod", "0.5",
        "--time-horizon", "10", "--num-samples", "1", "--output", str(output),
    ]))
    return output


@pytest.mark.parametrize("module", [online, oracle, paired])
def test_ortools_evaluators_load_scaled_horizon(monkeypatch, tmp_path, scaled_dataset, module):
    calls = []

    def evaluate(args, problem, actual, return_details=False):
        assert actual["time_horizon"].item() == 600
        assert actual["arrival_times"].max() > 480
        calls.append(True)
        details = {"solve_count": 1, "improvement_count": 0, "termination_reasons": {},
                   "solver_objective": 1.0, "solver_status": 1, "termination_reason": "time_limit"}
        return 1.0, 0.01, [0, 1], details

    if module is not oracle:
        monkeypatch.setattr(module, "evaluate_online", evaluate)
    if module is not online:
        monkeypatch.setattr(module, "evaluate_static", evaluate)
    output = tmp_path / "results"
    module.main(["--problem", "pdtrp", "--dataset", str(scaled_dataset), "--output", str(output),
                 "--num-samples", "1", "--time-horizon", "10"])
    assert calls
    assert json.loads((output / "summary.json").read_text())["dataset_parameters"]["time_horizon"] == 10


def test_msa_uses_scaled_horizon_for_actual_and_future_requests(monkeypatch, tmp_path, scaled_dataset):
    def evaluate(args, problem, item, ortec=None, executor=None, return_details=False):
        assert item["time_horizon"].item() == 600
        actual = dataset_to_input(item, "pdtrp", "cpu")
        np.random.seed(123)
        scenario = msa.make_scenario(problem, "pdtrp", actual, problem.make_state(actual), 51, 0.5, 3, 9)
        future = [i for i, node in enumerate(scenario["mapping"]) if i > 0 and node is None]
        assert future
        assert 480 < max(scenario["arrival_times"][future]) <= 600
        return 1.0, 0.01, {"termination_counts": {}, "scenario_count": 1,
                           "aggregate_solver_seconds": 0.0, "solver_wall_time_seconds": 0.0,
                           "improvement_count": 0, "workers": 1}

    monkeypatch.setattr(msa, "evaluate_instance", evaluate)
    output = tmp_path / "msa.json"
    msa.main(["--problem", "pdtrp", "--dataset", str(scaled_dataset),
              "--output", str(output), "--dod", "0.5",
              "--num-samples", "1", "--time-horizon", "10"])
    result = json.loads(output.read_text())
    assert result["dataset_parameters"]["time_horizon"] == 10


def test_policy_receives_scaled_horizon(monkeypatch, tmp_path, scaled_dataset):
    class InspectingPolicy(torch.nn.Module):
        def set_decode_type(self, decode_type):
            pass

        def forward(self, actual, mode):
            assert actual["time_horizon"].item() == 600
            assert actual["all_nodes"].shape == (1, 51, 2)
            return torch.ones(1), None, {}

    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setattr(policy, "load_model", lambda path: (InspectingPolicy(), {}))
    result = policy.main(["--problem", "pdtrp", "--datasets", str(scaled_dataset),
                          "--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "policy"),
                          "--device", "cpu", "--time-horizon", "10", "--num-samples", "1"])
    assert result["dataset_parameters"]["time_horizon"] == 10


def test_hec_plan_freezes_horizon_and_passes_it_to_all_methods(tmp_path, scaled_dataset):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"problem": "pdtrp", "dataset": str(scaled_dataset),
                                  "dod": 0.5, "time_horizon": 10}]))
    plan_path = tmp_path / "plan" / "plan.json"
    jobs.main(["plan", "--cases", str(cases), "--output", str(plan_path.parent),
               "--num-samples", "1", "--seeds", "0", "--scenarios", "1"])
    plan = json.loads(plan_path.read_text())
    for task in plan["tasks"]:
        assert task["dataset_parameters"]["time_horizon"] == 10
        command = jobs.evaluation_command(plan, task, tmp_path)
        assert command[command.index("--time-horizon") + 1] == "10"


def test_generated_scaled_dataset_loads_exactly_as_recorded(scaled_dataset):
    metadata = json.loads(scaled_dataset.with_suffix(".txt.json").read_text())
    dataset = PDTRP.make_dataset(filename=str(scaled_dataset), **metadata["dataset_options"])
    sample = dataset[0]
    assert sample["all_nodes"].shape == (51, 2)
    assert sample["arrival_times"].gt(0).sum() == 25
    assert sample["time_horizon"].item() == 600
    assert sample["arrival_times"].max() > 480
