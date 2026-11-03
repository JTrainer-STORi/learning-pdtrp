"""Exercise comparison manifests and local execution without submitting jobs."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

import comparison_jobs as jobs


def make_plan(tmp_path, *, samples=3, methods=("online", "oracle", "msa"),
              seeds=(0, 1), max_array=1000, total_cpus=512, total_memory_gib=1800,
              service_prior=None):
    dataset = tmp_path / "data.txt"
    dataset.write_text(
        "0 0 1 0 0 1 arrival_times 0 0 0 service_times 0 0 0 "
        "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n" * samples
    )
    cases = tmp_path / "cases.json"
    case = {"problem": "pdtrp", "dataset": "data.txt", "dod": 0.5}
    case.update(service_prior or {})
    cases.write_text(json.dumps([case]))
    output = tmp_path / "run"
    jobs.main([
        "plan", "--cases", str(cases), "--output", str(output),
        "--methods", *methods, "--num-samples", str(samples),
        "--seeds", *map(str, seeds), "--scenarios", "2", "--msa-workers", "2",
        "--max-array-size", str(max_array), "--total-cpus", str(total_cpus),
        "--total-memory-gib", str(total_memory_gib), "--no-improvement-ms", "5",
    ])
    path = output / "plan.json"
    return path, json.loads(path.read_text())


def first_task(plan, method="online"):
    task = next(task for task in plan["tasks"] if task["method"] == method)
    array = next(array for array in plan["arrays"] if task["id"] in array["task_ids"])
    return task, array, array["task_ids"].index(task["id"])


def test_mixed_plan_accounts_for_all_methods_and_parent_process_cpus(tmp_path):
    _, plan = make_plan(tmp_path)

    assert len(plan["tasks"]) == 12
    assert {task["method"] for task in plan["tasks"]} == {"online", "oracle", "msa"}
    assert plan["resources"]["peak_cpus"] <= 512
    assert plan["resources"]["peak_memory_mib"] <= 1800 * 1024
    for task in plan["tasks"]:
        assert task["cpus_per_task"] == (task["workers"] + 1 if task["method"] == "msa" else 1)
        assert task["memory_mib"] == task["cpus_per_task"] * 3072
        assert task["time_limit_ms"] is None
        assert task["no_improvement_ms"] == 5
        assert task["dataset"]["sha256"]
        assert Path(task["dataset"]["path"]).is_absolute()
    expected_ids = {task["id"] for task in plan["tasks"]}
    dispatched_ids = [task_id for array in plan["arrays"] for task_id in array["task_ids"]]
    assert set(dispatched_ids) == expected_ids
    assert len(dispatched_ids) == len(expected_ids)


def test_memory_budget_limits_combined_groups(tmp_path):
    _, plan = make_plan(tmp_path, total_cpus=16, total_memory_gib=24)

    assert plan["resources"]["peak_cpus"] <= 8
    assert plan["resources"]["peak_memory_mib"] <= 24 * 1024


def test_array_batches_chain_with_one_concurrent_batch_per_group(tmp_path):
    _, plan = make_plan(tmp_path, samples=5, max_array=2)
    groups = {}
    for array in plan["arrays"]:
        groups.setdefault(array["group"], []).append(array)
        assert 1 <= len(array["task_ids"]) <= 2
    assert all(len(arrays) > 1 for arrays in groups.values())
    for arrays in groups.values():
        assert arrays[0]["after"] is None
        for previous, following in zip(arrays, arrays[1:]):
            assert following["after"] == previous["name"]
    # Batch chaining makes the peak the sum of one array from each group,
    # rather than the sum of every batch in the manifest.
    peak_cpus = sum(max(array["cpus_per_task"] * array["concurrency"] for array in arrays)
                    for arrays in groups.values())
    peak_memory = sum(max(array["memory_mib"] * array["concurrency"] for array in arrays)
                      for arrays in groups.values())
    assert peak_cpus == plan["resources"]["peak_cpus"] <= 512
    assert peak_memory == plan["resources"]["peak_memory_mib"] <= 1800 * 1024


def test_submit_is_a_dry_run_without_explicit_execute(tmp_path, monkeypatch):
    plan_path, _ = make_plan(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run submission executed an external command")

    monkeypatch.setattr(jobs.subprocess, "run", forbidden)
    result = jobs.submit_plan(plan_path)
    assert "sbatch" in json.dumps(result)


def test_execute_submission_chains_real_job_ids_and_validates_sbatch_output(tmp_path, monkeypatch):
    plan_path, plan = make_plan(tmp_path, samples=5, max_array=2)
    calls = []
    job_ids = {}

    def sbatch(command, **kwargs):
        calls.append(command)
        assert command[0] == "sbatch"
        job_id = str(100 + len(calls))
        array = plan["arrays"][len(calls) - 1]
        command_text = " ".join(map(str, command))
        if array["after"]:
            assert f"afterany:{job_ids[array['after']]}" in command_text
        else:
            assert "--dependency" not in command_text
        job_ids[array["name"]] = job_id
        return subprocess.CompletedProcess(command, 0, stdout=job_id + ";cluster\n", stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", sbatch)
    jobs.submit_plan(plan_path, execute=True)
    assert len(calls) == len(plan["arrays"])

    invalid_calls = []

    def invalid_sbatch(command, **kwargs):
        invalid_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="invalid job id\n", stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", invalid_sbatch)
    with pytest.raises(RuntimeError, match="Unexpected sbatch job id"):
        jobs.submit_plan(plan_path, execute=True, retry=True)
    assert len(invalid_calls) == 1


def test_retry_submits_only_unfinished_indices_after_previous_jobs(tmp_path, monkeypatch):
    plan_path, plan = make_plan(tmp_path, samples=3, methods=("online",), max_array=2)
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", submit)
    jobs.submit_plan(plan_path, execute=True)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="already submitted"):
        jobs.submit_plan(plan_path, execute=True)
    assert len(calls) == 2

    task, array, index = first_task(plan)
    monkeypatch.setattr(jobs.subprocess, "run", write_fake_result)
    jobs.run_task(plan_path, array["name"], index)
    monkeypatch.setattr(jobs.subprocess, "run", submit)
    jobs.submit_plan(plan_path, execute=True, retry=True)
    assert "--dependency=afterany:102" in calls[2]
    assert any(item.startswith("--array=1%") for item in calls[2])
    assert "--dependency=afterany:103" in calls[3]


def test_plan_keeps_the_virtual_environment_interpreter_symlink(tmp_path):
    dataset = tmp_path / "data.txt"
    dataset.write_text("one instance\n")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"problem": "pdtrp", "dataset": "data.txt", "dod": 0.5}]))
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)

    jobs.main([
        "plan", "--cases", str(cases), "--output", str(tmp_path / "run"),
        "--num-samples", "1", "--python", str(interpreter),
    ])

    plan = jobs.load_plan(tmp_path / "run" / "plan.json")
    assert plan["python"] == str(interpreter)
    assert plan["python"] != str(interpreter.resolve())


def test_real_local_tasks_preserve_ids_resume_and_collect_all_methods(tmp_path):
    plan_path, plan = make_plan(tmp_path, samples=1, seeds=(0,))

    for array in plan["arrays"]:
        for index in range(len(array["task_ids"])):
            assert jobs.run_task(plan_path, array["name"], index)["skipped"] is False
            assert jobs.run_task(plan_path, array["name"], index)["skipped"] is True

    summary = jobs.collect_results(plan_path)
    assert summary["completed"] == summary["expected"] == 3
    assert not summary["missing_task_ids"]
    assert {group["method"] for group in summary["groups"]} == {"online", "oracle", "msa"}
    with (plan_path.parent / "collected" / "samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(int(row["sample"]) == 0 for row in rows)


def test_each_hec_method_uses_its_own_entry_point(tmp_path):
    _, plan = make_plan(tmp_path, samples=1, seeds=(0,))
    expected_scripts = {
        "online": "evaluate_online_ortools.py",
        "oracle": "evaluate_clairvoyant_ortools.py",
        "msa": "evaluate_msa_ortools.py",
    }
    for task in plan["tasks"]:
        command = jobs.evaluation_command(plan, task, tmp_path)
        assert Path(command[2]).name == expected_scripts[task["method"]]
        assert "--method" not in command


def test_hec_plan_records_and_passes_known_service_prior(tmp_path):
    prior = {"service_times_mean": 7.0, "service_times_var": 2.0}
    _, plan = make_plan(tmp_path, samples=1, seeds=(0,), service_prior=prior)
    for task in plan["tasks"]:
        assert task["service_time_prior"] == prior
        command = jobs.evaluation_command(plan, task, tmp_path)
        if task["method"] == "oracle":
            assert "--service-times-mean" not in command
            assert "--service-times-var" not in command
        else:
            assert command[command.index("--service-times-mean") + 1] == "7.0"
            assert command[command.index("--service-times-var") + 1] == "2.0"


def test_collector_keeps_different_service_priors_separate(tmp_path, monkeypatch):
    make_plan(tmp_path, samples=1, methods=("online",))
    cases_path = tmp_path / "cases.json"
    case = json.loads(cases_path.read_text())[0]
    cases_path.write_text(json.dumps([case, {**case, "service_times_mean": 7.0}]))
    output = tmp_path / "different-priors"
    jobs.main([
        "plan", "--cases", str(cases_path), "--output", str(output),
        "--methods", "online", "--num-samples", "1",
    ])
    plan_path = output / "plan.json"
    plan = jobs.load_plan(plan_path)
    monkeypatch.setattr(jobs.subprocess, "run", write_fake_result)
    for array in plan["arrays"]:
        for index in range(len(array["task_ids"])):
            jobs.run_task(plan_path, array["name"], index)

    summary = jobs.collect_results(plan_path)
    assert len(summary["groups"]) == 2
    assert {group["service_times_mean"] for group in summary["groups"]} == {3.0, 7.0}
    assert all(group["service_time_model"] == "expected" for group in summary["groups"])
    with (output / "collected" / "samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {float(row["service_times_mean"]) for row in rows} == {3.0, 7.0}


@pytest.mark.parametrize("prior", [
    {"service_times_mean": -1}, {"service_times_var": -1},
    {"service_times_mean": float("nan")}, {"service_times_var": float("inf")},
    {"service_times_mean": 0, "service_times_var": 5},
])
def test_hec_plan_rejects_invalid_service_prior(tmp_path, prior):
    with pytest.raises(SystemExit):
        make_plan(tmp_path, service_prior=prior)


def write_fake_result(command, **kwargs):
    """Write the public evaluator result format at the supplied staging path."""
    command = list(map(str, command))
    argument = lambda name: command[command.index(name) + 1]
    output = Path(argument("--output"))
    sample = int(argument("--offset"))
    method = "online" if Path(command[2]).name == "evaluate_online_ortools.py" else "oracle"
    prefix = "online" if method == "online" else "static"
    output.mkdir(parents=True, exist_ok=True)
    row = {"sample": sample, f"{prefix}_objective": 3.0, f"{prefix}_runtime_seconds": 0.01}
    if method == "online":
        row.update(online_solve_count=2, online_improvement_count=0,
                   online_termination_reasons=json.dumps({"no_improvement": 2}), online_tour="[0, 1, 2]")
    else:
        row.update(static_solver_objective=3.0, static_solver_status=1,
                   static_termination_reason="no_improvement", static_improvement_count=0,
                   static_tour="[0, 1, 2]")
    with (output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (output / "summary.json").write_text(json.dumps({
        "problem": argument("--problem"), "dataset": argument("--dataset"),
        "offset": sample, "samples": 1, "method": method,
        f"{prefix}_objective_mean": 3.0,
    }))
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_task_completion_is_atomic_and_resume_checks_fingerprint(tmp_path, monkeypatch):
    plan_path, plan = make_plan(tmp_path, samples=1, methods=("online",))
    task, array, index = first_task(plan)
    destination = plan_path.parent / "results" / task["id"]
    calls = []

    def evaluate(command, **kwargs):
        assert not destination.exists()
        calls.append(command)
        return write_fake_result(command, **kwargs)

    monkeypatch.setattr(jobs.subprocess, "run", evaluate)
    jobs.run_task(plan_path, array["name"], index)
    marker_path = destination / "completed.json"
    marker = json.loads(marker_path.read_text())
    assert marker["status"] == "complete"
    assert marker["fingerprint"] == task["fingerprint"]
    assert marker["task_id"] == task["id"]
    assert marker["sample"] == task["sample"]
    jobs.run_task(plan_path, array["name"], index)
    assert len(calls) == 1

    marker["fingerprint"] = "a different task configuration"
    marker_path.write_text(json.dumps(marker))
    with pytest.raises((ValueError, RuntimeError)):
        jobs.run_task(plan_path, array["name"], index)
    assert len(calls) == 1


def test_failed_task_never_creates_completion_marker(tmp_path, monkeypatch):
    plan_path, plan = make_plan(tmp_path, samples=1, methods=("online",))
    task, array, index = first_task(plan)

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(jobs.subprocess, "run", fail)
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        jobs.run_task(plan_path, array["name"], index)
    assert not (plan_path.parent / "results" / task["id"] / "completed.json").exists()


def test_collector_requires_complete_results_unless_partial_requested(tmp_path, monkeypatch):
    plan_path, plan = make_plan(tmp_path, samples=2, methods=("online",))
    task, array, index = first_task(plan)
    monkeypatch.setattr(jobs.subprocess, "run", write_fake_result)
    jobs.run_task(plan_path, array["name"], index)

    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        jobs.collect_results(plan_path)
    jobs.collect_results(plan_path, allow_partial=True)
    with (plan_path.parent / "collected" / "samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert int(rows[0]["sample"]) == task["sample"]


@pytest.mark.parametrize("damage", ["decode", "identity"])
def test_collector_rejects_malformed_or_wrong_identity_completion(tmp_path, monkeypatch, damage):
    plan_path, plan = make_plan(tmp_path, samples=1, methods=("online",))
    task, array, index = first_task(plan)
    monkeypatch.setattr(jobs.subprocess, "run", write_fake_result)
    jobs.run_task(plan_path, array["name"], index)
    marker_path = plan_path.parent / "results" / task["id"] / "completed.json"
    if damage == "decode":
        marker_path.write_text("not JSON")
    else:
        marker = json.loads(marker_path.read_text())
        marker["sample"] += 1
        marker_path.write_text(json.dumps(marker))
    with pytest.raises((ValueError, RuntimeError)):
        jobs.collect_results(plan_path)
