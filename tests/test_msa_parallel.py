"""Check scenario ordering, process isolation and MSA search configuration."""

import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_msa_ortools as msa
from problems import PDTRP


CLI_ARGS = [
    "--problem", "pdtrp", "--dataset", "input.txt", "--output", "output.json",
    "--dod", "0.5",
]


def actual_input():
    return {
        "all_nodes": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]),
        "distance_matrix": torch.tensor([[[0.0, 1.0, 1.0], [2.0, 0.0, 1.0], [1.0, 1.0, 0.0]]]),
        "graph": torch.zeros(1, 3, 3, dtype=torch.bool),
        "speed": torch.tensor([1.0]),
        "time_horizon": torch.tensor([1000.0]),
        "latest_end": torch.tensor([100.0]),
        "service_times": torch.zeros(1, 3),
        "service_times_mean": torch.zeros(1),
        "service_times_std": torch.zeros(1),
        "arrival_times": torch.zeros(1, 3),
    }


def search_scenario():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                            [1.0, 1.0], [2.0, 0.3], [0.2, 2.0]])
    return {
        "distance": np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1),
        "service": np.zeros(6), "arrival_times": np.zeros(6), "speed": 1.0,
        "mapping": [None, 1, 2, 3, 4, 5], "windows": [(0.0, 1e6)] * 6,
        "demand": np.array([]), "capacity": 1.0, "reload_nodes": set(),
        "gamma": 0.0, "theta": 1.0,
    }


def evaluation_args(workers=1):
    return SimpleNamespace(
        problem="pdtrp", scenarios=2, solver_time_ms=None,
        solver_no_improvement_ms=10, workers=workers, dod=0.5,
        min_trips=1, max_trips=9, n_subregions=1,
        arrival_weights=None, arrival_skews=None,
    )


def coordinated_worker(job):
    """Require two active children without comparing machine-dependent speed."""
    directory, token = job
    directory = Path(directory)
    started = time.monotonic()
    (directory / f"{token}.ready").write_text(str(os.getpid()))
    while len(list(directory.glob("*.ready"))) < 2:
        if time.monotonic() - started > 20:
            raise RuntimeError("The second scenario worker did not start")
        time.sleep(0.01)
    return {
        "pid": os.getpid(), "started": started, "finished": time.monotonic(),
        "torch_threads": torch.get_num_threads(),
    }


def test_msa_defaults_have_patience_without_total_deadline():
    args = msa.parse_args(CLI_ARGS)

    assert args.solver_time_ms is None
    assert args.solver_no_improvement_ms == 60000
    assert args.workers == 1


def test_msa_accepts_auto_workers_and_explicit_search_limits():
    args = msa.parse_args([
        *CLI_ARGS, "--workers", "0", "--solver-time-ms", "80",
        "--solver-no-improvement-ms", "20",
    ])

    assert args.workers == 0
    assert args.solver_time_ms == 80
    assert args.solver_no_improvement_ms == 20


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--workers", "-1"),
        ("--scenarios", "0"),
        ("--scenarios", "-1"),
        ("--solver-no-improvement-ms", "0"),
        ("--solver-no-improvement-ms", "-1"),
        ("--solver-time-ms", "0"),
        ("--solver-time-ms", "-1"),
    ],
)
def test_msa_rejects_invalid_worker_and_search_limits(flag, value):
    with pytest.raises(SystemExit):
        msa.parse_args([*CLI_ARGS, flag, value])


def test_auto_workers_follow_affinity_and_do_not_exceed_scenarios(monkeypatch):
    monkeypatch.setattr(msa.os, "sched_getaffinity", lambda _: {2, 3, 4}, raising=False)

    assert msa.resolve_worker_count(0, scenarios=10) == 3
    assert msa.resolve_worker_count(0, scenarios=2) == 2
    assert msa.resolve_worker_count(7, scenarios=3) == 3
    assert msa.resolve_worker_count(1, scenarios=10) == 1


class OneDecisionState:
    def __init__(self):
        self.selected = None

    def all_finished(self):
        return self.selected is not None

    def get_mask(self):
        return torch.tensor([[[True, False, False]]])

    def update(self, selected):
        self.selected = int(selected.item())
        return self


class OneDecisionProblem:
    @staticmethod
    def make_state(actual):
        return OneDecisionState()

    @staticmethod
    def get_times(actual, tour):
        return torch.zeros(1, tour.shape[1] + 1), {}

    @staticmethod
    def get_costs(actual, tour):
        return tour[:, -1].float(), {}


def test_parent_sampling_and_tied_votes_are_stable_with_ordered_map(monkeypatch):
    actual = actual_input()
    monkeypatch.setattr(msa, "dataset_to_input", lambda *args: actual)

    def solve_prepared(job):
        scenario, problem_name, total_limit, patience = job
        assert problem_name == "pdtrp"
        assert total_limit is None
        assert patience == 10
        return {
            "route": [1], "elapsed_seconds": 0.25,
            "termination_reason": "no_improvement", "improvement_count": 1,
            "worker_pid": os.getpid(),
        }

    monkeypatch.setattr(msa, "solve_prepared_scenario", solve_prepared)

    def evaluate(workers):
        generated = []

        def sample(*args, **kwargs):
            # Sampling belongs to the parent, before dispatch. Two different
            # first-stage actions get one vote each; scenario order breaks ties.
            index = len(generated)
            random_draw = np.random.random(3).tolist()
            generated.append((os.getpid(), random_draw))
            return {"mapping": [None, 2 if index == 0 else 1], "draw": random_draw}

        monkeypatch.setattr(msa, "make_scenario", sample)

        class OrderedExecutor:
            shutdown_called = False

            def map(self, function, jobs):
                jobs = list(jobs)
                assert len(generated) == len(jobs) == 2
                assert [job[0]["draw"] for job in jobs] == [draw for _, draw in generated]
                return map(function, jobs)

            def shutdown(self, *args, **kwargs):
                self.shutdown_called = True

        executor = OrderedExecutor() if workers > 1 else None
        np.random.seed(8128)
        result = msa.evaluate_instance(
            evaluation_args(workers), OneDecisionProblem,
            {"all_nodes": actual["all_nodes"][0]}, executor=executor, return_details=True,
        )
        if executor is not None:
            assert executor.shutdown_called is False
        return generated, result

    sequential_draws, sequential = evaluate(1)
    dispatched_draws, dispatched = evaluate(2)

    assert sequential_draws == dispatched_draws
    assert all(pid == os.getpid() for pid, _ in dispatched_draws)
    assert sequential[0] == dispatched[0] == 2.0
    assert dispatched[2]["scenario_count"] == 2
    assert dispatched[2]["aggregate_solver_seconds"] == pytest.approx(0.5)
    assert dispatched[2]["termination_counts"] == {"no_improvement": 2}
    assert dispatched[2]["improvement_count"] == 2
    assert dispatched[2]["workers"] == 2


def test_spawned_workers_overlap_and_run_real_scenarios_without_total_cap(tmp_path):
    scenario = search_scenario()
    jobs = [(scenario, "pdtrp", None, 10) for _ in range(4)]
    sequential = [msa.solve_prepared_scenario(job) for job in jobs]

    with msa.create_scenario_executor(2) as executor:
        active = list(executor.map(coordinated_worker, [(str(tmp_path), 0), (str(tmp_path), 1)]))
        assert len({worker["pid"] for worker in active}) == 2
        assert all(worker["pid"] != os.getpid() for worker in active)
        assert all(worker["torch_threads"] == 1 for worker in active)
        assert max(worker["started"] for worker in active) < min(worker["finished"] for worker in active)

        parallel = list(executor.map(msa.solve_prepared_scenario, jobs))
        assert {result["worker_pid"] for result in parallel} <= {worker["pid"] for worker in active}
        assert all(result["worker_pid"] != os.getpid() for result in parallel)

        # Use the same pool for full MSA decisions, exercising sampled scenarios
        # and returning a valid scored tour without requiring deadline-sensitive
        # heuristic routes to match across processes.
        actual = actual_input()
        item = {key: value[0] if value.dim() > 1 else value for key, value in actual.items()}
        np.random.seed(45)
        torch.manual_seed(45)
        sequential_instance = msa.evaluate_instance(
            evaluation_args(1), PDTRP, item, return_details=True
        )
        np.random.seed(45)
        torch.manual_seed(45)
        parallel_instance = msa.evaluate_instance(
            evaluation_args(2), PDTRP, item, executor=executor, return_details=True
        )

    for result in sequential + parallel:
        assert sorted(result["route"]) == [1, 2, 3, 4, 5]
        assert result["termination_reason"] == "no_improvement"
        assert np.isfinite(result["solver_objective"])
    for result in (sequential_instance, parallel_instance):
        objective, runtime, details = result
        assert np.isfinite(objective)
        assert runtime >= 0
        assert details["scenario_count"] == 4
        assert sum(details["termination_counts"].values()) == 4
        assert details["aggregate_solver_seconds"] >= 0
        assert details["solver_no_improvement_ms"] == 10
        assert details["solver_time_ms"] is None
