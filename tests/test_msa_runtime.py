"""Check timing-study configuration and distinguish latency from episode scores."""

import pytest

import evaluate_msa_runtime as timing


CLI = ["--problem", "pdtrp", "--dataset", "input.txt", "--output", "timing.json"]


@pytest.mark.parametrize("flag,value", [
    ("--scenarios", "0"), ("--num-samples", "0"), ("--repeats", "0"),
    ("--offset", "-1"), ("--seed", "-1"), ("--solve-seconds", "0.0001"),
    ("--solve-seconds", "nan"), ("--dod", "1.1"),
])
def test_invalid_runtime_options(flag, value):
    with pytest.raises(SystemExit):
        timing.parse_args([*CLI, flag, value])


def test_latency_summary_keeps_methods_separate():
    rows = [dict(method="online", decision_seconds=1.),
            dict(method="online", decision_seconds=3.),
            dict(method="msa_parallel", decision_seconds=4.),
            dict(method="msa_parallel", decision_seconds=8.)]
    summary = timing.summarize(rows)
    assert summary["online"]["mean_seconds"] == 2.
    assert summary["msa_parallel"]["mean_seconds"] == 6.
    assert summary["msa_parallel"]["mean_ratio_to_online"] == 3.
    assert summary["msa_parallel"]["observations"] == 2


def test_pool_requires_a_core_per_worker_plus_parent(monkeypatch):
    monkeypatch.setattr(timing.os, "sched_getaffinity", lambda _: {0, 1})
    args = timing.parse_args([*CLI, "--scenarios", "2"])
    with pytest.raises(ValueError, match="at least 3 CPUs"):
        timing.run_evaluation(args)
