#!/usr/bin/env python
"""Compare online and MSA latency at the same initial decision state.

This is a controlled decision-timing probe, not an episode-quality evaluation.
Worker warm-up is recorded separately. Every repetition uses the same scenarios;
an optional serial arm solves those exact scenarios without the process pool.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

import evaluate_msa_ortools as msa
from evaluate_online_ortools import online_scenario
from utils import load_problem
from utils.data_utils import dataset_to_input


def warm_worker(_):
    """Load solver code before timing; report which pool processes are ready."""
    from ortools.constraint_solver import pywrapcp  # noqa: F401
    time.sleep(0.02)
    return os.getpid()


def warm_pool(executor, workers):
    started = time.perf_counter()
    seen = set()
    while len(seen) < workers:
        seen.update(executor.map(warm_worker, range(workers)))
        if time.perf_counter() - started > 60:
            raise RuntimeError("Not all scenario workers became ready within 60 seconds")
    return time.perf_counter() - started


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(msa.TW_PROBLEMS | {"pdtrp", "pdctrp"}), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dod", type=float, default=.5)
    parser.add_argument("--solve-seconds", type=float, default=1)
    parser.add_argument("--speed", type=float, default=4)
    parser.add_argument("--time-horizon", type=float, default=10)
    parser.add_argument("--latest-end", type=float, default=2)
    parser.add_argument("--ortec-locations")
    parser.add_argument("--include-serial", action="store_true")
    args = parser.parse_args(argv)
    if min(args.scenarios, args.num_samples, args.repeats) < 1:
        parser.error("scenarios, num-samples and repeats must be positive")
    if min(args.offset, args.seed) < 0:
        parser.error("offset and seed must be nonnegative")
    if not np.isfinite(args.solve_seconds) or round(args.solve_seconds * 1000) < 1:
        parser.error("solve-seconds must be finite and at least 0.001")
    if not 0 <= args.dod <= 1:
        parser.error("dod must be between zero and one")
    if min(args.speed, args.time_horizon) <= 0 or args.latest_end < 0:
        parser.error("speed/horizon must be positive and latest-end nonnegative")
    return args


def summarize(rows):
    summaries = {}
    for method in sorted({row["method"] for row in rows}):
        group = [r for r in rows if r["method"] == method]
        latencies = np.array([r["decision_seconds"] for r in group])
        summaries[method] = {
            "observations": len(group),
            "mean_seconds": float(latencies.mean()),
            "median_seconds": float(np.median(latencies)),
            "p95_seconds": float(np.percentile(latencies, 95)),
        }
    if summaries:
        reference = summaries["online"]["mean_seconds"]
        for values in summaries.values():
            values["mean_ratio_to_online"] = values["mean_seconds"] / reference
    return summaries


def run_evaluation(args):
    torch.set_num_threads(1)
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if available < args.scenarios + 1:
        raise ValueError(f"Request at least {args.scenarios + 1} CPUs for {args.scenarios} workers plus the parent")
    problem = load_problem(args.problem)
    dataset = problem.make_dataset(
        filename=args.dataset, batch_size=1, num_samples=args.num_samples, offset=args.offset,
        speed=args.speed, time_horizon=args.time_horizon, latest_end=args.latest_end,
        service_times_mean=3., service_times_var=5.,
    )
    if len(dataset) != args.num_samples:
        raise ValueError("Dataset does not contain the requested number of samples")
    ortec = msa.load_ortec_locations(args.ortec_locations, 1.) if args.ortec_locations else None
    time_ms = round(args.solve_seconds * 1000)
    rows = []
    methods = ["online", "msa_parallel"]
    if args.include_serial:
        methods.append("msa_serial")
    warm_worker(None)
    with msa.create_scenario_executor(args.scenarios) as executor:
        warmup = warm_pool(executor, args.scenarios)
        for index in range(len(dataset)):
            item = dataset[index]
            actual = dataset_to_input(item, args.problem, torch.device("cpu"))
            state = problem.make_state(actual)
            feasible = ~state.get_mask()[0, 0]
            if not feasible[1:].any():
                raise ValueError("The initial state has no feasible customer; choose another sample/DoD")
            sample = args.offset + index
            for repeat in range(args.repeats):
                # Rotate timing order to reduce systematic first/last bias.
                shift = (index + repeat) % len(methods)
                for method in methods[shift:] + methods[:shift]:
                    started = time.perf_counter()
                    if method == "online":
                        scenario = online_scenario(args.problem, actual, state, 9)
                        prepared = time.perf_counter()
                        details = msa.solve_scenario(scenario, args.problem, time_ms,
                                                     no_improvement_ms=None, return_details=True)
                        solved = time.perf_counter()
                        actions = [scenario["mapping"][node] for node in details["route"]
                                   if scenario["mapping"][node] is not None
                                   and feasible[scenario["mapping"][node]].item()]
                        if not actions:
                            raise RuntimeError("Online solve returned no executable first action")
                        action, worker_count = actions[0], 1
                        terminations = Counter([details["termination_reason"]])
                    else:
                        # Repeating the seed also pairs serial and parallel scenario inputs.
                        sample_seed = msa.instance_seed(args.seed, sample)
                        np.random.seed(sample_seed)
                        torch.manual_seed(sample_seed)
                        source_ids = msa.map_ortec_nodes(item, ortec) if ortec is not None else None
                        scenarios = [msa.make_scenario(
                            problem, args.problem, actual, state, len(item["all_nodes"]),
                            args.dod, 3, 9, ortec, source_ids, 1, [1], ["uniform"],
                        ) for _ in range(args.scenarios)]
                        prepared = time.perf_counter()
                        jobs = [(s, args.problem, time_ms, None) for s in scenarios]
                        dispatch = executor.map if method == "msa_parallel" else map
                        results = list(dispatch(msa.solve_prepared_scenario, jobs))
                        solved = time.perf_counter()
                        votes = [s["mapping"][d["route"][0]] for s,d in zip(scenarios,results) if d["route"]]
                        if not votes or any(v is None or not feasible[v].item() for v in votes):
                            raise RuntimeError("MSA solve returned no vote or an infeasible first action")
                        action = Counter(votes).most_common(1)[0][0]
                        worker_count = args.scenarios if method == "msa_parallel" else 1
                        terminations = Counter(d["termination_reason"] for d in results)
                    finished = time.perf_counter()
                    rows.append(dict(
                        sample=sample, repeat=repeat, method=method,
                        scenarios=1 if method == "online" else args.scenarios, workers=worker_count,
                        preparation_seconds=prepared-started, solver_seconds=solved-prepared,
                        selection_seconds=finished-solved, decision_seconds=finished-started,
                        selected_action=action, termination_counts=dict(terminations),
                    ))
    result = dict(
        problem=args.problem, dataset=str(Path(args.dataset).resolve()),
        dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        parameters=vars(args), experiment="initial_decision_latency",
        hardware=dict(hostname=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0))
                      if hasattr(os,"sched_getaffinity") else None,
                      slurm_cpus=os.environ.get("SLURM_CPUS_PER_TASK")),
        worker_warmup_seconds=warmup, samples=rows, summary=summarize(rows),
        limitations=["Common initial states only; these are not complete episode runtimes.",
                     "Warm worker pool; startup is reported separately from decision latency.",
                     "Equal per-solve budgets use different total CPU work and different routing scenarios.",
                     "Repeated wall-clock searches may return different first actions."],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "samples"}, indent=2))
    return result


def main(argv=None):
    return run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
