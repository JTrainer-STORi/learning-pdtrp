#!/usr/bin/env python
"""Evaluate online OR-Tools reoptimization using only revealed requests."""

from collections import Counter
import time

import torch

from evaluate_msa_ortools import CAPACITY_PROBLEMS, TW_PROBLEMS, solve_scenario
from utils.ortools_baselines import (
    parse_args as parse_baseline_args,
    run_evaluation,
    scenario_from_nodes,
    score,
)
from utils.service_times import service_time_parameters


def online_scenario(problem_name, actual, state, max_trips):
    visited = state.visited_[0, 0]
    arrived = ~state.not_arrived_[0, 0]
    node_ids = torch.nonzero((~visited) & arrived).flatten().tolist()
    node_ids = [node for node in node_ids if node != 0]
    mean, _ = service_time_parameters(actual)
    scenario = scenario_from_nodes(
        problem_name,
        actual,
        int(state.prev_a.item()),
        node_ids,
        float(state.used_cap.item()) if problem_name in CAPACITY_PROBLEMS else 0.0,
        max_trips,
        service_time_mean=mean,
    )
    if problem_name in TW_PROBLEMS:
        current_time = float(state.cur_times.item())
        scenario["windows"] = [
            scenario["windows"][0],
            *[
                (start - current_time, end - current_time)
                for start, end in scenario["windows"][1:]
            ],
        ]
        scenario["arrival_times"] = scenario["arrival_times"] - current_time
        scenario["arrival_times"][0] = 0.0
    return scenario


def evaluate_online(args, problem, actual, *, return_details=False):
    state = problem.make_state(actual)
    tour = [0]
    runtime = 0.0
    termination_reasons = Counter()
    improvement_count = 0
    solve_count = 0
    while not state.all_finished():
        feasible = ~state.get_mask()[0, 0]
        feasible_customers = feasible.clone()
        feasible_customers[0] = False
        if not feasible_customers.any():
            if not feasible[0]:
                raise RuntimeError("Unfinished online instance has no feasible action")
            selected = 0
        else:
            scenario = online_scenario(
                args.problem, actual, state, args.max_trips
            )
            started = time.perf_counter()
            details = solve_scenario(
                scenario,
                args.problem,
                args.online_time_ms,
                no_improvement_ms=getattr(args, "online_no_improvement_ms", 60000),
                return_details=True,
            )
            route = details["route"]
            runtime += time.perf_counter() - started
            termination_reasons[details["termination_reason"]] += 1
            improvement_count += details["improvement_count"]
            solve_count += 1
            selected = next(
                (
                    scenario["mapping"][node]
                    for node in route
                    if scenario["mapping"][node] is not None
                    and feasible[scenario["mapping"][node]].item()
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    "Online OR-Tools solve returned no feasible action"
                )
        state = state.update(torch.tensor([selected], dtype=torch.long))
        if not hasattr(state, "last_action_executed") or state.last_action_executed.item():
            tour.append(selected)
    tour_tensor = torch.tensor([tour], dtype=torch.long)
    visit_times = torch.tensor(state.get_visit_times()) if hasattr(state, "get_visit_times") else None
    result = (score(problem, args.problem, actual, tour_tensor, visit_times), runtime, tour)
    details = {
        "action_policy": "execute_first_feasible_planned_action",
        "solve_count": solve_count,
        "termination_reasons": dict(termination_reasons),
        "improvement_count": improvement_count,
    }
    return (*result, details) if return_details else result


def parse_args(argv=None):
    return parse_baseline_args(argv, method="online")


def main(argv=None):
    return run_evaluation(parse_args(argv), online_evaluator=evaluate_online)


if __name__ == "__main__":
    main()
