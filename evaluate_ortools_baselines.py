#!/usr/bin/env python
"""Compatibility entry point for paired online and clairvoyant comparisons.

Use evaluate_online_ortools.py or evaluate_clairvoyant_ortools.py to run one
baseline independently. The paired CLI and existing result format remain
available here.
"""

from evaluate_clairvoyant_ortools import evaluate_static
from evaluate_online_ortools import evaluate_online, online_scenario
from utils.ortools_baselines import parse_args, run_evaluation, scenario_from_nodes, score


def main(argv=None):
    return run_evaluation(
        parse_args(argv), online_evaluator=evaluate_online, static_evaluator=evaluate_static
    )


if __name__ == "__main__":
    main()
