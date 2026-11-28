"""Pair the greedy test routes from one seed of the current-actor baseline trial."""

import argparse
from pathlib import Path

from compare_raw import greedy_results
from summarize import write_csv


def compare(trial, seed):
    baselines = ("rollout", "current-full", "current-revealed")
    records, paired = [], []
    costs = {}
    hashes = (trial / "rollout" / "evaluation-inputs.sha256").read_bytes()
    for baseline in baselines:
        case = trial / baseline
        if (case / "evaluation-inputs.sha256").read_bytes() != hashes:
            raise ValueError(f"Test datasets differ in {case}")
        rows, costs[baseline] = greedy_results(case, seed, clipping=0, baseline=baseline)
        records.extend(dict(seed=seed, baseline=baseline, **row) for row in rows)

    for control, candidate in (("rollout", "current-full"), ("current-full", "current-revealed"),
                               ("rollout", "current-revealed")):
        for (loc, dod), ref in costs[control].items():
            candidate_cost = costs[candidate][loc, dod]
            diff = candidate_cost - ref
            paired.append(dict(seed=seed, control=control, candidate=candidate, location=loc, dod=dod,
                               routes=len(ref), control_mean=ref.mean(), candidate_mean=candidate_cost.mean(),
                               mean_change=diff.mean(), pct_change=100 * diff.mean() / ref.mean(),
                               wins=int((diff < -1e-5).sum()), ties=int((abs(diff) <= 1e-5).sum()),
                               losses=int((diff > 1e-5).sum())))
            if loc == "all":
                print(f"Seed {seed}, {candidate} vs {control}: {diff.mean():+.3f} mean route cost")
    write_csv(trial / "metrics.csv", records)
    write_csv(trial / "paired.csv", paired)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_directory", type=Path)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    compare(args.seed_directory, args.seed)
