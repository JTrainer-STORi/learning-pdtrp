"""Pair greedy results on dataset 26413 from the continuation and clipping arrays."""

import argparse
from pathlib import Path

from compare_raw import greedy_results
from summarize import write_csv


def summarize(raw, clipped):
    methods = (
        ("raw-40", raw, 0, "current-revealed"),
        ("raw-100", raw, 0, "current-revealed"),
        ("clipped-100", clipped, 10, "current-revealed"),
        ("full-ppo", raw, 10, "rollout"),
        ("reference", raw, 10, "rollout"),
    )
    contrasts = (
        ("raw-40", "raw-100"),
        ("raw-100", "clipped-100"),
        ("full-ppo", "raw-100"),
        ("reference", "raw-100"),
        ("full-ppo", "clipped-100"),
        ("reference", "clipped-100"),
    )
    hashes = (raw / "seed-0" / "raw-40" / "evaluation-inputs.sha256").read_bytes()
    records, paired = [], []
    for seed in (0, 1, 2):
        costs = {}
        for method, root, clipping, baseline in methods:
            case = root / f"seed-{seed}" / method
            if (case / "evaluation-inputs.sha256").read_bytes() != hashes:
                raise ValueError(f"Test datasets differ in {case}")
            rows, costs[method] = greedy_results(case, seed, clipping, baseline, dataset_seed=26413)
            records.extend(dict(seed=seed, method=method, **row) for row in rows)

        for control, candidate in contrasts:
            for (location, dod), ref in costs[control].items():
                diff = costs[candidate][location, dod] - ref
                paired.append(dict(seed=seed, control=control, candidate=candidate, location=location,
                                   dod=dod, routes=len(ref), mean_change=diff.mean(),
                                   pct_change=100 * diff.mean() / ref.mean(),
                                   wins=int((diff < -1e-5).sum()), ties=int((abs(diff) <= 1e-5).sum()),
                                   losses=int((diff > 1e-5).sum())))
    write_csv(raw / "metrics.csv", records)
    write_csv(raw / "paired.csv", paired)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuation_directory", type=Path)
    parser.add_argument("clipped_directory", type=Path)
    args = parser.parse_args()
    summarize(args.continuation_directory, args.clipped_directory)
