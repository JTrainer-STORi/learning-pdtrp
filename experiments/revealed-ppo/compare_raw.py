"""Compare one raw-score revealed PPO seed with its archived clipped greedy control."""

import argparse
from pathlib import Path

import numpy as np

from summarize import DODS, LOCATIONS, load_manifest, metrics, read_cell, write_csv


def comparison(raw_cost, clipped_cost):
    change = raw_cost - clipped_cost
    return dict(routes=len(change), raw_mean=raw_cost.mean(), clipped_mean=clipped_cost.mean(),
                mean_change=change.mean(), pct_change=100 * change.mean() / clipped_cost.mean(),
                wins=int((change < -1e-5).sum()), ties=int((abs(change) <= 1e-5).sum()),
                losses=int((change > 1e-5).sum()))


def greedy_results(case, seed, clipping, baseline=None):
    hashes = (case / "evaluation-inputs.sha256").read_text()
    expected = {f"{loc}/pdctrptw50-dod{dod}-th10-seed26412.txt" for loc in LOCATIONS for dod in DODS}
    if {line.split()[1] for line in hashes.splitlines()} != expected:
        raise ValueError("Expected checksums for all six evaluation datasets")

    _, checkpoint = (case / "checkpoint.sha256").read_text().strip().split(maxsplit=1)
    blocks, times = [], []
    costs = {}
    records = []
    for loc in LOCATIONS:
        folder = case / "candidates-0" / loc
        manifest = load_manifest(folder, seed, 0, 300, checkpoint)
        if baseline and manifest.get("training_baseline") != baseline:
            raise ValueError(f"Unexpected training baseline in {folder}")
        # The archived clipped manifests predate the tanh_clipping field.
        if manifest.get("tanh_clipping", 10) != clipping:
            raise ValueError(f"Unexpected decoder clipping in {folder}")
        for dod in DODS:
            stem = f"pdctrptw50-dod{dod}-th10-seed26412"
            rows, timing = read_cell(folder, stem, 0, 300, manifest)
            costs[loc, dod] = rows["objective"]
            records.append(dict(location=loc, dod=dod, **metrics(rows, timing)))
            blocks.append(rows)
            times.append(timing)
    rows = np.concatenate(blocks)
    costs["all", "all"] = rows["objective"]
    records.append(dict(location="all", dod="all",
                        **metrics(rows, np.concatenate(times))))
    return records, costs


def compare(raw, clipped, seed):
    hashes = (raw / "evaluation-inputs.sha256").read_text()
    if hashes != (clipped / "evaluation-inputs.sha256").read_text():
        raise ValueError("Raw and clipped evaluations used different datasets")
    records, paired = [], []
    costs = {}
    for mode, case, clipping in (("raw", raw, 0), ("clipped", clipped, 10)):
        rows, costs[mode] = greedy_results(case, seed, clipping)
        records.extend(dict(seed=seed, decoder=mode, **row) for row in rows)
    for (loc, dod), raw_cost in costs["raw"].items():
        paired.append(dict(seed=seed, location=loc, dod=dod,
                           **comparison(raw_cost, costs["clipped"][loc, dod])))
    write_csv(raw / "metrics.csv", records)
    write_csv(raw / "paired.csv", paired)
    total = paired[-1]
    print(f"Seed {seed}, {total['routes']} paired greedy routes: raw {total['raw_mean']:.3f}, "
          f"clipped {total['clipped_mean']:.3f}, change {total['pct_change']:+.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("clipped", type=Path)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    compare(args.raw, args.clipped, args.seed)
