"""Report matched PDCTRPTW results from revealed-ppo-evaluate.sh, pooling actual decision times."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARMS = ("full", "revealed", "reference")
LOCATIONS = ("uniform-uniform", "ortec-locations")
DODS = (0.2, 0.5, 0.8)


def read_cell(directory, stem, width, samples, manifest):
    summary = json.loads((directory / f"{stem}.summary.json").read_text())
    if summary not in manifest["datasets"] or summary["samples"] != samples:
        raise ValueError(f"Incomplete evaluation of {stem} in {directory}")
    if (Path(summary["dataset"]).parts[-2:] != (directory.name, f"{stem}.txt")
            or summary["checkpoint"] != manifest["checkpoint"]
            or summary["rollout_candidates"] != width or summary["offset"] != 0):
        raise ValueError(f"Unexpected dataset, mode or checkpoint in {directory}/{stem}")
    routes = np.genfromtxt(directory / f"{stem}.samples.csv", delimiter=",", names=True, ndmin=1)
    decisions = np.loadtxt(directory / f"{stem}.decisions.csv", delimiter=",", skiprows=1, ndmin=2)
    if not np.array_equal(routes["sample"], np.arange(samples)):
        raise ValueError(f"Missing or unordered sample IDs in {directory}/{stem}")
    counts = routes["decisions"].astype(int)
    if ((counts < 1).any() or (counts != routes["decisions"]).any() or
            not np.array_equal(decisions[:, 0], np.repeat(routes["sample"], counts)) or
            not np.array_equal(decisions[:, 1], np.concatenate([np.arange(n) for n in counts]))):
        raise ValueError(f"Decision rows do not match samples in {directory}/{stem}")
    if (not np.isfinite(routes.view(float)).all() or not np.isfinite(decisions).all()
            or not np.allclose(routes["objective"], routes["distance_penalty"] + routes["lateness_penalty"])):
        raise ValueError(f"Invalid route metrics in {directory}/{stem}")
    return routes, decisions[:, 2]


def metrics(routes, timings):
    return {
        "routes": len(routes), "mean_cost": routes["objective"].mean(),
        "p95_cost": np.quantile(routes["objective"], .95),
        "mean_distance": routes["distance_penalty"].mean(),
        "mean_lateness_minutes": routes["lateness_penalty"].mean(),
        "routes_over_10_minutes_late": int((routes["lateness_penalty"] > 10).sum()),
        "mean_route_seconds": routes["runtime_seconds"].mean(),
        "p95_decision_ms": 1000 * np.quantile(timings, .95),
    }


def write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def summarize(root, seeds=(0, 1, 2), samples=300, arms=ARMS):
    reference_hashes = None
    records, paired = [], []
    for seed in seeds:
        costs = {}
        for arm in arms:
            case = root / arm / f"seed-{seed}"
            hashes = (case / "evaluation-inputs.sha256").read_text().splitlines()
            expected_files = {f"{loc}/pdctrptw50-dod{dod}-th10-seed26412.txt" for loc in LOCATIONS for dod in DODS}
            hashes = {name: digest for digest, name in (line.split() for line in hashes)}
            if hashes.keys() != expected_files or (reference_hashes is not None and hashes != reference_hashes):
                raise ValueError(f"Evaluation dataset hashes differ or are incomplete in {case}")
            reference_hashes = hashes
            _, checkpoint = (case / "checkpoint.sha256").read_text().strip().split(maxsplit=1)
            for width in (0, 2):
                blocks, times = [], []
                for loc in LOCATIONS:
                    folder = case / f"candidates-{width}" / loc
                    manifest = json.loads((folder / "manifest.json").read_text())
                    expected = dict(training_seed=seed, evaluation_seed=12346, problem="pdctrptw",
                                    offset=0, rollout_candidates=width, rollout_wait_penalty=0,
                                    num_samples_requested=samples, record_decision_times=True,
                                    dataset_parameters=dict(speed=4, time_horizon=10, latest_end=2))
                    # sha256sum and Python may record different mount prefixes on HEC.
                    if (any(manifest.get(k) != v for k, v in expected.items()) or
                            Path(checkpoint).parts[-4:] != Path(manifest["checkpoint"]).parts[-4:]):
                        raise ValueError(f"Unexpected evaluation settings in {folder}")
                    for dod in DODS:
                        stem = f"pdctrptw50-dod{dod}-th10-seed26412"
                        rows, t = read_cell(folder, stem, width, samples, manifest)
                        blocks.append(rows)
                        times.append(t)
                        records.append(dict(seed=seed, arm=arm, width=width, location=loc,
                                            dod=dod, **metrics(rows, t)))
                routes = np.concatenate(blocks)
                costs[arm, width] = routes["objective"]
                records.append(dict(seed=seed, arm=arm, width=width, location="all",
                                    dod="all", **metrics(routes, np.concatenate(times))))
        comparisons = [(("full", w), ("revealed", w)) for w in (0, 2)]
        comparisons.extend(((arm, 0), (arm, 2)) for arm in arms)
        if "reference" in arms:
            comparisons.extend((("reference", w), ("revealed", w)) for w in (0, 2))
        comparisons.append((("full", 2), ("revealed", 0)))
        for control, candidate in comparisons:
            baseline = costs[control]
            change = costs[candidate] - baseline
            paired.append(dict(seed=seed, control=control[0], control_width=control[1],
                               candidate=candidate[0], candidate_width=candidate[1], routes=len(change),
                               mean_change=change.mean(), pct_change=100 * change.mean() / baseline.mean(),
                               wins=int((change < -1e-5).sum()), ties=int((abs(change) <= 1e-5).sum()),
                               losses=int((change > 1e-5).sum())))
    write_csv(root / "metrics.csv", records)
    write_csv(root / "paired.csv", paired)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_directory", type=Path)
    parser.add_argument("--without-reference", action="store_true",
                        help="Summarize the six from-scratch evaluation tasks alone.")
    args = parser.parse_args()
    summarize(args.evaluation_directory, arms=ARMS[:2] if args.without_reference else ARMS)
