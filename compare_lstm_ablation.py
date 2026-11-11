#!/usr/bin/env python3
"""Compare matched LSTM-off/on policy evaluations for one training seed."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def compare_runs(without_lstm, with_lstm, output):
    roots = [Path(without_lstm), Path(with_lstm)]
    manifests = [json.loads((root / "manifest.json").read_text()) for root in roots]
    args = [json.loads((root / "training-args.json").read_text()) for root in roots]
    if args[0]["use_arrival_lstm"] or not args[1]["use_arrival_lstm"]:
        raise ValueError("Expected LSTM disabled in --without-lstm and enabled in --with-lstm")
    ignored = {"use_arrival_lstm", "run_name", "save_dir", "output_dir", "log_dir"}
    different = [key for key in set(args[0]) | set(args[1])
                 if key not in ignored and args[0].get(key) != args[1].get(key)]
    if different:
        raise ValueError(f"Training settings differ: {sorted(different)}")
    if args[0].get("training_data_seed") is None:
        raise ValueError("Matched ablation requires --training-data-seed")
    for key in ("problem", "dataset_parameters", "evaluation_seed", "offset", "num_samples_requested"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Evaluation settings differ: {key}")
    histories = [read_csv(root / "training-history.csv") for root in roots]
    epochs = [[(row["epoch"], row["epoch_data_seed"]) for row in history] for history in histories]
    if not epochs[0] or epochs[0] != epochs[1] or any(not seed for _, seed in epochs[0]):
        raise ValueError("Training epochs and per-epoch data seeds must match")
    if len(histories[0]) != args[0]["n_epochs"]:
        raise ValueError("Training history does not cover the configured epoch budget")
    for manifest in manifests:
        if Path(manifest["checkpoint"]).stem != f"epoch-{histories[0][-1]['epoch']}":
            raise ValueError("Evaluation must use the final checkpoint from the matched training budget")
    datasets = [{row["dataset"]: row for row in manifest["datasets"]} for manifest in manifests]
    if (not datasets[0] or datasets[0].keys() != datasets[1].keys()
            or any(len(index) != len(manifest["datasets"]) for index, manifest in zip(datasets, manifests))):
        raise ValueError("Evaluation datasets differ")

    paired, summaries = [], []
    for dataset in datasets[0]:
        stem = Path(dataset).stem
        samples = [read_csv(root / f"{stem}.samples.csv") for root in roots]
        indexed = [{int(row["sample"]): row for row in rows} for rows in samples]
        if (not indexed[0] or indexed[0].keys() != indexed[1].keys()
                or any(len(rows) != len(index) for rows, index in zip(samples, indexed))):
            raise ValueError(f"Sample indices differ or repeat in {stem}")
        group = []
        for sample in sorted(indexed[0]):
            values = [float(index[sample]["objective"]) for index in indexed]
            runtimes = [float(index[sample]["runtime_seconds"]) for index in indexed]
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"Invalid objectives in {stem}, sample {sample}")
            if not all(math.isfinite(v) and v >= 0 for v in runtimes):
                raise ValueError(f"Invalid runtimes in {stem}, sample {sample}")
            row = dict(dataset=dataset, sample=sample, training_seed=args[0]["seed"],
                       objective_without=values[0], objective_with=values[1],
                       change=values[1] - values[0],
                       runtime_without=runtimes[0], runtime_with=runtimes[1])
            group.append(row)
            paired.append(row)
        for manifest_rows in datasets:
            if manifest_rows[dataset]["samples"] != len(group):
                raise ValueError(f"Summary/sample count mismatch in {stem}")
        off, on = mean(r["objective_without"] for r in group), mean(r["objective_with"] for r in group)
        summaries.append(dict(dataset=dataset, samples=len(group), mean_without=off, mean_with=on,
                              change_percent=100 * (on / off - 1) if off else None,
                              lstm_wins=sum(r["change"] < -1e-6 for r in group),
                              ties=sum(abs(r["change"]) <= 1e-6 for r in group),
                              lstm_losses=sum(r["change"] > 1e-6 for r in group),
                              mean_runtime_without=mean(r["runtime_without"] for r in group),
                              mean_runtime_with=mean(r["runtime_with"] for r in group)))
    profiles = []
    for root, history in zip(roots, histories):
        profile_path = root / "profile.json"
        profile = json.loads(profile_path.read_text()) if profile_path.is_file() else {}
        # Older histories contain only total epoch times. Preserve compatibility
        # while exposing measured phase totals for the optimized training runs.
        phase_times = {}
        for name in ("data_generation_seconds", "baseline_precompute_seconds", "optimization_seconds",
                     "checkpoint_seconds", "validation_seconds", "baseline_update_seconds"):
            if all(row.get(name) not in (None, "") for row in history):
                phase_times[f"recorded_{name}"] = sum(float(row[name]) for row in history)
        profiles.append({**profile, "recorded_epoch_seconds": sum(float(r["epoch_seconds"]) for r in history),
                         "peak_gpu_allocated_bytes": max(int(r["peak_gpu_allocated_bytes"]) for r in history),
                         **phase_times})
    result = dict(problem=args[0]["problem"], training_seed=args[0]["seed"],
                  training_data_seed=args[0]["training_data_seed"], epochs=len(histories[0]),
                  max_customers=args[0]["max_customers"], profiles=dict(without=profiles[0], with_lstm=profiles[1]),
                  datasets=summaries)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-lstm", type=Path, required=True)
    parser.add_argument("--with-lstm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(compare_runs(args.without_lstm, args.with_lstm, args.output), indent=2))


if __name__ == "__main__":
    main()
