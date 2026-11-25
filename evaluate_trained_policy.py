#!/usr/bin/env python3
"""Evaluate one trained policy checkpoint on one or more fixed datasets."""

import argparse
from contextlib import redirect_stdout
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import set_decode_type
from utils import load_model
from utils.data_utils import dataset_to_input
from utils.dataset_parameters import add_dataset_time_arguments, dataset_time_options, validate_dataset_time_arguments
from utils.service_times import add_service_time_arguments, validate_service_time_arguments


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--num-samples", type=int, default=12800,
        help="Maximum instances to evaluate from each dataset after offset (default: 12800)",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="First zero-based dataset instance to evaluate (default: 0)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-seed", type=int, default=1234)
    add_dataset_time_arguments(parser)
    add_service_time_arguments(parser)
    parser.add_argument(
        "--use-ortec",
        help="ORTEC source instance; used when loading *_testing_ortec_* files",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--rollout-candidates", type=int, default=0,
        help="Roll out this many highest-ranked feasible customers plus the legal depot action. Zero uses greedy.",
    )
    parser.add_argument(
        "--rollout-wait-penalty", type=float, default=0.0,
        help="Ranking cost per minute of avoidable pre-departure waiting for a time-window rollout's first action.",
    )
    parser.add_argument(
        "--record-decision-times", action="store_true",
        help="Record time between action selections and include objective components in the samples CSV.",
    )
    args = parser.parse_args(argv)
    validate_dataset_time_arguments(parser, args)
    validate_service_time_arguments(parser, args)
    if args.num_samples < 1:
        parser.error("num-samples must be positive")
    if args.offset < 0:
        parser.error("offset must be nonnegative")
    if args.rollout_candidates < 0:
        parser.error("rollout-candidates must be nonnegative")
    if not math.isfinite(args.rollout_wait_penalty) or args.rollout_wait_penalty < 0:
        parser.error("rollout-wait-penalty must be finite and nonnegative")
    if args.rollout_wait_penalty and (not args.rollout_candidates or args.problem not in ("pdtrptw", "pdctrptw")):
        parser.error("rollout-wait-penalty requires rollout candidates and a time-window problem")
    return args


def problem_class(name):
    from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW

    classes = {
        "pdtrp": PDTRP,
        "pdctrp": PDCTRP,
        "pdtrptw": PDTRPTW,
        "pdctrptw": PDCTRPTW,
    }
    try:
        return classes[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported problem: {name}") from exc


def evaluate_dataset(model, problem, dataset_path, device, args, decision_records=None):
    is_ortec = "_testing_ortec_" in dataset_path.name
    if is_ortec and not args.use_ortec:
        raise ValueError(f"--use-ortec is required for ORTEC dataset {dataset_path}")
    dataset = problem.make_dataset(
        filename=str(dataset_path),
        batch_size=1,
        num_samples=args.num_samples,
        offset=args.offset,
        speed=args.speed,
        time_horizon=args.time_horizon,
        latest_end=args.latest_end,
        service_times_mean=args.service_times_mean,
        service_times_var=args.service_times_var,
        use_ortec=args.use_ortec if is_ortec else None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rows = []

    with torch.inference_mode():
        for sample_index, batch in enumerate(loader):
            model_input = dataset_to_input(batch, problem.NAME, device)
            sample = args.offset + sample_index
            options = {"mode": "recursive"}
            if args.rollout_candidates:
                options["rollout_candidates"] = args.rollout_candidates
            if args.rollout_wait_penalty:
                options["rollout_wait_penalty"] = args.rollout_wait_penalty
            if decision_records is not None:
                decision_times = []
                options["decision_times"] = decision_times
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            cost, _, info = model(model_input, **options)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            row = {"sample": sample, "objective": cost.reshape(-1)[0].item(), "runtime_seconds": elapsed}
            if decision_records is not None:
                decision_records.extend((sample, decision, seconds) for decision, seconds in enumerate(decision_times))
                row.update(
                    distance_penalty=info.get("distance_penalty", cost).item(),
                    lateness_penalty=info["lateness_penalty"].item() if "lateness_penalty" in info else 0.0,
                    decisions=len(decision_times),
                )
            rows.append(row)
    return rows


def main(argv=None):
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    datasets = [Path(path).resolve() for path in args.datasets]

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    missing = [str(path) for path in datasets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset files do not exist: {missing}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    np.random.seed(args.evaluation_seed)
    torch.manual_seed(args.evaluation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.evaluation_seed)

    device = torch.device(args.device)
    problem = problem_class(args.problem)()
    with redirect_stdout(sys.stderr):
        model, model_args = load_model(str(checkpoint))
    model.to(device)
    model.eval()
    set_decode_type(model, "greedy")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint": str(checkpoint),
        "problem": args.problem,
        "evaluation_seed": args.evaluation_seed,
        "device": str(device),
        "dataset_parameters": dataset_time_options(args),
        "ortec_source": args.use_ortec,
        "training_seed": model_args.get("seed"),
        "rollout_candidates": args.rollout_candidates,
        "rollout_wait_penalty": args.rollout_wait_penalty,
        "service_times_mean": args.service_times_mean,
        "service_times_var": args.service_times_var,
        "record_decision_times": args.record_decision_times,
        "num_samples_requested": args.num_samples,
        "offset": args.offset,
        "datasets": [],
    }
    if device.type == "cuda":
        manifest["gpu"] = torch.cuda.get_device_name(device)

    for dataset_path in datasets:
        print(f"Evaluating {dataset_path}", file=sys.stderr, flush=True)
        decision_records = [] if args.record_decision_times else None
        with redirect_stdout(sys.stderr):
            rows = evaluate_dataset(model, problem, dataset_path, device, args, decision_records)
        stem = dataset_path.stem
        samples_path = output_dir / f"{stem}.samples.csv"
        summary_path = output_dir / f"{stem}.summary.json"

        with samples_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        objectives = np.asarray([row["objective"] for row in rows], dtype=float)
        runtimes = np.asarray([row["runtime_seconds"] for row in rows], dtype=float)
        summary = {
            "checkpoint": str(checkpoint),
            "problem": args.problem,
            "dataset": str(dataset_path),
            "samples": len(rows),
            "num_samples_requested": args.num_samples,
            "offset": args.offset,
            "rollout_candidates": args.rollout_candidates,
            "rollout_wait_penalty": args.rollout_wait_penalty,
            "mean_objective": float(objectives.mean()),
            "std_objective": float(objectives.std()),
            "mean_runtime_seconds": float(runtimes.mean()),
            "std_runtime_seconds": float(runtimes.std()),
        }
        if decision_records is not None:
            decisions_path = output_dir / f"{stem}.decisions.csv"
            with decisions_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("sample", "decision", "runtime_seconds"))
                writer.writerows(decision_records)
            decision_seconds = [seconds for _, _, seconds in decision_records]
            lateness = [row["lateness_penalty"] for row in rows]
            summary.update(
                decisions=len(decision_records),
                mean_decision_seconds=float(np.mean(decision_seconds)),
                p95_decision_seconds=float(np.quantile(decision_seconds, 0.95)),
                mean_distance_penalty=float(np.mean([row["distance_penalty"] for row in rows])),
                mean_lateness_penalty=float(np.mean(lateness)),
                max_lateness_penalty=float(np.max(lateness)),
                p95_objective=float(np.quantile(objectives, 0.95)),
                max_objective=float(objectives.max()),
            )
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        manifest["datasets"].append(summary)

    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


if __name__ == "__main__":
    main()
