#!/usr/bin/env python3
"""Evaluate one trained policy checkpoint on one or more fixed datasets."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import set_decode_type
from utils import load_model
from utils.data_utils import dataset_to_input


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-seed", type=int, default=1234)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--time-horizon", type=int, default=8)
    parser.add_argument("--latest-end", type=int, default=2)
    parser.add_argument(
        "--use-ortec",
        help="ORTEC source instance; used when loading *_testing_ortec_* files",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


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


def evaluate_dataset(model, problem, dataset_path, device, args):
    is_ortec = "_testing_ortec_" in dataset_path.name
    if is_ortec and not args.use_ortec:
        raise ValueError(f"--use-ortec is required for ORTEC dataset {dataset_path}")
    dataset = problem.make_dataset(
        filename=str(dataset_path),
        batch_size=1,
        speed=args.speed,
        time_horizon=args.time_horizon,
        latest_end=args.latest_end,
        use_ortec=args.use_ortec if is_ortec else None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rows = []

    with torch.inference_mode():
        for sample_index, batch in enumerate(loader):
            model_input = dataset_to_input(batch, problem.NAME, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            cost, _, _ = model(model_input, mode="recursive")
            if device.type == "cuda":
                torch.cuda.synchronize()
            rows.append(
                {
                    "sample": sample_index,
                    "objective": float(cost.reshape(-1)[0].cpu()),
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
    return rows


def main():
    args = parse_args()
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
        "ortec_source": args.use_ortec,
        "training_seed": model_args.get("seed"),
        "datasets": [],
    }

    for dataset_path in datasets:
        print(f"Evaluating {dataset_path}", flush=True)
        rows = evaluate_dataset(model, problem, dataset_path, device, args)
        stem = dataset_path.stem
        samples_path = output_dir / f"{stem}.samples.csv"
        summary_path = output_dir / f"{stem}.summary.json"

        with samples_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample", "objective", "runtime_seconds")
            )
            writer.writeheader()
            writer.writerows(rows)

        objectives = np.asarray([row["objective"] for row in rows], dtype=float)
        runtimes = np.asarray([row["runtime_seconds"] for row in rows], dtype=float)
        summary = {
            "checkpoint": str(checkpoint),
            "problem": args.problem,
            "dataset": str(dataset_path),
            "samples": len(rows),
            "mean_objective": float(objectives.mean()),
            "std_objective": float(objectives.std()),
            "mean_runtime_seconds": float(runtimes.mean()),
            "std_runtime_seconds": float(runtimes.std()),
        }
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        manifest["datasets"].append(summary)

    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"POLICY EVALUATION PASSED: {len(datasets)} datasets", flush=True)


if __name__ == "__main__":
    main()
