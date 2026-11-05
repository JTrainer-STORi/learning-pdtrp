#!/usr/bin/env python
"""Generate Larsen's chapter 5 PDTRP scenarios with variable customer counts."""

import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from evaluate_model_free import nearest_neighbour
from problems import PDTRP
from problems.larsen_generation import DYNAMISM_LEVELS, SCENARIOS, generate_balanced_instances
from problems.size_conventions import CUSTOMER_SIZE_CONVENTION
from utils.data_utils import dataset_to_input, format_dataset_row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="A")
    parser.add_argument("--instances-per-dod", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.instances_per_dod < 1 or args.max_candidates < 1:
        parser.error("instances-per-dod and max-candidates must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    return args


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)), "max": float(values.max())}


def plot_overview(groups, rows, output, scenario):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    levels = list(DYNAMISM_LEVELS)
    sizes = [[item["customers"] for item in groups[level]] for level in levels]
    axes[0, 0].hist([row["customers"] for row in rows],
                    bins=np.arange(min(row["customers"] for row in rows) - 0.5,
                                   max(row["customers"] for row in rows) + 1.5),
                    color="#27849b", edgecolor="white")
    axes[0, 0].axvline(SCENARIOS[scenario], color="#c05a32", linestyle="--", label="Nominal expected count")
    axes[0, 0].set(xlabel="Realized customers (depot excluded)", ylabel="Instances")
    axes[0, 0].legend()
    axes[0, 1].errorbar(levels, [np.mean(x) for x in sizes],
                       yerr=[np.std(x) for x in sizes], fmt="o-", capsize=3, color="#27849b")
    axes[0, 1].set(xlabel="Exact dynamism (%)", ylabel="Customer count: mean ± SD", xticks=range(0, 101, 10))
    calls = np.concatenate([item["sample"]["arrival_times"][item["sample"]["arrival_times"] > 0]
                            for group in groups.values() for item in group])
    axes[1, 0].hist(calls, bins=np.linspace(0, 480, 25), color="#27849b", edgecolor="white")
    axes[1, 0].set(xlabel="Dynamic request time (minutes)", ylabel="Requests")
    example = groups[50][0]
    points = example["sample"]["all_nodes"]
    dynamic = example["sample"]["arrival_times"] > 0
    advance = ~dynamic
    advance[0] = False
    axes[1, 1].scatter(*points[advance].T, label="Advance", color="#27849b")
    scatter = axes[1, 1].scatter(*points[dynamic].T, c=example["sample"]["arrival_times"][dynamic],
                                cmap="plasma", vmin=0, vmax=480, marker="^")
    axes[1, 1].scatter(*points[0], marker="*", s=150, color="black", label="Depot")
    axes[1, 1].set(xlabel="Normalized x", ylabel="Normalized y", xlim=(0, 1), ylim=(0, 1),
                   title=f"Example at 50% dynamism: {example['customers']} customers")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].legend(loc="upper left", fontsize="small")
    fig.colorbar(scatter, ax=axes[1, 1], label="Dynamic call time (minutes)")
    fig.suptitle(f"Larsen Scenario {scenario}: μ={SCENARIOS[scenario]}, T=8 h, "
                 f"{len(rows)} instances; exact dynamism quotas")
    fig.savefig(output / "overview.png", dpi=180)
    fig.savefig(output / "overview.pdf")
    plt.close(fig)


def generate(args):
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    expected = SCENARIOS[args.scenario]
    groups, selection = generate_balanced_instances(
        expected, args.instances_per_dod, args.seed, args.max_candidates
    )
    rows, datasets, group_metadata = [], {}, {}
    for level, group in groups.items():
        lines = []
        for group_index, instance in enumerate(group):
            sample = instance["sample"]
            tensor_sample = {key: torch.as_tensor(value, dtype=torch.bool if key == "graph" else torch.float32)
                             for key, value in sample.items()}
            actual = dataset_to_input(tensor_sample, "pdtrp", "cpu")
            tour, times = nearest_neighbour(PDTRP, actual)
            distance = float(PDTRP.get_costs(actual, tour)[0].item())
            service = float(sample["service_times"].sum())
            finish = float(times[0, -1].item())
            travel = distance / (4.0 / 60.0)
            row = {key: value for key, value in instance.items() if key != "sample"}
            row.update(sample=len(rows), group_sample=group_index, nodes=instance["customers"] + 1,
                       reference_distance=distance, reference_completion_minutes=finish,
                       reference_service_minutes=service, reference_travel_minutes=travel,
                       reference_waiting_minutes=max(0.0, finish - service - travel))
            rows.append(row)
            lines.append(format_dataset_row(sample, tour, times))
        filename = f"dod-{level:03d}.txt"
        datasets[filename] = "".join(lines)
        group_metadata[str(level)] = {
            "dataset": filename, "samples": len(group),
            "customers": describe([instance["customers"] for instance in group]),
            "candidate_indices": [instance["candidate_index"] for instance in group],
        }
    datasets["instances.txt"] = "".join(datasets.values())
    source_root = Path(__file__).resolve().parent
    source_files = ["generate_larsen_data.py", "problems/larsen_generation.py", "problems/utils.py",
                    "problems/size_conventions.py",
                    "utils/data_utils.py", "problems/pdtrp/problem_pdtrp.py",
                    "problems/pdtrp/state_pdtrp.py", "problems/request_events.py", "evaluate_model_free.py"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root,
                              capture_output=True, text=True).stdout.strip() or None
    metadata = {
        "schema_version": 1, "size_convention": CUSTOMER_SIZE_CONVENTION,
        "problem": "pdtrp", "scenario": args.scenario,
        "nominal_expected_customers": expected, "counts_exclude_depot": True,
        "instances_per_dod": args.instances_per_dod, "num_samples": len(rows), "seed": args.seed,
        "source": "imm143.pdf, chapter 5, section 5.2.1, printed pp. 69–70",
        "advance_count_convention": "Discrete uniform on integers 0 through mu inclusive",
        "dynamic_count_distribution": "Poisson(mu - advance_customers)",
        "call_time_distribution": "Uniform on (0, 480] minutes conditional on dynamic count",
        "selection": "Exact rational dynamism on 0,5,...,100 percent; balanced quotas; empty instances excluded",
        "selection_statistics": selection,
        "rounding_note": "Integer advance-count endpoints and exact percentage selection are explicit interpretations; original code rounding is not specified in the thesis.",
        "coordinates": "Continuous uniform unit square; depot (0.5,0.5); intended square width 10 km",
        "speed_units_per_hour": 4, "time_horizon_hours": 8,
        "service_time_mean_minutes": 3, "service_time_variance_minutes_squared": 5,
        "reference_method": "nearest_neighbour", "reference_is_optimal": False,
        "reference_completion_definition": "Final return after serving the generated requests; no forced wait until the call horizon",
        "msa_note": "Current fixed-count MSA sampling has not been adapted to this Poisson prior; realized counts are evaluation metadata, not online observations.",
        "realized_customers": describe([row["customers"] for row in rows]),
        "realized_customer_count_frequencies": dict(sorted(Counter(row["customers"] for row in rows).items())),
        "reference_completion_minutes": describe([row["reference_completion_minutes"] for row in rows]),
        "groups": group_metadata, "git_revision": revision,
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                          for name in source_files},
        "dataset_sha256": {name: hashlib.sha256(payload.encode()).hexdigest() for name, payload in datasets.items()},
    }
    output.mkdir(parents=True)
    for name, payload in datasets.items():
        (output / name).write_text(payload)
    with (output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_overview(groups, rows, output, args.scenario)
    return {"output_dir": str(output), "samples": len(rows), "candidates": selection["candidates"],
            "customers": metadata["realized_customers"], "preview": str(output / "overview.png")}


def main(argv=None):
    args = parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        result = generate(args)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
