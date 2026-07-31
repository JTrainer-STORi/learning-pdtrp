#!/usr/bin/env python3
"""Combine corrected model-free result jobs into CSV and LaTeX tables."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PROBLEMS = ("pdtrp", "pdctrp", "pdtrptw", "pdctrptw")
DISTRIBUTIONS = ("uniform", "skew-spatio-temporal", "real-world")
SIZES = (50, 100)
DODS = ("0.2", "0.5", "0.8")
PROBLEM_LABELS = {
    "pdtrp": "PDTRP",
    "pdctrp": "PDCTRP",
    "pdtrptw": r"PDTRP-TW",
    "pdctrptw": r"PDCTRP-TW",
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--nearest-job", default="23268938")
    parser.add_argument("--ortools-job", default="23268937")
    parser.add_argument("--msa-jobs", nargs="+", default=("23268787", "23270643"))
    parser.add_argument("--policy-job", default="23282643")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    return parser.parse_args()


def configuration_paths(root, job):
    return list((root / job / "results").rglob("summary.json"))


def path_configuration(path):
    parts = path.parts
    results_index = parts.index("results")
    problem, size, dod_part, distribution = parts[results_index + 1 : results_index + 5]
    return problem, int(size), dod_part.removeprefix("dod-"), distribution


def load_results(args):
    records = {}
    ortools_root = args.results_root / "ortools-corrected"
    nearest_root = args.results_root / "model-free"

    for path in configuration_paths(nearest_root, args.nearest_job):
        key = path_configuration(path)
        data = json.loads(path.read_text())
        records[key, "Nearest neighbour"] = (
            data["objective_mean"],
            data["runtime_mean_seconds"],
            100,
        )

    static_means = {}
    for path in configuration_paths(ortools_root, args.ortools_job):
        key = path_configuration(path)
        data = json.loads(path.read_text())
        static_means[key] = data["static_objective_mean"]
        records[key, "Online OR-Tools"] = (
            data["online_objective_mean"],
            data["online_runtime_mean_seconds"],
            100,
        )
        records[key, "Clairvoyant OR-Tools"] = (
            data["static_objective_mean"],
            data["static_runtime_mean_seconds"],
            100,
        )

    msa_by_key = defaultdict(list)
    for job in args.msa_jobs:
        for path in (args.results_root / "msa" / job / "results").rglob("seed-*.json"):
            data = json.loads(path.read_text())
            parts = path.parts
            results_index = parts.index("results")
            key = (
                data["problem"],
                int(parts[results_index + 2]),
                parts[results_index + 3].removeprefix("dod-"),
                parts[results_index + 4],
            )
            msa_by_key[key].append(data)

    for key, seed_results in msa_by_key.items():
        objectives = [
            sample["objective"]
            for result in seed_results
            for sample in result["samples"]
        ]
        runtimes = [
            sample["runtime"]
            for result in seed_results
            for sample in result["samples"]
        ]
        records[key, "MSA OR-Tools"] = (
            float(np.mean(objectives)),
            float(np.mean(runtimes)),
            len(objectives),
        )

    policy_by_key = defaultdict(list)
    policy_root = args.results_root / "policy-evaluation" / args.policy_job
    for manifest_path in policy_root.glob("task-*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        checkpoint_name = Path(manifest["checkpoint"]).parent.name
        match = re.search(
            r"corrected-(uniform|real-world|joshi)-"
            r"(pdtrp|pdctrp|pdtrptw|pdctrptw)-20-(50|100)-seed-(\d+)_",
            checkpoint_name,
        )
        if not match:
            raise RuntimeError(f"Cannot identify policy checkpoint: {checkpoint_name}")
        training_distribution, problem, training_max, training_seed = match.groups()
        for result in manifest["datasets"]:
            dataset_name = Path(result["dataset"]).name
            dataset_match = re.match(
                r".*?(50|100)dod(0\.2|0\.5|0\.8)v4th8_testing"
                r"(?:(?:_skew-st)|(?:_ortec))?_ortools\.txt",
                dataset_name,
            )
            if not dataset_match:
                raise RuntimeError(f"Cannot identify policy dataset: {dataset_name}")
            size, dod = dataset_match.groups()
            distribution = (
                "real-world"
                if "_testing_ortec_" in dataset_name
                else "skew-spatio-temporal"
                if "_testing_skew-st_" in dataset_name
                else "uniform"
            )
            policy_by_key[
                (
                    problem,
                    int(size),
                    dod,
                    distribution,
                    training_distribution,
                    int(training_max),
                )
            ].append(
                (
                    int(training_seed),
                    result["mean_objective"],
                    result["mean_runtime_seconds"],
                )
            )

    expected = {
        (problem, size, dod, distribution)
        for problem in PROBLEMS
        for size in SIZES
        for dod in DODS
        for distribution in DISTRIBUTIONS
    }
    for method in (
        "Nearest neighbour",
        "Online OR-Tools",
        "Clairvoyant OR-Tools",
        "MSA OR-Tools",
    ):
        missing = sorted(key for key in expected if (key, method) not in records)
        if missing:
            raise RuntimeError(f"{method} is missing {len(missing)} configurations: {missing}")

    rows = []
    for key in sorted(expected):
        baseline = static_means[key]
        for method in (
            "Nearest neighbour",
            "Online OR-Tools",
            "MSA OR-Tools",
            "Clairvoyant OR-Tools",
        ):
            objective, runtime, observations = records[key, method]
            rows.append(
                {
                    "problem": key[0],
                    "size": key[1],
                    "dod": key[2],
                    "distribution": key[3],
                    "method": method,
                    "objective": objective,
                    "gap_percent": 100.0 * (objective / baseline - 1.0),
                    "runtime_seconds": runtime,
                    "observations": observations,
                    "training_seeds": "",
                    "training_seed_std_objective": "",
                }
            )
    for key, seed_results in sorted(policy_by_key.items()):
        problem, size, dod, distribution, training_distribution, training_max = key
        unique_seeds = {item[0] for item in seed_results}
        if len(unique_seeds) != len(seed_results):
            raise RuntimeError(f"Duplicate policy seed results for {key}")
        objective = float(np.mean([item[1] for item in seed_results]))
        runtime_seconds = float(np.mean([item[2] for item in seed_results]))
        seed_std_objective = float(
            np.std([item[1] for item in seed_results], ddof=1)
        )
        baseline = static_means[problem, size, dod, distribution]
        rows.append(
            {
                "problem": problem,
                "size": size,
                "dod": dod,
                "distribution": distribution,
                "method": f"RL {training_distribution} 20--{training_max}",
                "objective": objective,
                "gap_percent": 100.0 * (objective / baseline - 1.0),
                "runtime_seconds": runtime_seconds,
                "observations": 100 * len(unique_seeds),
                "training_seeds": len(unique_seeds),
                "training_seed_std_objective": seed_std_objective,
            }
        )

    expected_policy_keys = set()
    for problem in PROBLEMS:
        for training_max in SIZES:
            for size in SIZES:
                for dod in DODS:
                    for distribution in DISTRIBUTIONS:
                        expected_policy_keys.add(
                            (
                                problem,
                                size,
                                dod,
                                distribution,
                                "uniform",
                                training_max,
                            )
                        )
        for size in SIZES:
            for dod in DODS:
                expected_policy_keys.add(
                    (problem, size, dod, "real-world", "real-world", 50)
                )
    for size in SIZES:
        for dod in DODS:
            expected_policy_keys.add(
                ("pdtrp", size, dod, "uniform", "joshi", 50)
            )
    missing_policy_keys = sorted(expected_policy_keys - set(policy_by_key))
    if missing_policy_keys:
        raise RuntimeError(
            f"Policy evaluation is missing {len(missing_policy_keys)} aggregates: "
            f"{missing_policy_keys}"
        )
    incomplete_policy_keys = sorted(
        key
        for key in expected_policy_keys
        if len({item[0] for item in policy_by_key[key]}) != 3
    )
    if incomplete_policy_keys:
        raise RuntimeError(
            f"Policy aggregates without three seeds: {incomplete_policy_keys}"
        )
    return rows


def objective(value):
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def gap(value):
    return f"{value:.1f}"


def runtime(value):
    return f"{value:.2f}s"


def latex_table(rows, distribution, caption, label):
    lookup = {
        (row["problem"], row["size"], row["dod"], row["method"]): row
        for row in rows
        if row["distribution"] == distribution
    }
    policy_methods = (
        ("RL 20--50", "RL uniform 20--50"),
        ("RL 20--100", "RL uniform 20--100"),
    )
    if distribution == "real-world":
        policy_methods = (
            ("RL U.U.", "RL uniform 20--50"),
            ("RL R.W.", "RL real-world 20--50"),
        )
    baseline_methods = (
        "Nearest neighbour",
        "Online OR-Tools",
        "MSA OR-Tools",
        "Clairvoyant OR-Tools",
    )
    lines = [
        r"\begin{table}[hpbt]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cl|rrl|rrl|rrl}",
        r"\hline",
    ]
    for size in SIZES:
        title = (
            "Uniform--Uniform"
            if distribution == "uniform"
            else "Real-world"
            if distribution == "real-world"
            else "Skew spatio-temporal"
        )
        lines.extend(
            [
                rf"\multicolumn{{2}}{{c|}}{{}} & \multicolumn{{9}}{{c}}{{{title} $N={size}$}} \\ \hline",
                r"\multicolumn{2}{c|}{} & \multicolumn{3}{c|}{$\mathrm{dod}=0.2$} & "
                r"\multicolumn{3}{c|}{$\mathrm{dod}=0.5$} & "
                r"\multicolumn{3}{c}{$\mathrm{dod}=0.8$} \\",
                r" & Method & Obj. & \%Gap & Time & Obj. & \%Gap & Time & Obj. & \%Gap & Time \\ \hline",
            ]
        )
        for problem in PROBLEMS:
            methods = list(policy_methods)
            if distribution == "uniform" and problem == "pdtrp":
                methods.append(("Joshi", "RL joshi 20--50"))
            methods.extend(baseline_methods)
            comparable_lookup_methods = [
                item[1] if isinstance(item, tuple) else item
                for item in methods
                if (item[1] if isinstance(item, tuple) else item)
                != "Clairvoyant OR-Tools"
            ]
            for method_index, method_spec in enumerate(methods):
                if isinstance(method_spec, tuple):
                    method, lookup_method = method_spec
                else:
                    method = lookup_method = method_spec
                prefix = (
                    rf"\multirow{{{len(methods)}}}{{*}}{{\rotatebox{{90}}{{{PROBLEM_LABELS[problem]}}}}}"
                    if method_index == 0
                    else ""
                )
                cells = []
                for dod in DODS:
                    row = lookup.get((problem, size, dod, lookup_method))
                    if row is None:
                        cells.extend((r"--", r"--", r"--"))
                    else:
                        best_objective = min(
                            float(lookup[problem, size, dod, candidate]["objective"])
                            for candidate in comparable_lookup_methods
                            if (problem, size, dod, candidate) in lookup
                        )
                        objective_text = objective(row["objective"])
                        seed_std = row.get("training_seed_std_objective")
                        if seed_std not in ("", None):
                            objective_text += (
                                r"\,{\scriptsize$\pm$"
                                + objective(float(seed_std))
                                + "}"
                            )
                        if (
                            lookup_method != "Clairvoyant OR-Tools"
                            and np.isclose(
                                float(row["objective"]),
                                best_objective,
                                rtol=1e-10,
                                atol=1e-10,
                            )
                        ):
                            objective_text = r"\textbf{" + objective_text + "}"
                        cells.extend(
                            (
                                objective_text,
                                gap(row["gap_percent"]),
                                runtime(row["runtime_seconds"]),
                            )
                        )
                end = r" \\ \hline" if method_index == len(methods) - 1 else r" \\"
                lines.append(
                    f"{prefix} & {method} & " + " & ".join(cells) + end
                )
    lines.extend((r"\end{tabular}%", r"}", r"\end{table}"))
    return "\n".join(lines)


def main():
    args = arguments()
    rows = load_results(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    tables = [
        latex_table(
            rows,
            "uniform",
            "Recomputed results on uniform--uniform test sets. Learned-policy results are "
            r"reported as mean objective $\pm$ standard deviation over three training seeds. "
            r"Bold denotes the best non-clairvoyant mean objective. Gaps are relative to "
            r"clairvoyant OR-Tools; times are per instance.",
            "tab:recomputed-uniform",
        ),
        latex_table(
            rows,
            "real-world",
            "Recomputed results on real-world test sets. Learned-policy results are averaged "
            r"and reported as mean objective $\pm$ standard deviation over three training seeds. "
            r"Bold denotes the best non-clairvoyant mean objective. Gaps are relative to "
            r"clairvoyant OR-Tools; times are per instance.",
            "tab:recomputed-real-world",
        ),
        latex_table(
            rows,
            "skew-spatio-temporal",
            "Recomputed results on skew spatio-temporal test sets. Learned-policy results are "
            r"reported as mean objective $\pm$ standard deviation over three training seeds. "
            r"Bold denotes the best non-clairvoyant mean objective. Gaps are relative to "
            r"clairvoyant OR-Tools; times are per instance.",
            "tab:recomputed-skew-spatio-temporal",
        ),
    ]
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n\n".join(tables) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote 3 tables to {args.output_tex}")


if __name__ == "__main__":
    main()
