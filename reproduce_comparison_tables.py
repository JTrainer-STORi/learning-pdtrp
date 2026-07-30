#!/usr/bin/env python3
"""Combine corrected model-free result jobs into CSV and LaTeX tables."""

import argparse
import csv
import json
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
                }
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
    methods = (
        "RL 20--50 (pending)",
        "RL 20--100 (pending)",
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
            for method_index, method in enumerate(methods):
                prefix = (
                    rf"\multirow{{{len(methods)}}}{{*}}{{\rotatebox{{90}}{{{PROBLEM_LABELS[problem]}}}}}"
                    if method_index == 0
                    else ""
                )
                cells = []
                if method.startswith("RL "):
                    cells = [r"--", r"--", r"--"] * len(DODS)
                else:
                    for dod in DODS:
                        row = lookup[problem, size, dod, method]
                        cells.extend(
                            (
                                objective(row["objective"]),
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
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    tables = [
        latex_table(
            rows,
            "uniform",
            "Partial reproduction on uniform--uniform test sets. Learned-policy results are pending. "
            r"Gaps are relative to clairvoyant OR-Tools; times are per instance.",
            "tab:recomputed-uniform-partial",
        ),
        latex_table(
            rows,
            "real-world",
            "Partial reproduction on real-world test sets. Learned-policy results are pending. "
            r"Gaps are relative to clairvoyant OR-Tools; times are per instance.",
            "tab:recomputed-real-world-partial",
        ),
    ]
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n\n".join(tables) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote 2 tables to {args.output_tex}")


if __name__ == "__main__":
    main()
