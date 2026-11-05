"""Arguments and result files for fixed per-solve time-budget sweeps."""

import copy
import csv
from decimal import Decimal
import json
from pathlib import Path


def add_solver_budget_arguments(parser):
    parser.add_argument(
        "--solve-seconds", type=float, nargs="+", metavar="SECONDS",
        help=(
            "Evaluate each listed search budget in seconds per solve, with separate "
            "outputs. Disables no-improvement stopping; cannot be combined with "
            "the millisecond time-limit or patience options. Supports whole milliseconds."
        ),
    )


def seconds_to_milliseconds(seconds):
    milliseconds = Decimal(str(seconds)) * 1000
    if (not milliseconds.is_finite() or milliseconds < 1
            or milliseconds > 2**63 - 1
            or milliseconds != milliseconds.to_integral_value()):
        raise ValueError(
            "solve-seconds values must be finite, positive whole milliseconds "
            "(at least 0.001 seconds) within the solver's integer range"
        )
    return int(milliseconds)


def validate_solver_budget_arguments(parser, args, prefixes):
    """Retain legacy defaults, or select unambiguous time-only sweep stopping."""
    budgets = args.solve_seconds
    if budgets is not None:
        try:
            milliseconds = [seconds_to_milliseconds(value) for value in budgets]
        except ValueError as error:
            parser.error(str(error))
        if len(set(milliseconds)) != len(milliseconds):
            parser.error("solve-seconds values must be unique")
        for prefix in prefixes:
            for suffix in ("time_ms", "no_improvement_ms"):
                name = f"{prefix}_{suffix}"
                if getattr(args, name) is not None:
                    parser.error(
                        f"--solve-seconds cannot be combined with --{name.replace('_', '-')}"
                    )
        return

    for prefix in prefixes:
        patience_name = f"{prefix}_no_improvement_ms"
        if getattr(args, patience_name) is None:
            setattr(args, patience_name, 60000)
        cap = getattr(args, f"{prefix}_time_ms")
        if getattr(args, patience_name) < 1 or (cap is not None and cap < 1):
            parser.error("time limits and no-improvement intervals must be positive")


def budget_label(milliseconds):
    seconds, remainder = divmod(milliseconds, 1000)
    value = str(seconds) if not remainder else f"{seconds}.{remainder:03d}".rstrip("0")
    return f"solve-{value}s"


def run_solver_budget_sweep(args, evaluate, prefixes, *, file_output=False):
    """Run budgets sequentially, keeping each result and a comparison summary."""
    output = Path(args.output)
    if file_output:
        manifest_path = output.with_name(f"{output.stem}.sweep.json")
        csv_path = output.with_name(f"{output.stem}.budget-summary.csv")
    else:
        manifest_path = output / "sweep.json"
        csv_path = output / "budget-summary.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"solve_seconds": args.solve_seconds, "runs": []}
    comparison_rows = []
    for seconds in args.solve_seconds:
        milliseconds = seconds_to_milliseconds(seconds)
        label = budget_label(milliseconds)
        run_args = copy.copy(args)
        run_args.solve_seconds = None
        run_args.solve_budget_seconds = seconds
        run_args.output = str(
            output.with_name(f"{output.stem}.{label}{output.suffix or '.json'}")
            if file_output else output / label
        )
        for prefix in prefixes:
            setattr(run_args, f"{prefix}_time_ms", milliseconds)
            setattr(run_args, f"{prefix}_no_improvement_ms", None)
        print(f"Evaluating {seconds:g} seconds per solve -> {run_args.output}", flush=True)
        result = evaluate(run_args)
        summary = {key: value for key, value in result.items() if not isinstance(value, list)}
        if isinstance(result.get("samples"), list):
            summary["samples"] = len(result["samples"])
        manifest["runs"].append({
            "solve_seconds": seconds, "output": run_args.output, "summary": summary,
        })
        comparison_rows.append({
            "solve_seconds": seconds, "output": run_args.output,
            **{key: value for key, value in summary.items()
               if value is None or isinstance(value, (str, int, float, bool))},
        })
        # Preserve completed budgets even if a later run reaches Slurm walltime.
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)
    return manifest
