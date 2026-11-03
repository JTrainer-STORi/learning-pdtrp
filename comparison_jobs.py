"""Prepare and run independent comparison instances as bounded Slurm arrays.

Planning, submission and collection use only the standard library. Evaluators
run in the selected Python environment on compute nodes, one instance per task.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


PROJECT_DIR = Path(__file__).resolve().parent
METHODS = ("online", "oracle", "msa")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    """Replace a JSON file only after all its contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_digest():
    paths = [PROJECT_DIR / name for name in (
        "comparison_jobs.py", "evaluate_ortools_baselines.py", "evaluate_msa_ortools.py",
        "evaluate_online_ortools.py", "evaluate_clairvoyant_ortools.py",
        "requirements.txt", "scripts/slurm-evaluate.sh",
    )]
    paths += list((PROJECT_DIR / "utils").glob("*.py"))
    paths += list((PROJECT_DIR / "problems").glob("*.py"))
    for problem in ("pdtrp", "pdctrp", "pdtrp_tw", "pdctrp_tw"):
        paths += list((PROJECT_DIR / "problems" / problem).glob("*.py"))
    checksum = hashlib.sha256()
    for path in sorted(paths):
        checksum.update(str(path.relative_to(PROJECT_DIR)).encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def file_identity(path, *, count_lines=False):
    path = Path(path).resolve(strict=True)
    checksum = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for line in handle:
            checksum.update(line)
            lines += 1
            if count_lines and not line.strip():
                raise ValueError(f"Dataset contains a blank instance at line {lines}: {path}")
    stat = path.stat()
    identity = {"path": str(path), "sha256": checksum.hexdigest(), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}
    if count_lines:
        identity["line_count"] = lines
    return identity


def build_plan(args):
    cases_path = Path(args.cases).resolve(strict=True)
    cases = json.loads(cases_path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty JSON list")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("methods and seeds must not contain duplicates")
    if args.offset < 0 or any(seed < 0 for seed in args.seeds):
        raise ValueError("offset and seeds must be nonnegative")
    for name in ("num_samples", "scenarios", "msa_workers", "total_cpus", "total_memory_gib",
                 "memory_per_cpu_mib", "max_array_size", "no_improvement_ms"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.time_limit_ms is not None and args.time_limit_ms < 1:
        raise ValueError("time-limit-ms must be positive when supplied")
    python = shutil.which(args.python)
    if python is None:
        raise ValueError(f"Python interpreter not found: {args.python}")
    # Keep the venv executable path: resolving its symlink would select the base
    # interpreter and discard the environment's installed dependencies.
    python = str(Path(python).absolute())
    output = Path(args.output).resolve()
    code_digest = source_digest()
    workers = min(args.msa_workers, args.scenarios)
    identities = {}
    tasks = []
    seen_cases = set()
    for case in cases:
        if not isinstance(case, dict) or case.get("problem") not in ("pdtrp", "pdctrp", "pdtrptw", "pdctrptw"):
            raise ValueError("Each case needs a supported problem and dataset")
        if not isinstance(case.get("dataset"), str) or not case["dataset"]:
            raise ValueError("Each case needs a nonempty dataset path")
        dataset_path = Path(case["dataset"])
        if not dataset_path.is_absolute():
            dataset_path = cases_path.parent / dataset_path
        dataset_path = str(dataset_path.resolve(strict=True))
        if dataset_path not in identities:
            identities[dataset_path] = file_identity(dataset_path, count_lines=True)
        dataset = identities[dataset_path]
        if dataset["line_count"] < args.offset + args.num_samples:
            raise ValueError(f"Not enough instances in {dataset_path} for the requested offset/count")
        options = {key: case[key] for key in (
            "n_subregions", "arrival_weights", "arrival_skews",
            "time_window_scenario", "min_time_window", "max_time_window",
        ) if key in case}
        service_prior = {
            "service_times_mean": case.get("service_times_mean", 3.0),
            "service_times_var": case.get("service_times_var", 5.0),
        }
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in service_prior.values()):
            raise ValueError("Service-time mean and variance must be finite and nonnegative")
        if service_prior["service_times_mean"] == 0 and service_prior["service_times_var"] != 0:
            raise ValueError("A zero service-time mean requires zero variance")
        ortec = None
        if case.get("use_ortec"):
            ortec_path = Path(case["use_ortec"])
            if not ortec_path.is_absolute():
                ortec_path = cases_path.parent / ortec_path
            key = str(ortec_path.resolve(strict=True))
            if key not in identities:
                identities[key] = file_identity(key)
            ortec = identities[key]
        elif "_testing_ortec_" in Path(dataset_path).name:
            raise ValueError("ORTEC cases need use_ortec so the saved asymmetric distance matrix is loaded")
        dod = case.get("dod")
        if "msa" in args.methods and (not isinstance(dod, (int, float)) or not 0 <= dod <= 1):
            raise ValueError("MSA cases need dod between zero and one")
        case_key = digest({"problem": case["problem"], "dataset": dataset, "dod": dod,
                           "ortec": ortec, "options": options, "service_time_prior": service_prior})
        if case_key in seen_cases:
            raise ValueError("Duplicate comparison case")
        seen_cases.add(case_key)
        for method in args.methods:
            for seed in args.seeds if method == "msa" else [None]:
                for sample in range(args.offset, args.offset + args.num_samples):
                    cpus = workers + 1 if method == "msa" and workers > 1 else 1
                    task = {
                        "id": f"{len(tasks):06d}", "method": method, "problem": case["problem"],
                        "dataset": dataset, "ortec": ortec, "sample": sample, "seed": seed,
                        "dod": dod, "scenario_options": options, "scenarios": args.scenarios,
                        "service_time_prior": service_prior,
                        "workers": workers if method == "msa" else 1,
                        "no_improvement_ms": args.no_improvement_ms,
                        "time_limit_ms": args.time_limit_ms,
                        "cpus_per_task": cpus, "memory_mib": cpus * args.memory_per_cpu_mib,
                    }
                    task["fingerprint"] = digest({"task": task, "source": code_digest, "python": python})
                    tasks.append(task)
    groups = defaultdict(list)
    for task in tasks:
        groups["msa" if task["method"] == "msa" else "serial"].append(task)
    arrays = []
    peak_cpus = peak_memory = 0
    for group_index, (group, members) in enumerate(sorted(groups.items())):
        cpu_budget = args.total_cpus // len(groups) + (group_index < args.total_cpus % len(groups))
        total_memory = args.total_memory_gib * 1024
        memory_budget = total_memory // len(groups)
        cpus = members[0]["cpus_per_task"]
        memory = members[0]["memory_mib"]
        concurrency = min(len(members), args.max_array_size, cpu_budget // cpus, memory_budget // memory)
        if concurrency < 1:
            raise ValueError(f"CPU/memory budget cannot run one {group} task; lower workers or change budgets")
        peak_cpus += concurrency * cpus
        peak_memory += concurrency * memory
        previous = None
        for start in range(0, len(members), args.max_array_size):
            name = f"{group}-{start // args.max_array_size:03d}"
            batch = members[start:start + args.max_array_size]
            arrays.append({
                "name": name, "group": group, "task_ids": [task["id"] for task in batch],
                "cpus_per_task": cpus, "memory_mib": memory,
                "concurrency": min(concurrency, len(batch)),
                "partition": args.msa_partition if group == "msa" and cpus > 1 else args.serial_partition,
                "walltime": args.walltime, "after": previous,
            })
            previous = name
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    plan = {
        "schema_version": 1, "project_dir": str(PROJECT_DIR), "python": python,
        "source_digest": code_digest, "git_revision": revision,
        "result_root": str(output / "results"), "account": args.account,
        "resources": {"total_cpus": args.total_cpus, "total_memory_mib": args.total_memory_gib * 1024,
                      "peak_cpus": peak_cpus, "peak_memory_mib": peak_memory},
        "tasks": tasks, "arrays": arrays,
    }
    plan["plan_id"] = digest(plan)
    plan_path = output / "plan.json"
    if plan_path.exists():
        if load_plan(plan_path)["plan_id"] != plan["plan_id"]:
            raise ValueError("A different plan already exists here; use a new output directory")
    else:
        write_json(plan_path, plan)
    (output / "logs").mkdir(exist_ok=True)
    return plan


def load_plan(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    expected = plan.get("plan_id")
    if plan.get("schema_version") != 1 or digest({k: v for k, v in plan.items() if k != "plan_id"}) != expected:
        raise ValueError("Invalid or modified comparison plan")
    return plan


def sbatch_command(plan_path, plan, array, dependency=None, indices=None):
    plan_path = Path(plan_path).resolve()
    command = [
        "sbatch", "--parsable", "--nodes=1", "--ntasks=1",
        f"--cpus-per-task={array['cpus_per_task']}", f"--mem={array['memory_mib']}M",
        "--array=" + (
            ",".join(map(str, indices)) if indices is not None
            else f"0-{len(array['task_ids']) - 1}"
        ) + f"%{array['concurrency']}",
        f"--partition={array['partition']}", f"--time={array['walltime']}",
        f"--output={plan_path.parent / 'logs' / (array['name'] + '.%A_%a.out')}",
        f"--error={plan_path.parent / 'logs' / (array['name'] + '.%A_%a.err')}",
    ]
    if plan.get("account"):
        command.append(f"--account={plan['account']}")
    if dependency:
        command.append(f"--dependency=afterany:{dependency}")
    command.extend([
        str(Path(plan["project_dir"]) / "scripts" / "slurm-evaluate.sh"),
        str(plan_path), array["name"], plan["python"], plan["project_dir"],
    ])
    return command


def submit_plan(plan_path, execute=False, retry=False):
    plan = load_plan(plan_path)
    if execute:
        with task_lock(Path(plan_path).resolve().parent / ".submission.lock"):
            return _submit_plan(plan_path, plan, execute=True, retry=retry)
    return _submit_plan(plan_path, plan, execute=False, retry=retry)


def _submit_plan(plan_path, plan, execute, retry):
    submission_path = Path(plan_path).parent / "submitted.json"
    previous = json.loads(submission_path.read_text()) if submission_path.exists() else {}
    if previous and previous.get("plan_id") != plan["plan_id"]:
        raise ValueError("Submission history belongs to a different comparison plan")
    old_jobs = previous.get("jobs", {})
    if execute and old_jobs and not retry:
        raise ValueError("This plan was already submitted; use --retry-incomplete to submit only unfinished tasks")
    last_old_job = {}
    for array in plan["arrays"]:
        if array["name"] in old_jobs:
            last_old_job[array["group"]] = old_jobs[array["name"]]
    last_old_job.update(previous.get("group_tails", {}))
    group_tails = dict(last_old_job)
    commands = []
    submitted = {}
    last_new_array = {}
    tasks = {task["id"]: task for task in plan["tasks"]}
    for array in plan["arrays"]:
        indices = None
        if retry:
            indices = []
            for index, task_id in enumerate(array["task_ids"]):
                directory = Path(plan["result_root"]) / task_id
                if directory.exists():
                    completion_record(directory, plan, tasks[task_id])
                else:
                    indices.append(index)
            if not indices:
                continue
        prior_array = last_new_array.get(array["group"])
        dependency = submitted.get(prior_array, prior_array) if prior_array else last_old_job.get(array["group"])
        if not retry and not old_jobs:
            dependency = submitted.get(array["after"], array["after"])
        command = sbatch_command(plan_path, plan, array, dependency, indices)
        commands.append({"array": array["name"], "command": command})
        last_new_array[array["group"]] = array["name"]
        if execute:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Unexpected sbatch job id: {result.stdout!r}")
            submitted[array["name"]] = job_id
            group_tails[array["group"]] = job_id
            write_json(submission_path, {
                "plan_id": plan["plan_id"], "jobs": {**old_jobs, **submitted},
                "group_tails": group_tails,
            })
    return {"executed": execute, "resources": plan["resources"], "arrays": commands, "jobs": submitted}


@contextmanager
def task_lock(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This comparison task is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def completion_record(directory, plan, task):
    marker = json.loads((directory / "completed.json").read_text())
    if (marker.get("status") != "complete" or marker.get("plan_id") != plan["plan_id"]
            or marker.get("fingerprint") != task["fingerprint"]
            or marker.get("task_id") != task["id"]
            or marker.get("sample") != task["sample"]):
        raise ValueError(f"Completed result has a different configuration or instance: {directory}")
    record = marker.get("record", {})
    identity = {"task_id": task["id"], "method": task["method"], "problem": task["problem"],
                "dataset": task["dataset"]["path"], "sample": task["sample"], "seed": task["seed"],
                "workers": task["workers"]}
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Completed result record has a different identity: {directory}")
    if any(not isinstance(record.get(key), (float, int)) or not math.isfinite(record[key])
           for key in ("objective", "runtime_seconds", "job_wall_time_seconds")):
        raise ValueError(f"Completed result has invalid measurements: {directory}")
    return record


def evaluation_command(plan, task, directory):
    msa = task["method"] == "msa"
    scripts = {
        "msa": "evaluate_msa_ortools.py",
        "online": "evaluate_online_ortools.py",
        "oracle": "evaluate_clairvoyant_ortools.py",
    }
    command = [plan["python"], "-u", str(Path(plan["project_dir"]) / scripts[task["method"]]),
               "--problem", task["problem"], "--dataset", task["dataset"]["path"],
               "--offset", str(task["sample"]), "--num-samples", "1", "--output",
               str(directory / ("raw.json" if msa else "raw"))]
    if task["ortec"]:
        command.extend(["--use-ortec", task["ortec"]["path"]])
    if task["method"] != "oracle":
        for name, value in task["service_time_prior"].items():
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if msa:
        command.extend(["--dod", str(task["dod"]), "--seed", str(task["seed"]),
                        "--scenarios", str(task["scenarios"]), "--workers", str(task["workers"])])
        for key, value in task["scenario_options"].items():
            command.append("--" + key.replace("_", "-"))
            command.extend(map(str, value if isinstance(value, list) else [value]))
        prefix = "solver"
    else:
        prefix = "online" if task["method"] == "online" else "static"
    command.extend([f"--{prefix}-no-improvement-ms", str(task["no_improvement_ms"])])
    if task["time_limit_ms"] is not None:
        command.extend([f"--{prefix}-time-ms", str(task["time_limit_ms"])])
    return command


def read_evaluation(directory, task):
    if task["method"] == "msa":
        raw = json.loads((directory / "raw.json").read_text())
        rows = raw["samples"]
        if len(rows) != 1 or rows[0]["sample"] != task["sample"]:
            raise ValueError("MSA result does not contain the requested global instance")
        row = rows[0]
        values = {"objective": row["objective"], "runtime_seconds": row["runtime"],
                  "aggregate_solver_seconds": row["aggregate_solver_seconds"],
                  "scenario_count": row["scenario_count"], "termination_counts": row["termination_counts"],
                  "instance_seed": row["instance_seed"]}
    else:
        with (directory / "raw" / "samples.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or int(rows[0]["sample"]) != task["sample"]:
            raise ValueError("OR-Tools result does not contain the requested global instance")
        row = rows[0]
        prefix = "online" if task["method"] == "online" else "static"
        values = {"objective": float(row[f"{prefix}_objective"]),
                  "runtime_seconds": float(row[f"{prefix}_runtime_seconds"]),
                  "termination_counts": json.loads(row["online_termination_reasons"])
                  if prefix == "online" else {row["static_termination_reason"]: 1}}
    if not math.isfinite(values["objective"]) or not math.isfinite(values["runtime_seconds"]):
        raise ValueError("Comparison returned a nonfinite objective or runtime")
    service_metadata = {"service_time_model": "realized"}
    if task["method"] != "oracle":
        service_metadata.update(task["service_time_prior"])
        service_metadata["service_time_model"] = "sampled" if task["method"] == "msa" else "expected"
    return {"task_id": task["id"], "method": task["method"], "problem": task["problem"],
            "dataset": task["dataset"]["path"], "sample": task["sample"], "seed": task["seed"],
            "workers": task["workers"], **values, **service_metadata}


def run_task(plan_path, array_name, index):
    plan = load_plan(plan_path)
    array = next((item for item in plan["arrays"] if item["name"] == array_name), None)
    if array is None or index < 0 or index >= len(array["task_ids"]):
        raise ValueError("Unknown array or out-of-range array index")
    task_id = array["task_ids"][index]
    task = next(item for item in plan["tasks"] if item["id"] == task_id)
    root = Path(plan["result_root"])
    root.mkdir(parents=True, exist_ok=True)
    with task_lock(root / ".locks" / f"{task_id}.lock"):
        destination = root / task_id
        if destination.exists():
            return {"skipped": True, "record": completion_record(destination, plan, task)}
        if source_digest() != plan["source_digest"]:
            raise ValueError("Source changed after planning; create a new plan from the frozen checkout")
        for identity in (task["dataset"], task["ortec"]):
            if identity is not None:
                stat = Path(identity["path"]).stat()
                if stat.st_size != identity["size"] or stat.st_mtime_ns != identity["mtime_ns"]:
                    raise ValueError("Input file changed after planning")
        allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", task["cpus_per_task"]))
        if allocated < task["cpus_per_task"]:
            raise ValueError("Slurm CPU allocation is smaller than this task requires")
        environment = os.environ.copy()
        environment.update({name: "1" for name in THREAD_VARIABLES})
        environment["PYTHONUNBUFFERED"] = "1"
        with tempfile.TemporaryDirectory(dir=root, prefix=f".{task_id}.partial-") as temporary:
            directory = Path(temporary)
            command = evaluation_command(plan, task, directory)
            started = time.perf_counter()
            subprocess.run(command, cwd=plan["project_dir"], env=environment, check=True)
            record = read_evaluation(directory, task)
            record["job_wall_time_seconds"] = time.perf_counter() - started
            versions = {}
            for package in ("ortools", "protobuf", "numpy", "torch"):
                try:
                    versions[package] = metadata.version(package)
                except metadata.PackageNotFoundError:
                    versions[package] = None
            write_json(directory / "completed.json", {
                "status": "complete", "plan_id": plan["plan_id"], "task_id": task_id,
                "fingerprint": task["fingerprint"], "sample": task["sample"], "record": record,
                "command": command, "host": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cpus_per_task": task["cpus_per_task"], "versions": versions,
                "git_revision": plan["git_revision"], "source_digest": plan["source_digest"],
            })
            os.replace(directory, destination)
        return {"skipped": False, "record": record}


def collect_results(plan_path, allow_partial=False):
    plan = load_plan(plan_path)
    records = []
    missing = []
    for task in plan["tasks"]:
        directory = Path(plan["result_root"]) / task["id"]
        if not directory.exists():
            missing.append(task["id"])
        else:
            records.append(completion_record(directory, plan, task))
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} tasks are incomplete; retry them or explicitly use --allow-partial")
    groups = defaultdict(list)
    group_fields = ("method", "problem", "dataset", "seed", "service_time_model",
                    "service_times_mean", "service_times_var")
    for record in records:
        key = tuple(record.get(field) for field in group_fields)
        groups[key].append(record)
    summaries = []
    for key, group in sorted(groups.items(), key=lambda pair: str(pair[0])):
        summaries.append(dict(zip(group_fields, key),
                              samples=len(group),
                              objective_mean=sum(row["objective"] for row in group) / len(group),
                              runtime_mean_seconds=sum(row["runtime_seconds"] for row in group) / len(group)))
    summary = {"plan_id": plan["plan_id"], "completed": len(records), "expected": len(plan["tasks"]),
               "missing_task_ids": missing, "groups": summaries, "resources": plan["resources"]}
    output = Path(plan_path).resolve().parent / "collected"
    output.mkdir(exist_ok=True)
    columns = ["task_id", "method", "problem", "dataset", "sample", "seed", "workers",
               "objective", "runtime_seconds", "job_wall_time_seconds", "aggregate_solver_seconds",
               "scenario_count", "instance_seed", "termination_counts", "service_time_model",
               "service_times_mean", "service_times_var"]
    with (output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                             for key, value in record.items()})
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan", help="Freeze cases and bounded resource requests without submitting")
    plan.add_argument("--cases", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    plan.add_argument("--num-samples", type=int, default=100)
    plan.add_argument("--offset", type=int, default=0)
    plan.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    plan.add_argument("--scenarios", type=int, default=10)
    plan.add_argument("--msa-workers", type=int, default=15)
    plan.add_argument("--total-cpus", type=int, default=512)
    plan.add_argument("--total-memory-gib", type=int, default=1800)
    plan.add_argument("--memory-per-cpu-mib", type=int, default=3072)
    plan.add_argument("--max-array-size", type=int, default=1000)
    plan.add_argument("--no-improvement-ms", type=int, default=60000)
    plan.add_argument("--time-limit-ms", type=int)
    plan.add_argument("--python", default=sys.executable)
    plan.add_argument("--serial-partition", default="serial")
    plan.add_argument("--msa-partition", default="parallel")
    plan.add_argument("--walltime", default="1-00:00:00")
    plan.add_argument("--account")
    submit = commands.add_parser("submit", help="Print sbatch commands; --execute submits the reviewed plan")
    submit.add_argument("--plan", required=True)
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--retry-incomplete", action="store_true",
                        help="Submit unfinished instances after the preceding submission's jobs")
    task = commands.add_parser("run-task")
    task.add_argument("--plan", required=True)
    task.add_argument("--array", required=True)
    task.add_argument("--index", type=int, required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--plan", required=True)
    collect.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            result = build_plan(args)
            print(json.dumps({"plan": str(Path(args.output).resolve() / "plan.json"),
                              "tasks": len(result["tasks"]), "arrays": len(result["arrays"]),
                              "resources": result["resources"]}, indent=2))
        elif args.action == "submit":
            print(json.dumps(submit_plan(args.plan, args.execute, args.retry_incomplete), indent=2))
        elif args.action == "run-task":
            print(json.dumps(run_task(args.plan, args.array, args.index), indent=2))
        else:
            print(json.dumps(collect_results(args.plan, args.allow_partial), indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
