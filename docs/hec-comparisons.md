# Running comparisons on HEC

`comparison_jobs.py` plans and runs online OR-Tools, clairvoyant oracle, and
MSA comparisons as Slurm arrays. Each task evaluates one dataset instance;
MSA tasks also select one seed. Online and oracle runs are evaluated once per
instance, rather than repeated for every MSA seed.

The planner invokes `evaluate_online_ortools.py` for online tasks and
`evaluate_clairvoyant_ortools.py` for oracle tasks. These are independent
entry points; see the [individual evaluation commands](../README.md#baselines)
to run either method directly. Plan selection remains `--methods online`,
`--methods oracle`, or both alongside `msa`.

The allocation is **512 CPUs and 2 TB across nodes**. Each MSA process pool
runs within one allocated node; independent array tasks distribute the work
across nodes. Lancaster HEC uses the `serial` partition for single-core jobs
and `parallel` for jobs using multiple cores on one node, as described in its
[parallel jobs guide](https://lancaster-hec.readthedocs.io/en/latest/parallel.html).

## Prepare the checkout and cases

Use the [Python 3.11 venv setup](../README.md#installation) on a filesystem
visible to compute nodes. The checkout, interpreter, datasets, ORTEC source
files, and output directory must all be accessible there. Load any Python
module required by your site before creating the venv and submitting jobs.
The runner invokes the venv interpreter directly; activation is unnecessary
inside a job.

For synthetic uniform-uniform instances, generate the fixed files with
`generate_data.py` before planning comparisons. For example:

```bash
python generate_data.py --problem pdtrp --num-customers 50 --dod 0.5 \
  --num-samples 100 --seed 20 \
  --output artifacts/benchmarks/uniform-uniform/pdtrp50-dod0.5-seed20.txt
```

Repeat for the desired canonical problems and dynamism levels, then reference
those files in the cases JSON. Keep the `.txt.json` metadata alongside each
dataset when copying it to HEC. N=50 means 50 customers plus the depot, for 51
graph nodes. The embedded nearest-neighbour tours are heuristic references;
comparison tasks compute their own method-specific routes. See the
[generation details](../README.md#generate-uniform-uniform-benchmarks).

Create a JSON file such as `/path/to/cases.json`:

```json
[
  {
    "problem": "pdtrp",
    "dataset": "datasets/pdtrp50dod0.5v4th8_testing_ortools.txt",
    "dod": 0.5,
    "service_times_mean": 3.0,
    "service_times_var": 5.0
  },
  {
    "problem": "pdctrptw",
    "dataset": "/shared/datasets/pdctrptw50_testing_ortec_example.txt",
    "dod": 0.5,
    "use_ortec": "/shared/ortec/source.vrp"
  }
]
```

Paths may be absolute or relative to the cases file's directory. Set `dod`
to the dataset's degree of dynamism. Optional `n_subregions`, `arrival_weights`,
and `arrival_skews` describe the MSA sampling distribution; `time_window_scenario`
selects `narrow` (30–90 minutes) or `wide` (60–180, default) for synthetic TW
scenarios, with optional `min_time_window` / `max_time_window` overrides. These
must match the dataset's generating prior; they do not rewrite saved windows.
Optional `speed` (distance units/hour), `time_horizon` (hours), and `latest_end`
(hours after the call horizon) must also match generation. Defaults are 4, 8,
and 2. For the N=50 Scenario C scaled benchmark, set `"time_horizon": 10`.
These settings reach every comparison method, are recorded in task
fingerprints, and remain separate in collected result groups. Evaluation
summaries record them under `dataset_parameters`; collected CSV/JSON results
include `speed`, `time_horizon`, and `latest_end`.
`use_ortec` selects
the source for ORTEC-derived datasets.

Optional `service_times_mean` and `service_times_var` specify the known prior in
minutes and minutes²; defaults are 3 and 5. Set these to the distribution used
to generate the data, rather than estimating them from unserved customers in an
evaluation instance. Online OR-Tools uses the prior mean. MSA samples independent
log-normal service durations for known and future customers in each scenario,
including ORTEC scenarios. Actual service durations advance the simulator and
determine the scored outcome; the clairvoyant reference keeps access to them.

## Run a pilot

Start with a few instances and scenarios to check timing and memory use:

```bash
python comparison_jobs.py plan \
  --cases /path/to/cases.json --output artifacts/hec-pilot \
  --num-samples 2 --seeds 0 --scenarios 4 --msa-workers 2 \
  --no-improvement-ms 1000 --time-limit-ms 10000 \
  --total-cpus 8 --total-memory-gib 24 --memory-per-cpu-mib 3072 \
  --walltime 00:20:00 --python "$PWD/.venv/bin/python" \
  --serial-partition serial --msa-partition parallel

python comparison_jobs.py submit --plan artifacts/hec-pilot/plan.json
```

`submit` prints the proposed `sbatch` commands by default. Review the plan and
commands, then submit the pilot explicitly:

```bash
python comparison_jobs.py submit --plan artifacts/hec-pilot/plan.json --execute
```

Check the pilot's elapsed time, peak memory, and solver stopping reasons before
choosing production settings. `--no-improvement-ms` controls the wait since a
strict best improvement for all three methods. The optional `--time-limit-ms`
adds a total cap to each online, oracle, or MSA scenario solve. These limits
apply to individual solves; Slurm's `--walltime` limits the entire instance
task, including every decision and scenario batch.

## Plan the full comparison

This example requests 100 instances per case, three MSA seeds, and 40 scenarios
per decision. It uses the default 60-second improvement wait and leaves total
solve caps unset:

```bash
python comparison_jobs.py plan \
  --cases /path/to/cases.json --output artifacts/hec-comparisons \
  --methods online oracle msa --num-samples 100 --seeds 0 1 2 \
  --scenarios 40 --msa-workers 15 --no-improvement-ms 60000 \
  --total-cpus 512 --total-memory-gib 1800 --memory-per-cpu-mib 3072 \
  --max-array-size 1000 --walltime 1-00:00:00 \
  --python "$PWD/.venv/bin/python" \
  --serial-partition serial --msa-partition parallel

python comparison_jobs.py submit --plan artifacts/hec-comparisons/plan.json
```

The memory budget of 1800 GiB stays below the 2 TB allocation. With these
settings, CPU capacity determines the concurrency limits:

| Methods in the plan | Serial tasks | MSA tasks | Maximum CPUs in use |
| --- | --- | --- | --- |
| Online/oracle and MSA | Up to 256 at one CPU each | Up to 16 at 16 CPUs each | 512 |
| MSA only | None | Up to 32 at 16 CPUs each | 512 |

Each MSA task reserves 15 worker CPUs plus one CPU for the parent process.
Workers are limited to the scenario count. Numerical thread counts are fixed
at one, so libraries do not create extra thread pools in each worker. The
planner splits CPU and memory budgets between the serial and MSA groups when
both are present; memory settings can reduce concurrency further.

Arrays are split into batches of at most 1000 tasks by default. Later batches
within each group depend on the preceding batch with Slurm `afterany`, keeping
the concurrency budget from multiplying across batches. The serial and MSA
groups can run concurrently. Scheduler availability can lower actual usage.

Each MSA task reuses its local worker pool across decisions. Forty scenarios
on 15 workers still require several scheduling waves; adding workers does not
guarantee proportional speedups. Parent-generated scenario order and vote tie
order are stable. Outputs distinguish wall time from summed solver durations.
Service durations are sampled in the parent before dispatch, preserving seeded
scenario draws across worker counts.

The 60-second improvement wait applies to each scenario solve at each decision,
so a large comparison can take substantial time even with 512 CPUs. Use pilot
measurements to choose the instance job's walltime. Checkpoints cover completed
instances; an instance interrupted by the scheduler must restart from its first
decision.

To run only MSA, use `--methods msa`. `--offset` selects a later starting
instance without editing the datasets. Use a new output directory when
changing the comparison configuration.

## Resume and collect

Tasks publish each completed instance atomically and reuse verified completions
when the same plan is run again. The plan records input identities and task
fingerprints; changed inputs are rejected before execution. Failed or
incomplete attempts do not count as completed results. Resubmit the same plan
to finish missing work with:

```bash
python comparison_jobs.py submit --plan artifacts/hec-comparisons/plan.json \
  --retry-incomplete --execute
```

Retry arrays contain only unfinished instances and wait for the preceding
submission's jobs in their resource group. This also prevents a retry from
doubling the resource budget while earlier jobs are still running. A second
ordinary `--execute` is rejected to avoid accidental duplicate submissions.
Each plan controls its own budget; account for other plans and unrelated jobs
when choosing that budget.

After the arrays finish:

```bash
python comparison_jobs.py collect --plan artifacts/hec-comparisons/plan.json
```

Collection writes `collected/samples.csv` and `collected/summary.json` under
the run directory. Rows record the service-time model and prior; summaries keep
different prior settings in separate groups. Collection requires a complete comparison by default; use
`--allow-partial` explicitly when collecting an unfinished run.

The tracked entry point is `scripts/slurm-evaluate.sh`, called by the submission
commands with the plan, array name, interpreter, and checkout paths. Historical
local `scripts/hec-*.sh` scripts retain their old limits and are outside this
workflow.
