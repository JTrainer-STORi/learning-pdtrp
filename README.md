# Learning policies for partially dynamic repairman problems

Research code for training and evaluating reinforcement-learning construction
heuristics for routing problems in which customer requests are revealed over
time. The repository supports four variants:

- **PDTRP** — partially dynamic travelling repairman problem;
- **PDCTRP** — capacitated PDTRP;
- **PDTRP-TW** — PDTRP with time windows;
- **PDCTRP-TW** — capacitated PDTRP with time windows.

The model combines a graph encoder and autoregressive attention decoder with
dynamic state updates. Time-window configurations can additionally use an
arrival-history LSTM. The implementation builds on ideas and code structure
from *Learning TSP Requires Rethinking Generalization* and *Attention, Learn to
Solve Routing Problems!*.

## Installation

Python 3.11 is the supported version. With Python 3.11 installed, run these
commands from the repository root to create and activate a virtual environment
using Python's built-in `venv` module:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Activate the environment with `source .venv/bin/activate` in each new shell
before running training, evaluation, or tests. Run `deactivate` when finished.
The `.venv/` directory is ignored by Git; dependencies are listed in
`requirements.txt`.

PyTorch may be replaced with the build appropriate for the local CUDA driver.
Weights & Biases and TensorBoard logging can be disabled with `--no_wandb` and
`--no_tensorboard`.

## Quick start

Train a small PDTRP model:

```bash
python run.py \
  --problem pdtrp \
  --model dattention \
  --encoder gnn \
  --baseline rollout \
  --min_total 20 \
  --max_total 50 \
  --min_dod 0.2 \
  --max_dod 0.8 \
  --n_epochs 1 \
  --epoch_size 1280 \
  --batch_size 128 \
  --no_wandb \
  --no_tensorboard \
  --no_videos
```

The portable wrappers in [`scripts/`](scripts/) expose the common workflows.
Run `python run.py --help` for the complete training interface.

Evaluate a saved checkpoint on a fixed dataset:

```bash
python evaluate_trained_policy.py \
  --problem pdtrp \
  --checkpoint /path/to/epoch-99.pt \
  --datasets /path/to/pdtrp50dod0.5v4th8_testing_ortools.txt \
  --output-dir evaluation-output \
  --device cpu
```

The evaluator writes per-instance CSV data, a JSON summary, and a manifest
recording the checkpoint and dataset used. It also prints the manifest as a
JSON dictionary containing each dataset's objective and runtime statistics.
Progress messages go to stderr, so stdout can be redirected to a JSON file.

Add `--num-samples 1` to evaluate the first instance, matching the baseline
scripts' sample selection. Use `--offset 10 --num-samples 5` to select zero-based
instances 10 through 14, with those original indices retained in the CSV.
Selection applies independently to each dataset; a shorter nonempty slice uses
all remaining instances, and an empty slice raises an error. Defaults are offset
0 and at most 12800 instances per dataset. The wrapper accepts these options
after `--` as extra arguments.

## Fully dynamic instances

All four problem variants support 100% dynamic customer requests. Set
`min_dod=1.0` and `max_dod=1.0` when generating a dataset:

```python
from problems import PDTRP

dataset = PDTRP.make_dataset(
    min_total=50, max_total=50,
    min_dod=1.0, max_dod=1.0,
    n_subregions=1, batch_size=1, num_samples=10,
)
```

Here, 50 is the node count including the depot, so each instance has 49
customers. Every customer calls after time zero. The vehicle waits at the depot
until the first call, reveals simultaneous calls together, and then starts
making decisions. This initial wait adds no travel or duplicate depot visit.

For training, use `--min_dod 1.0 --max_dod 1.0`, or configure the wrapper:

```bash
MIN_DOD=1.0 MAX_DOD=1.0 scripts/train-policy.sh pdtrp 50 0
```

ORTEC generation and MSA sampling also count dynamism over customers, excluding
the depot. A fully dynamic ORTEC draw samples customers whose derived call times
are positive; it raises an error if the source has too few eligible customers.
Existing ORTEC time windows are retained.

## Synthetic time windows

PDTRP-TW and PDCTRP-TW use chapter 7's immediate-request opening rule:
`window_start = call_time + distance_from_depot / speed`. Advance requests
are known at time zero and receive random openings no earlier than direct depot
arrival. A shared generator supports these width presets, in minutes:

| Scenario | Minimum width | Maximum width |
| --- | --- | --- |
| `narrow` | 30 | 90 |
| `wide` (default) | 60 | 180 |

Select a training preset with `--time_window_scenario narrow` or
`--time_window_scenario wide`. Direct dataset generation accepts
`time_window_scenario="narrow"` or `"wide"`. Explicit `--min_time_window` and
`--max_time_window` values override the preset bounds.

Closing times are sampled uniformly between opening plus the minimum width and
the smaller of opening plus the maximum width or `time_horizon + latest_end`.
The default final closing horizon is 600 minutes (480 + 120). This is our
explicit closing-time sampling convention; the thesis specifies the opening
rule and narrow/wide bounds without fully specifying that sampling law.
Impossible combinations of minimum width and closing horizon raise an error.

MSA accepts `--time-window-scenario narrow|wide` and optional
`--min-time-window` / `--max-time-window` overrides. It uses the configured
width prior when generating synthetic future scenarios. HEC case JSON can
include `"time_window_scenario": "narrow"` and optional explicit width bounds.

These settings generate new synthetic windows. Files loaded for evaluation
and ORTEC source windows retain their supplied values; regenerate synthetic
datasets to benchmark the new rule. For MSA, select the prior matching that
dataset. The legacy `reaction_time` option applies to ORTEC call-time derivation
and no longer delays synthetic TW openings. Fixed request counts are retained.

## Baselines

The public evaluation entry points are:

| Script | Method |
| --- | --- |
| `evaluate_model_free.py` | Nearest neighbour and first-come-first-served |
| `evaluate_online_ortools.py` | Online OR-Tools reoptimization |
| `evaluate_clairvoyant_ortools.py` | Clairvoyant OR-Tools oracle |
| `evaluate_ortools_baselines.py` | Paired online/oracle comparison (compatibility entry point) |
| `evaluate_msa_ortools.py` | Multiple-scenario/sample-average approximation |
| `evaluate_trained_policy.py` | Greedy inference with a trained policy |

Run online and clairvoyant evaluation independently:

```bash
DATASET=/path/to/pdtrp50dod0.5v4th8_testing_ortools.txt
python evaluate_online_ortools.py --problem pdtrp --dataset "$DATASET" \
  --num-samples 1 --output results/online-pdtrp50
python evaluate_clairvoyant_ortools.py --problem pdtrp --dataset "$DATASET" \
  --num-samples 1 --output results/oracle-pdtrp50
```

Each script evaluates its named method without a method selector. The baseline
wrapper exposes them as `online` and `oracle` (`clairvoyant` is an alias).
Use the combined `ortools` wrapper method for paired comparisons.

The routing baselines use the same request release times as policy evaluation:
the vehicle cannot depart for a request until it is revealed. Travel and service
times advance the clock; time-window variants wait for window starts and penalize
lateness beyond window ends. The objective is closed-tour distance without time
windows, or `theta * distance + gamma * summed lateness` with time windows.
In both time-window environments, the vehicle waits at its current location
until the planned departure time. A new request received during that wait
triggers another decision; a request arriving exactly at departure is also
revealed before committing. Once departure occurs, travel and service finish
without interruption. Canceled choices add no tour visit or capacity usage, but
time already spent waiting remains part of the recorded schedule. Training,
policy evaluation, and the online heuristic/OR-Tools/MSA baselines share this
rule. Online TW scoring uses recorded service-start times because recomputing
the earliest schedule from a tour can erase time spent on canceled waits.
Existing checkpoints still load, but should be retrained for this decision rule;
rerun older evaluations before comparing results.

OR-Tools uses integer arithmetic: times are rounded to four decimal places,
and lateness weights and demands to six. Outputs include both the solver's
objective and the route's objective recomputed by the library's scoring code.

Online OR-Tools plans with the expected service duration of each unserved
customer. MSA samples an independent log-normal service duration for every
unserved customer in each scenario, including already revealed customers and
customers sampled from an ORTEC pool. Neither planner reads those customers'
realized service durations. The simulator uses the realized durations when
executing actions and computing the final objective; the clairvoyant oracle
retains those realized durations as part of its reference information.

The known service-time prior defaults to mean **3 minutes** and variance
**5 minutes²**. Set `--service-times-mean` and `--service-times-var` on online,
paired, or MSA evaluation to match the prior used to generate a dataset. These
are distribution parameters, not statistics estimated from the held-out
instance. Evaluation summaries record the service-time assumptions. The
wrappers accept these options after `--` as extra arguments.

The oracle knows all requests in advance while respecting those release times.
It is a clairvoyant heuristic reference with no guarantee of optimality. By
default its search stops after 60 seconds without a strict improvement in the
best objective; equal or worse solutions do not reset that timer. There is no
default total time limit. Use `--static-no-improvement-ms` to change the wait and
optionally `--static-time-ms` to impose a total cap. Evaluation outputs record
the oracle runtime and stopping reason.

The baseline wrapper exposes these as `STATIC_NO_IMPROVEMENT_MS` (default
`60000`) and optional `STATIC_TIME_MS`. For example, wait five seconds without
improvement and also cap the oracle search at one minute per instance:

```bash
STATIC_NO_IMPROVEMENT_MS=5000 STATIC_TIME_MS=60000 NUM_SAMPLES=1 \
  scripts/evaluate-baseline.sh oracle pdtrp \
  /path/to/pdtrp50dod0.5v4th8_testing_ortools.txt \
  results/oracle-pdtrp50
```

Online reoptimization and each MSA scenario use the same stopping rule: 60
seconds without a strict improvement, with no default total time cap.
`--online-no-improvement-ms` and `--solver-no-improvement-ms` control their
respective waits. Optional `--online-time-ms` and `--solver-time-ms` impose
total limits per online solve and per scenario solve. In the wrapper, set
`ONLINE_NO_IMPROVEMENT_MS` or `SOLVER_NO_IMPROVEMENT_MS` to change the wait;
leave `ONLINE_TIME_MS` and `SOLVER_TIME_MS` unset for uncapped searches.

MSA can solve independent scenarios in parallel CPU worker processes. Use
`--workers` (`MSA_WORKERS` in the wrapper); the default is `1`, and `0` selects
the available CPU count, limited by the scenario count. `--scenarios`
(`SCENARIOS`, default `10`) controls how
many scenarios contribute to each decision. The parent process generates
scenarios in a stable order, and voting ties follow that order regardless of
worker completion order. Service durations are also sampled in the parent, so
changing the worker count preserves the seeded scenario draws. The CLI reuses
its worker pool across decisions and
instances. More workers can make larger scenario counts practical without
paying the full serial solve time after startup; speed depends on the workload
and available CPUs. Outputs distinguish elapsed wall time (`wall_time_seconds`),
time spent waiting for solve batches (`solver_wall_time_seconds`), and the sum
of scenario solver durations (`aggregate_solver_seconds`). The summed durations
can exceed wall time during parallel work.

For example, use four workers and 40 scenarios, each waiting one second without
improvement. Leave `SOLVER_TIME_MS` unset to run without a total scenario cap:

```bash
DOD=0.5 MSA_WORKERS=4 SCENARIOS=40 SOLVER_NO_IMPROVEMENT_MS=1000 NUM_SAMPLES=1 \
  scripts/evaluate-baseline.sh msa pdtrp \
  /path/to/pdtrp50dod0.5v4th8_testing_ortools.txt \
  results/msa-pdtrp50
```

Add `SOLVER_TIME_MS=10000` to also cap each scenario at ten seconds. For online
reoptimization, `ONLINE_TIME_MS=10000` adds the equivalent per-solve cap.

## Reproducing the experiments

Portable scripts under [`scripts/`](scripts/) cover training, smoke testing,
baseline evaluation, and trained-policy evaluation.

```bash
# Train a uniform PDTRP policy on instances containing 20--50 customers.
scripts/train-policy.sh pdtrp 50 0

# Evaluate nearest neighbour on a fixed dataset.
scripts/evaluate-baseline.sh nearest pdtrp \
  /path/to/pdtrp50dod0.5v4th8_testing_ortools.txt \
  results/nearest-pdtrp50

# Evaluate a checkpoint greedily.
scripts/evaluate-policy.sh pdtrp /path/to/epoch-99.pt \
  /path/to/pdtrp50dod0.5v4th8_testing_ortools.txt \
  results/policy-pdtrp50
```

Configuration is controlled through documented environment variables in each
script. For example, use `DEVICE=cpu`, `DEVICE=0`, or `NUM_SAMPLES=10`.

For Slurm arrays covering online, oracle, and MSA comparisons, follow the
[HEC comparison guide](docs/hec-comparisons.md).

Research datasets, checkpoints, and result tables are intentionally excluded
from the source repository. Training generates instances and validation sets
in memory. Evaluation commands accept explicit dataset and checkpoint paths,
allowing experimental artifacts to be distributed independently.

## Repository layout

```text
nets/                         neural policy and encoder implementations
problems/                     problem generators and dynamic environments
tests/                        unit and integration tests
scripts/                      portable training and evaluation entry points
```

## Testing

```bash
python -m pytest -q
```

The suite exercises instance generation, state transitions, capacity and time
window behaviour, comparison heuristics, OR-Tools scenario construction, model
integration, and evaluation utilities. Continuous integration runs the same
command on Python 3.11.

## License and disclaimer

This project is licensed under the [GNU General Public License v3.0](LICENSE).
