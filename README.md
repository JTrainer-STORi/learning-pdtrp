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

Python 3.11 is the supported version. Create the portable Conda environment:

```bash
conda env create -f environment.yml
conda activate learning-pdtrp
pytest -q
```

Alternatively, install the Python dependencies into an existing environment:

```bash
python -m pip install -r requirements.txt
pytest -q
```

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
recording the checkpoint and dataset used.

## Baselines

The public evaluation entry points are:

| Script | Method |
| --- | --- |
| `evaluate_model_free.py` | Nearest neighbour and first-come-first-served |
| `evaluate_ortools_baselines.py` | Online reoptimization and full-information OR-Tools |
| `evaluate_msa_ortools.py` | Multiple-scenario/sample-average approximation |
| `evaluate_trained_policy.py` | Greedy inference with a trained policy |

The full-information OR-Tools result is a time-limited heuristic reference,
not a proven optimum. Negative differences relative to it are therefore
possible and must not be described as negative optimality gaps.

## Reproducing the experiments

Portable scripts under [`scripts/`](scripts/) cover training, smoke testing,
baseline evaluation, and trained-policy evaluation on an ordinary workstation.
They resolve the repository directory automatically and require neither Slurm
nor site-specific environment variables.

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
pytest -q
```

The suite exercises instance generation, state transitions, capacity and time
window behaviour, comparison heuristics, OR-Tools scenario construction, model
integration, and evaluation utilities. Continuous integration runs the same
command on Python 3.11.

## Citation and licence

Citation metadata will be added when the associated thesis/paper record is
public. A source-code licence must also be selected before the repository is
made public, including confirmation that inherited third-party code may be
redistributed under that licence. No ORTEC-derived datasets are included.
