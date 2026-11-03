# Repository Guidelines

## Project Structure & Module Organization

This Python research library trains and compares routing policies for PDTRP,
PDCTRP, and their time-window variants. `problems/` contains instance generators,
state transitions, and scoring; `nets/` contains attention policies and graph
encoders. Shared loading, tensor, and OR-Tools utilities live in `utils/`.
`run.py`, `train.py`, and `reinforce_baselines.py` handle training.

Root-level `evaluate_*.py` files are evaluation commands. Keep this prefix to
distinguish them from pytest modules in `tests/`. Portable wrappers live in
`scripts/`; `docs/hec-comparisons.md` describes Slurm comparisons.

## Build, Test, and Development Commands

Use Python 3.11 and a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

There is no separate build step. `requirements.txt` pins compatible OR-Tools
and protobuf versions. Run these commands from the repository root:

- `python -m pytest tests/test_trained_policy_evaluation.py -q`: check policy evaluation.
- `scripts/smoke-test.sh`: run the suite and a generated-instance heuristic check.
- `python run.py --help`: inspect training options.
- `python evaluate_trained_policy.py --help`: inspect checkpoint evaluation options.

For local evaluation, begin with `--num-samples 1 --offset 0`. Set short solver
patience and explicit time caps for OR-Tools smoke runs.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, `PascalCase`
classes, and uppercase constants. Follow surrounding code; no formatter or
linter is currently configured. Keep shared calculations in utilities rather
than duplicating scoring or solver constraints across entry points.

## Testing Guidelines

Use pytest files named `tests/test_*.py` and functions named `test_*`. Add
regressions for changed behavior, including sample selection, output formats,
capacity constraints, or time-window handling. Keep fixtures small and seeded.
No coverage percentage is enforced. CI runs `python -m pytest -q` on Python 3.11.

## Commit & Pull Request Guidelines

History uses brief, plain-language commit subjects without an enforced prefix.
Describe the concrete change and keep commits focused. PRs should explain the
problem, resulting behavior, validation commands, and any experimental impact;
link relevant issues. Document new CLI options and changed result schemas.

## Data & Configuration

Keep datasets, checkpoints, results, and credentials out of commits. Local
artifact directories are ignored; pass explicit paths to evaluation commands.
Keep checkpoints alongside their `args.json`, and record seeds and solver
limits when comparing methods.
