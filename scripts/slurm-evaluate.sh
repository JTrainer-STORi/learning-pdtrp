#!/usr/bin/env bash
#SBATCH --job-name=pdtrp-compare
set -euo pipefail

usage() {
    echo "Usage: scripts/slurm-evaluate.sh PLAN_PATH ARRAY_NAME PYTHON PROJECT_DIR"
    echo "Run through comparison_jobs.py submit; resources are set by the plan."
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 4 ]]; then
    usage >&2
    exit 2
fi

PLAN_PATH=$1
ARRAY_NAME=$2
PYTHON_BIN=$3
PROJECT_DIR=$4
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?This runner requires a Slurm array task ID}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}

[[ -f "$PLAN_PATH" ]] || { echo "Plan not found: $PLAN_PATH" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python interpreter is not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$PROJECT_DIR/comparison_jobs.py" ]] || { echo "Comparison harness not found in: $PROJECT_DIR" >&2; exit 1; }
[[ "$TASK_INDEX" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm array task ID: $TASK_INDEX" >&2; exit 2; }
[[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid Slurm CPU allocation: $CPUS_PER_TASK" >&2; exit 2; }

# Each OR-Tools solve uses one CPU. MSA gets separate worker processes, with
# one extra allocated CPU for its parent, rather than nested numerical threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
exec srun --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --cpu-bind=cores \
    "$PYTHON_BIN" "$PROJECT_DIR/comparison_jobs.py" run-task \
    --plan "$PLAN_PATH" --array "$ARRAY_NAME" --index "$TASK_INDEX"
