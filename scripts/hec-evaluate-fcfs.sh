#!/usr/bin/env bash
# Evaluate first-come-first-served on all time-window table test sets:
# 2 problems x 3 distributions x 2 sizes x 3 degrees of dynamism.
#SBATCH --partition=serial
#SBATCH --job-name=pdtsp-fcfs
#SBATCH --array=0-35
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=pdtsp-fcfs.%A_%a.out
#SBATCH --error=pdtsp-fcfs.%A_%a.err

source /etc/profile
set -eo pipefail
module add miniforge
ENVIRONMENT_PATH="${global_storage:?global_storage is not set}/learning-tsp/tsp-new"
source activate "$ENVIRONMENT_PATH"
set -u

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be run as an array job}"
if (( TASK_ID < 0 || TASK_ID > 35 )); then
    echo "Task ID must be between 0 and 35, got ${TASK_ID}" >&2
    exit 2
fi

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
if ! [[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SAMPLES must be a positive integer, got ${NUM_SAMPLES}" >&2
    exit 2
fi
PROJECT_DIR="${global_storage:?global_storage is not set}/learning-pdtsp"
RESULT_ROOT="${global_scratch:?global_scratch is not set}/learning-pdtsp/fcfs/${ARRAY_JOB_ID}"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
exec >>"${LOG_ROOT}/fcfs.${ARRAY_JOB_ID}_${TASK_ID}.log" 2>&1
cd "$PROJECT_DIR"

PROBLEMS=(pdtrptw pdctrptw)
DATASET_DIRS=(pdtrptw pdcvrptw)
SIZES=(50 100)
DODS=(0.2 0.5 0.8)

PROBLEM_ID=$((TASK_ID / 18))
WITHIN_PROBLEM=$((TASK_ID % 18))
DISTRIBUTION_ID=$((WITHIN_PROBLEM / 6))
SIZE_ID=$(((WITHIN_PROBLEM % 6) / 3))
DOD_ID=$((WITHIN_PROBLEM % 3))

PROBLEM="${PROBLEMS[$PROBLEM_ID]}"
DATASET_DIR="${DATASET_DIRS[$PROBLEM_ID]}"
SIZE="${SIZES[$SIZE_ID]}"
DOD="${DODS[$DOD_ID]}"
MARKER=""
DISTRIBUTION="uniform"
ORTEC_ARGS=()
if [[ "$DISTRIBUTION_ID" -eq 1 ]]; then
    MARKER="_skew-st"
    DISTRIBUTION="skew-spatio-temporal"
elif [[ "$DISTRIBUTION_ID" -eq 2 ]]; then
    MARKER="_ortec"
    DISTRIBUTION="real-world"
    ORTEC_FILE="new_data/ortec_rl/ORTEC-VRPTW-ASYM-00c5356f-d1-n258-k12.txt"
    [[ -s "$ORTEC_FILE" ]] || { echo "Missing $ORTEC_FILE" >&2; exit 1; }
    ORTEC_ARGS+=(--use-ortec "$ORTEC_FILE")
fi

DATASET="new_data/${DATASET_DIR}/${DATASET_DIR}${SIZE}dod${DOD}v4th8_testing${MARKER}_ortools.txt"
OUTPUT="${RESULT_ROOT}/results/${PROBLEM}/${SIZE}/dod-${DOD}/${DISTRIBUTION}"
[[ -s "$DATASET" ]] || { echo "Missing $DATASET" >&2; exit 1; }

python -u evaluate_model_free.py \
    --problem "$PROBLEM" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --num-samples "$NUM_SAMPLES" \
    --method first_come_first_served \
    "${ORTEC_ARGS[@]}"

[[ -s "${OUTPUT}/summary.json" ]] || { echo "Missing summary" >&2; exit 1; }
grep -q '"method": "first_come_first_served"' "${OUTPUT}/summary.json"
echo "FCFS EVALUATION PASSED"
