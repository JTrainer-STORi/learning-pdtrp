#!/usr/bin/env bash
# Evaluate OR-Tools stored tours and the nearest-neighbour heuristic for every
# uniform and real-world table test set (4 problems x 2 sizes x 3 DoD x 2).
#SBATCH --partition=astro
#SBATCH --job-name=pdtsp-model-free
#SBATCH --array=0-47
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=pdtsp-model-free.%A_%a.out
#SBATCH --error=pdtsp-model-free.%A_%a.err

source /etc/profile
set -eo pipefail
module add miniforge
export PYTHONNOUSERSITE=1
ENVIRONMENT_PATH="${global_storage:?global_storage is not set}/learning-tsp/tsp-new"
source activate "$ENVIRONMENT_PATH"
set -u

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be run as an array job}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}}"
PROJECT_DIR="${global_storage}/learning-pdtsp"
RESULT_ROOT="${global_scratch:?global_scratch is not set}/learning-pdtsp/model-free/${ARRAY_JOB_ID}"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
exec >>"${LOG_ROOT}/model-free.${ARRAY_JOB_ID}_${TASK_ID}.log" 2>&1
cd "$PROJECT_DIR"

PROBLEMS=(pdtrp pdctrp pdtrptw pdctrptw)
DATASET_DIRS=(pdtrp pdcvrp pdtrptw pdcvrptw)
SIZES=(50 100)
DODS=(0.2 0.5 0.8)

PROBLEM_ID=$((TASK_ID / 12))
WITHIN_PROBLEM=$((TASK_ID % 12))
REAL_WORLD=$((WITHIN_PROBLEM / 6))
SIZE_ID=$(((WITHIN_PROBLEM % 6) / 3))
DOD_ID=$((WITHIN_PROBLEM % 3))

PROBLEM="${PROBLEMS[$PROBLEM_ID]}"
DATASET_DIR="${DATASET_DIRS[$PROBLEM_ID]}"
SIZE="${SIZES[$SIZE_ID]}"
DOD="${DODS[$DOD_ID]}"
MARKER=""
ORTEC_ARGS=()
if [[ "$REAL_WORLD" -eq 1 ]]; then
    MARKER="_ortec"
    ORTEC_FILE="new_data/ortec_rl/ORTEC-VRPTW-ASYM-00c5356f-d1-n258-k12.txt"
    ORTEC_ARGS+=(--use-ortec "$ORTEC_FILE")
fi

BASE="${DATASET_DIR}${SIZE}dod${DOD}v4th8_testing${MARKER}_ortools"
DATASET="new_data/${DATASET_DIR}/${BASE}.txt"
STATIC_DATASET="static_solves/${DATASET_DIR}/${BASE}_static.txt"
OUTPUT="${RESULT_ROOT}/results/${PROBLEM}/${SIZE}/dod-${DOD}/$([[ "$REAL_WORLD" -eq 1 ]] && echo real-world || echo uniform)"

[[ -s "$DATASET" ]] || { echo "Missing $DATASET" >&2; exit 1; }
[[ -s "$STATIC_DATASET" ]] || { echo "Missing $STATIC_DATASET" >&2; exit 1; }

python -u evaluate_model_free.py \
    --problem "$PROBLEM" \
    --dataset "$DATASET" \
    --static-dataset "$STATIC_DATASET" \
    --output "$OUTPUT" \
    "${ORTEC_ARGS[@]}"
