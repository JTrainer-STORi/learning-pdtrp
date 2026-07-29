#!/usr/bin/env bash
# Corrected rolling and clairvoyant OR-Tools baselines:
# 4 problems x 3 distributions x 2 sizes x 3 DoD.
#SBATCH --partition=astro
#SBATCH --job-name=pdtsp-ortools-fixed
#SBATCH --array=0-71
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=pdtsp-ortools-fixed.%A_%a.out
#SBATCH --error=pdtsp-ortools-fixed.%A_%a.err

source /etc/profile
set -eo pipefail
module add miniforge
ENVIRONMENT_PATH="${global_storage:?global_storage is not set}/learning-tsp/tsp-new"
source activate "$ENVIRONMENT_PATH"
set -u

TASK_ID="${SLURM_ARRAY_TASK_ID:?This must run as an array job}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}}"
PROJECT_DIR="${global_storage}/learning-pdtsp"
RESULT_ROOT="${global_scratch:?global_scratch is not set}/learning-pdtsp/ortools-corrected/${ARRAY_JOB_ID}"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/results"
exec >>"${RESULT_ROOT}/logs/ortools.${ARRAY_JOB_ID}_${TASK_ID}.log" 2>&1
cd "$PROJECT_DIR"

PROBLEMS=(pdtrp pdctrp pdtrptw pdctrptw)
DATASET_DIRS=(pdtrp pdcvrp pdtrptw pdcvrptw)
SIZES=(50 100)
DODS=(0.2 0.5 0.8)

PROBLEM_ID=$((TASK_ID / 18))
WITHIN_PROBLEM=$((TASK_ID % 18))
DISTRIBUTION_ID=$((WITHIN_PROBLEM / 6))
WITHIN_DISTRIBUTION=$((WITHIN_PROBLEM % 6))
SIZE_ID=$((WITHIN_DISTRIBUTION / 3))
DOD_ID=$((WITHIN_DISTRIBUTION % 3))

PROBLEM="${PROBLEMS[$PROBLEM_ID]}"
DATASET_DIR="${DATASET_DIRS[$PROBLEM_ID]}"
SIZE="${SIZES[$SIZE_ID]}"
DOD="${DODS[$DOD_ID]}"
MARKER=""
DISTRIBUTION="uniform"
EXTRA_ARGS=()
if [[ "$DISTRIBUTION_ID" -eq 1 ]]; then
    MARKER="_skew-st"
    DISTRIBUTION="skew-spatio-temporal"
elif [[ "$DISTRIBUTION_ID" -eq 2 ]]; then
    MARKER="_ortec"
    DISTRIBUTION="real-world"
    ORTEC_FILE="new_data/ortec_rl/ORTEC-VRPTW-ASYM-00c5356f-d1-n258-k12.txt"
    [[ -s "$ORTEC_FILE" ]] || { echo "Missing $ORTEC_FILE" >&2; exit 1; }
    EXTRA_ARGS+=(--use-ortec "$ORTEC_FILE")
fi

DATASET="new_data/${DATASET_DIR}/${DATASET_DIR}${SIZE}dod${DOD}v4th8_testing${MARKER}_ortools.txt"
OUTPUT="${RESULT_ROOT}/results/${PROBLEM}/${SIZE}/dod-${DOD}/${DISTRIBUTION}"
[[ -s "$DATASET" ]] || { echo "Missing $DATASET" >&2; exit 1; }

python -u evaluate_ortools_baselines.py \
    --problem "$PROBLEM" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --num-samples 100 \
    --online-time-ms 250 \
    --static-time-ms 3000 \
    --max-trips 9 \
    "${EXTRA_ARGS[@]}"

[[ -s "${OUTPUT}/summary.json" ]] || { echo "Missing summary" >&2; exit 1; }
echo "CORRECTED ORTOOLS EVALUATION PASSED"
