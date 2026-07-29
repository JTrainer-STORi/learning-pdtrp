#!/usr/bin/env bash
# One-instance, two-scenario MSA smoke test for all three test distributions.
#SBATCH --partition=serial
#SBATCH --job-name=pdtsp-msa-smoke
#SBATCH --array=0-11
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=pdtsp-msa-smoke.%A_%a.out
#SBATCH --error=pdtsp-msa-smoke.%A_%a.err

source /etc/profile
set -eo pipefail
module add miniforge
ENVIRONMENT_PATH="${global_storage:?global_storage is not set}/learning-tsp/tsp-new"
source activate "$ENVIRONMENT_PATH"
set -u

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be run as an array job}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}}"
PROJECT_DIR="${global_storage}/learning-pdtsp"
RESULT_ROOT="${global_scratch:?global_scratch is not set}/learning-pdtsp/msa-smoke/${ARRAY_JOB_ID}"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/results"
exec >>"${RESULT_ROOT}/logs/msa-smoke.${ARRAY_JOB_ID}_${TASK_ID}.log" 2>&1
cd "$PROJECT_DIR"

PROBLEMS=(pdtrp pdctrp pdtrptw pdctrptw)
DATASET_DIRS=(pdtrp pdcvrp pdtrptw pdcvrptw)
DISTRIBUTION_ID=$((TASK_ID / 4))
PROBLEM_ID=$((TASK_ID % 4))
PROBLEM="${PROBLEMS[$PROBLEM_ID]}"
DATASET_DIR="${DATASET_DIRS[$PROBLEM_ID]}"
MARKER=""
DISTRIBUTION="uniform"
SCENARIO_ARGS=()
if [[ "$DISTRIBUTION_ID" -eq 1 ]]; then
    MARKER="_skew-st"
    DISTRIBUTION="skew-spatio-temporal"
    SCENARIO_ARGS+=(
        --n-subregions 4
        --arrival-weights 5 1 5 1
        --arrival-skews early uniform late uniform
    )
elif [[ "$DISTRIBUTION_ID" -eq 2 ]]; then
    MARKER="_ortec"
    DISTRIBUTION="real-world"
    ORTEC_FILE="new_data/ortec_rl/ORTEC-VRPTW-ASYM-00c5356f-d1-n258-k12.txt"
    [[ -s "$ORTEC_FILE" ]] || { echo "Missing $ORTEC_FILE" >&2; exit 1; }
    SCENARIO_ARGS+=(--use-ortec "$ORTEC_FILE")
fi
DATASET="new_data/${DATASET_DIR}/${DATASET_DIR}50dod0.5v4th8_testing${MARKER}_ortools.txt"
OUTPUT="${RESULT_ROOT}/results/${PROBLEM}-${DISTRIBUTION}.json"

python -u evaluate_msa_ortools.py \
    --problem "$PROBLEM" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --dod 0.5 \
    --scenarios 2 \
    --solver-time-ms 10 \
    --num-samples 1 \
    --seed 0 \
    "${SCENARIO_ARGS[@]}"

[[ -s "$OUTPUT" ]] || { echo "Missing result: $OUTPUT" >&2; exit 1; }
echo "MSA SMOKE TEST PASSED"
