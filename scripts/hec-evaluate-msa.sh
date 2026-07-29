#!/usr/bin/env bash
# Uniform, skew spatio-temporal, and ORTEC MSA evaluation:
# 4 problems x 3 distributions x 2 sizes x 3 DoD x 3 sampling seeds.
#SBATCH --partition=astro
#SBATCH --job-name=pdtsp-msa
#SBATCH --array=0-215
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=pdtsp-msa.%A_%a.out
#SBATCH --error=pdtsp-msa.%A_%a.err

source /etc/profile
set -eo pipefail
module add miniforge
ENVIRONMENT_PATH="${global_storage:?global_storage is not set}/learning-tsp/tsp-new"
source activate "$ENVIRONMENT_PATH"
set -u

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be run as an array job}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}}"
PROJECT_DIR="${global_storage}/learning-pdtsp"
RESULT_ROOT="${global_scratch:?global_scratch is not set}/learning-pdtsp/msa/${ARRAY_JOB_ID}"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
exec >>"${LOG_ROOT}/msa.${ARRAY_JOB_ID}_${TASK_ID}.log" 2>&1
cd "$PROJECT_DIR"

PROBLEMS=(pdtrp pdctrp pdtrptw pdctrptw)
DATASET_DIRS=(pdtrp pdcvrp pdtrptw pdcvrptw)
SIZES=(50 100)
DODS=(0.2 0.5 0.8)

SEED=$((TASK_ID % 3))
CONFIG_ID=$((TASK_ID / 3))
PROBLEM_ID=$((CONFIG_ID / 18))
WITHIN_PROBLEM=$((CONFIG_ID % 18))
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
DATASET="new_data/${DATASET_DIR}/${DATASET_DIR}${SIZE}dod${DOD}v4th8_testing${MARKER}_ortools.txt"
OUTPUT="${RESULT_ROOT}/results/${PROBLEM}/${SIZE}/dod-${DOD}/${DISTRIBUTION}/seed-${SEED}.json"

[[ -s "$DATASET" ]] || { echo "Missing $DATASET" >&2; exit 1; }

python -u evaluate_msa_ortools.py \
    --problem "$PROBLEM" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --dod "$DOD" \
    --scenarios 10 \
    --solver-time-ms 50 \
    --num-samples 100 \
    --seed "$SEED" \
    "${SCENARIO_ARGS[@]}"
