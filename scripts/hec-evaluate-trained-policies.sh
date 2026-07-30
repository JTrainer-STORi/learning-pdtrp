#!/usr/bin/env bash
# Evaluate completed tasks from scripts/hec-retrain-main-table.sh.
#
# The evaluation array uses exactly the same task IDs as training. For example:
#   sbatch --array=0,1,4,8 scripts/hec-evaluate-trained-policies.sh
#   sbatch --array=0-11,24-29 scripts/hec-evaluate-trained-policies.sh
#
# A task is evaluated only if its final epoch-99.pt checkpoint is present.
#
#SBATCH --partition=gpu-short
#SBATCH --job-name=pdtsp-policy-eval
#SBATCH --array=0-38
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=policy-eval.%A_%a.out
#SBATCH --error=policy-eval.%A_%a.err

source /etc/profile

set -eo pipefail

module add miniforge

ENVIRONMENT_PATH="${global_storage:?global_storage is not set}/learning-tsp/tsp-new"
source activate "$ENVIRONMENT_PATH"

set -u

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be run as a Slurm array job}"
if (( TASK_ID < 0 || TASK_ID > 38 )); then
    echo "Task ID must be between 0 and 38, got ${TASK_ID}" >&2
    exit 2
fi

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}}"
SCRATCH_ROOT="${global_scratch:?global_scratch is not set}/learning-pdtsp"
MODEL_ROOT="${MODEL_ROOT:-${SCRATCH_ROOT}/models}"
RESULT_ROOT="${SCRATCH_ROOT}/policy-evaluation/${ARRAY_JOB_ID}"
RUN_ROOT="${RESULT_ROOT}/task-${TASK_ID}"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
RUN_LOG="${LOG_ROOT}/policy-eval.${ARRAY_JOB_ID}_${TASK_ID}.log"
exec >>"$RUN_LOG" 2>&1

PROJECT_DIR="${global_storage:?global_storage is not set}/learning-pdtsp"
cd "$PROJECT_DIR"

SEED=$((TASK_ID % 3))
CONFIG_ID=$((TASK_ID / 3))

PROBLEMS=(
    pdtrp pdtrp pdctrp pdctrp pdtrptw pdtrptw pdctrptw pdctrptw
    pdtrp pdctrp pdtrptw pdctrptw
    pdtrp
)
DATASET_DIRS=(
    pdtrp pdtrp pdcvrp pdcvrp pdtrptw pdtrptw pdcvrptw pdcvrptw
    pdtrp pdcvrp pdtrptw pdcvrptw
    pdtrp
)
MAX_TOTALS=(
    50 100 50 100 50 100 50 100
    50 50 50 50
    50
)
TRAINING_DISTRIBUTIONS=(
    uniform uniform uniform uniform uniform uniform uniform uniform
    real-world real-world real-world real-world
    joshi
)

PROBLEM="${PROBLEMS[$CONFIG_ID]}"
DATASET_DIR="${DATASET_DIRS[$CONFIG_ID]}"
MAX_TOTAL="${MAX_TOTALS[$CONFIG_ID]}"
TRAINING_DISTRIBUTION="${TRAINING_DISTRIBUTIONS[$CONFIG_ID]}"
RUN_PREFIX="corrected-${TRAINING_DISTRIBUTION}-${PROBLEM}-20-${MAX_TOTAL}-seed-${SEED}_"

# A retried task may leave more than one timestamped directory. Select the
# newest complete final checkpoint and report the choice in the manifest/log.
mapfile -t CHECKPOINTS < <(
    find "$MODEL_ROOT" -type f -path "*/${RUN_PREFIX}*/epoch-99.pt" \
        -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-
)
if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo "No completed checkpoint found for task ${TASK_ID}" >&2
    echo "Expected ${MODEL_ROOT}/*/${RUN_PREFIX}*/epoch-99.pt" >&2
    exit 1
fi
CHECKPOINT="${CHECKPOINTS[0]}"
if (( ${#CHECKPOINTS[@]} > 1 )); then
    echo "WARNING: found ${#CHECKPOINTS[@]} completed checkpoints; using newest:"
fi
echo "Checkpoint: ${CHECKPOINT}"

DATASET_ROOT="${PROJECT_DIR}/new_data/${DATASET_DIR}"
DODS=(0.2 0.5 0.8)
TEST_SIZES=(50 100)
DATASETS=()

add_distribution() {
    local suffix="$1"
    local size
    local dod
    local dataset
    for size in "${TEST_SIZES[@]}"; do
        for dod in "${DODS[@]}"; do
            dataset="${DATASET_ROOT}/${DATASET_DIR}${size}dod${dod}v4th8_testing${suffix}_ortools.txt"
            if [[ ! -f "$dataset" ]]; then
                echo "Missing fixed test dataset: ${dataset}" >&2
                exit 1
            fi
            DATASETS+=("$dataset")
        done
    done
}

# Uniform policies are useful on all three test distributions. Policies
# trained on ORTEC and the learned static baseline are evaluated on the table
# rows for which they were trained.
case "$TRAINING_DISTRIBUTION" in
    uniform)
        add_distribution ""
        add_distribution "_skew-st"
        add_distribution "_ortec"
        ;;
    real-world)
        add_distribution "_ortec"
        ;;
    joshi)
        add_distribution ""
        ;;
esac

echo "Starting policy evaluation"
echo "  task_id: ${TASK_ID}"
echo "  config_id: ${CONFIG_ID}"
echo "  problem: ${PROBLEM}"
echo "  training_distribution: ${TRAINING_DISTRIBUTION}"
echo "  max_total: ${MAX_TOTAL}"
echo "  training_seed: ${SEED}"
echo "  datasets: ${#DATASETS[@]}"
echo "  result_dir: ${RUN_ROOT}"
echo "  git_commit: $(git rev-parse HEAD)"
echo "  started_at: $(date --iso-8601=seconds)"
nvidia-smi

ORTEC_FILE="${PROJECT_DIR}/new_data/ortec_rl/ORTEC-VRPTW-ASYM-00c5356f-d1-n258-k12.txt"
[[ -s "$ORTEC_FILE" ]] || { echo "Missing ${ORTEC_FILE}" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=0 python -u evaluate_trained_policy.py \
    --problem "$PROBLEM" \
    --checkpoint "$CHECKPOINT" \
    --datasets "${DATASETS[@]}" \
    --output-dir "$RUN_ROOT" \
    --evaluation-seed 1234 \
    --use-ortec "$ORTEC_FILE" \
    --device cuda

echo "Finished at $(date --iso-8601=seconds)"
