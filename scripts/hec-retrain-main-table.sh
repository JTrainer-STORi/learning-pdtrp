#!/usr/bin/env bash
# Retrain the 39 learned policies needed by both result tables:
#   8 uniform configurations, 4 real-world configurations, and the Joshi
#   baseline, each with 3 random seeds.
#
# Submit from the repository root with:
#   sbatch scripts/hec-retrain-main-table.sh
#
#SBATCH --partition=astro
#SBATCH --job-name=pdtsp-retrain
#SBATCH --array=0-38
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Slurm opens these before the script can expand $global_scratch. The complete
# combined job log is additionally streamed to global scratch below.
#SBATCH --output=retrain-main.%A_%a.out
#SBATCH --error=retrain-main.%A_%a.err

set -euo pipefail

SCRATCH_ROOT="${global_scratch:?global_scratch is not set}/learning-pdtsp"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be run as a Slurm array job}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}}"
MODEL_ROOT="${SCRATCH_ROOT}/models"
LOG_ROOT="${SCRATCH_ROOT}/logs"
SLURM_LOG_ROOT="${SCRATCH_ROOT}/slurm"

mkdir -p "$MODEL_ROOT" "$LOG_ROOT" "$SLURM_LOG_ROOT"
RUN_LOG="${SLURM_LOG_ROOT}/retrain-main.${ARRAY_JOB_ID}_${TASK_ID}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

source /etc/profile
module add miniforge

PROJECT_DIR="${global_storage:?global_storage is not set}/learning-pdtsp"
ENVIRONMENT_PATH="${global_storage}/learning-tsp/tsp-new"

cd "$PROJECT_DIR"
source activate "$ENVIRONMENT_PATH"

SEED=$((TASK_ID % 3))
CONFIG_ID=$((TASK_ID / 3))

# Configurations 0--7 reproduce the uniform-trained rows in both tables.
# Configurations 8--11 reproduce the real-world-trained rows. Configuration
# 12 is the learned static PDTRP baseline labelled "Joshi".
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
MIN_TOTAL=20
MIN_DOD=0.2
MAX_DOD=0.8

if [[ "$TRAINING_DISTRIBUTION" == "joshi" ]]; then
    MIN_DOD=0.0
    MAX_DOD=0.0
fi

RUN_NAME="corrected-${TRAINING_DISTRIBUTION}-${PROBLEM}-20-${MAX_TOTAL}-seed-${SEED}"
OUTPUT_DIR="$MODEL_ROOT"

VALIDATION_SUFFIX="v4th8_val_ortools.txt"
if [[ "$TRAINING_DISTRIBUTION" == "real-world" ]]; then
    VALIDATION_SUFFIX="v4th8ortec_val_ortools.txt"
fi

VAL_DATASETS=(
    "new_data/${DATASET_DIR}/${DATASET_DIR}50dod0.2${VALIDATION_SUFFIX}"
    "new_data/${DATASET_DIR}/${DATASET_DIR}50dod0.5${VALIDATION_SUFFIX}"
    "new_data/${DATASET_DIR}/${DATASET_DIR}50dod0.8${VALIDATION_SUFFIX}"
)

for dataset in "${VAL_DATASETS[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Missing validation dataset: $dataset" >&2
        exit 1
    fi
done

# Preserve the final historical training configurations where they affect
# learning. Validation is less frequent and videos are disabled because
# neither operation changes policy parameters.
N_EPOCHS=100
EPOCH_SIZE=12800
BATCH_SIZE=256
VAL_SIZE=128
VAL_BATCH_SIZE=1
ROLLOUT_SIZE=1280
ROLLOUT_BATCH_SIZE=256

# The original PDTRP 20--50 policies used smaller validation/rollout batches.
if [[ "$PROBLEM" == "pdtrp" && "$MAX_TOTAL" -eq 50 ]]; then
    VAL_SIZE=64
    ROLLOUT_BATCH_SIZE=128
fi

PROBLEM_ARGS=()
if [[ "$TRAINING_DISTRIBUTION" == "real-world" ]]; then
    ORTEC_FILE="new_data/ortec_rl/ORTEC-VRPTW-ASYM-00c5356f-d1-n258-k12.txt"
    if [[ ! -f "$ORTEC_FILE" ]]; then
        echo "Missing real-world training instance: $ORTEC_FILE" >&2
        exit 1
    fi
    PROBLEM_ARGS+=(--use_ortec "$ORTEC_FILE")
fi
if [[ "$PROBLEM" == "pdtrptw" || "$PROBLEM" == "pdctrptw" ]]; then
    PROBLEM_ARGS+=(
        --use_time_feature
        --scale_times
        --use_arrival_times
        --use_arrival_lstm
        --gamma 1.0
        --theta 1.0
    )
fi
if [[ "$PROBLEM" == "pdctrp" || "$PROBLEM" == "pdctrptw" ]]; then
    PROBLEM_ARGS+=(
        --vehicle_capacity 1.0
        --min_trips_required_lb 3
        --min_trips_required_ub 9
    )
fi

echo "Starting corrected training run"
echo "  task_id:     $TASK_ID"
echo "  problem:     $PROBLEM"
echo "  distribution:$TRAINING_DISTRIBUTION"
echo "  train_range: ${MIN_TOTAL}-${MAX_TOTAL}"
echo "  dod_range:   ${MIN_DOD}-${MAX_DOD}"
echo "  seed:        $SEED"
echo "  run_name:    $RUN_NAME"
echo "  model_root:  $MODEL_ROOT"
echo "  log_root:    $LOG_ROOT"
echo "  job_log:     $RUN_LOG"
echo "  git_commit:  $(git rev-parse HEAD)"
echo "  host:        $(hostname)"
echo "  started_at:  $(date --iso-8601=seconds)"
nvidia-smi

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --problem "$PROBLEM" \
    --model dattention \
    --encoder gnn \
    --baseline rollout \
    --min_total "$MIN_TOTAL" \
    --max_total "$MAX_TOTAL" \
    --min_dod "$MIN_DOD" \
    --max_dod "$MAX_DOD" \
    --speed 4 \
    --time_horizon 8 \
    --dimension_scale 1.0 \
    --n_subregions 1 \
    --neighbors 0.2 \
    --knn_strat percentage \
    --val_datasets "${VAL_DATASETS[@]}" \
    --n_epochs "$N_EPOCHS" \
    --epoch_size "$EPOCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --accumulation_steps 1 \
    --val_size "$VAL_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --val_every 5 \
    --rollout_size "$ROLLOUT_SIZE" \
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
    --baseline_update_every 1 \
    --aggregation max \
    --aggregation_graph mean \
    --normalization batch \
    --learn_norm \
    --gated \
    --embedding_dim 128 \
    --hidden_dim 128 \
    --n_encode_layers 3 \
    --edge_features adjacency \
    --lr_model 0.0001 \
    --max_grad_norm 1 \
    --seed "$SEED" \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --checkpoint_epochs 10 \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_ROOT" \
    --no_tensorboard \
    --no_videos \
    "${PROBLEM_ARGS[@]}"

echo "Finished at $(date --iso-8601=seconds)"
