#!/usr/bin/env bash
# End-to-end HEC smoke test for every distinct training path used by the
# reproduction array:
#   4 uniform environments + 4 ORTEC environments + 1 static Joshi baseline.
#
# Submit from the repository root:
#   sbatch scripts/hec-smoke-test.sh
#
# All nine tasks should finish successfully and print "SMOKE TEST PASSED".
#
#SBATCH --partition=astro
#SBATCH --job-name=pdtsp-smoke
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pdtsp-smoke.%A_%a.out
#SBATCH --error=pdtsp-smoke.%A_%a.err

source /etc/profile

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be run as a Slurm array job}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}}"
SMOKE_ROOT="${global_scratch:?global_scratch is not set}/learning-pdtsp/smoke/${ARRAY_JOB_ID}"
MODEL_ROOT="${SMOKE_ROOT}/models"
FRAMEWORK_LOG_ROOT="${SMOKE_ROOT}/logs"
SLURM_LOG_ROOT="${SMOKE_ROOT}/slurm"

mkdir -p "$MODEL_ROOT" "$FRAMEWORK_LOG_ROOT" "$SLURM_LOG_ROOT"
RUN_LOG="${SLURM_LOG_ROOT}/pdtsp-smoke.${ARRAY_JOB_ID}_${TASK_ID}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

module add miniforge

PROJECT_DIR="${global_storage:?global_storage is not set}/learning-pdtsp"
ENVIRONMENT_PATH="${global_storage}/learning-tsp/tsp-new"

cd "$PROJECT_DIR"
source activate "$ENVIRONMENT_PATH"

PROBLEMS=(
    pdtrp pdctrp pdtrptw pdctrptw
    pdtrp pdctrp pdtrptw pdctrptw
    pdtrp
)
DATASET_DIRS=(
    pdtrp pdcvrp pdtrptw pdcvrptw
    pdtrp pdcvrp pdtrptw pdcvrptw
    pdtrp
)
TRAINING_DISTRIBUTIONS=(
    uniform uniform uniform uniform
    real-world real-world real-world real-world
    joshi
)

PROBLEM="${PROBLEMS[$TASK_ID]}"
DATASET_DIR="${DATASET_DIRS[$TASK_ID]}"
TRAINING_DISTRIBUTION="${TRAINING_DISTRIBUTIONS[$TASK_ID]}"
MIN_DOD=0.2
MAX_DOD=0.8
VALIDATION_SUFFIX="v4th8_val_ortools.txt"

if [[ "$TRAINING_DISTRIBUTION" == "joshi" ]]; then
    MIN_DOD=0.0
    MAX_DOD=0.0
elif [[ "$TRAINING_DISTRIBUTION" == "real-world" ]]; then
    VALIDATION_SUFFIX="v4th8ortec_val_ortools.txt"
fi

VALIDATION_DATASET="new_data/${DATASET_DIR}/${DATASET_DIR}50dod0.5${VALIDATION_SUFFIX}"
if [[ ! -f "$VALIDATION_DATASET" ]]; then
    echo "Missing validation dataset: $VALIDATION_DATASET" >&2
    exit 1
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

RUN_NAME="smoke-${TRAINING_DISTRIBUTION}-${PROBLEM}-task-${TASK_ID}"

echo "Starting HEC smoke test"
echo "  task_id:      $TASK_ID"
echo "  problem:      $PROBLEM"
echo "  distribution: $TRAINING_DISTRIBUTION"
echo "  validation:   $VALIDATION_DATASET"
echo "  scratch_root: $SMOKE_ROOT"
echo "  git_commit:   $(git rev-parse HEAD)"
echo "  host:         $(hostname)"
echo "  started_at:   $(date --iso-8601=seconds)"
nvidia-smi

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --problem "$PROBLEM" \
    --model dattention \
    --encoder gnn \
    --baseline rollout \
    --min_total 20 \
    --max_total 50 \
    --min_dod "$MIN_DOD" \
    --max_dod "$MAX_DOD" \
    --speed 4 \
    --time_horizon 8 \
    --dimension_scale 1.0 \
    --n_subregions 1 \
    --neighbors 0.2 \
    --knn_strat percentage \
    --val_datasets "$VALIDATION_DATASET" \
    --n_epochs 1 \
    --epoch_size 64 \
    --batch_size 16 \
    --accumulation_steps 1 \
    --val_size 2 \
    --val_batch_size 1 \
    --val_every 1 \
    --rollout_size 32 \
    --rollout_batch_size 16 \
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
    --seed 0 \
    --num_workers "${SLURM_CPUS_PER_TASK:-2}" \
    --checkpoint_epochs 1 \
    --run_name "$RUN_NAME" \
    --output_dir "$MODEL_ROOT" \
    --log_dir "$FRAMEWORK_LOG_ROOT" \
    --no_tensorboard \
    --no_wandb \
    --no_videos \
    --no_progress_bar \
    "${PROBLEM_ARGS[@]}"

SAVE_DIR_PATTERN="${MODEL_ROOT}/${PROBLEM}_20-50-${MIN_DOD}-${MAX_DOD}/${RUN_NAME}_"*
ARGS_FILE=""
CHECKPOINT_FILE=""
for save_dir in $SAVE_DIR_PATTERN; do
    if [[ -d "$save_dir" ]]; then
        ARGS_FILE="${save_dir}/args.json"
        CHECKPOINT_FILE="${save_dir}/epoch-0.pt"
    fi
done

if [[ ! -s "$ARGS_FILE" || ! -s "$CHECKPOINT_FILE" ]]; then
    echo "Smoke run completed but expected artifacts were not created" >&2
    echo "  args:       $ARGS_FILE" >&2
    echo "  checkpoint: $CHECKPOINT_FILE" >&2
    exit 1
fi

echo "  args:       $ARGS_FILE"
echo "  checkpoint: $CHECKPOINT_FILE"
echo "SMOKE TEST PASSED"
