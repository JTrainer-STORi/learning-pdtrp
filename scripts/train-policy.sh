#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/train-policy.sh PROBLEM [MAX_CUSTOMERS] [SEED] [-- EXTRA_ARGS...]

PROBLEM is one of: pdtrp, pdctrp, pdtrptw, pdctrptw.

Environment variables:
  DEVICE=auto|cpu|GPU_INDEX  Execution device (default: auto)
  MIN_CUSTOMERS=20           Minimum N, excluding the depot
  MIN_DOD=0.2 / MAX_DOD=0.8  Training degree-of-dynamism range
  HOURS_PER_CUSTOMER=0.2     Call horizon T=N/5 hours (explicit time flags override)
  ARRIVAL_LSTM=auto|on|off   Auto enables it for TW variants, matching earlier runs
  VALIDATION_SEED=12345      Common generated held-out data across training seeds
  N_EPOCHS=100               Training epochs
  EPOCH_SIZE=12800           Instances generated per epoch
  BATCH_SIZE=256             Training batch size
  NUM_WORKERS=0              DataLoader workers
  OUTPUT_DIR=artifacts/models Model output directory
  LOG_DIR=artifacts/logs      Log directory
  ENABLE_WANDB=0             Set to 1 to enable Weights & Biases
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 1 ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0 || exit 2
fi

PROBLEM=$1
MAX_CUSTOMERS=${2:-50}
SEED=${3:-0}
shift $(( $# >= 3 ? 3 : $# ))
[[ ${1:-} == "--" ]] && shift

case "$PROBLEM" in
    pdtrp|pdctrp|pdtrptw|pdctrptw) ;;
    *) echo "Unsupported problem: $PROBLEM" >&2; usage >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

MIN_CUSTOMERS=${MIN_CUSTOMERS:-20}
MIN_DOD=${MIN_DOD:-0.2}
MAX_DOD=${MAX_DOD:-0.8}
N_EPOCHS=${N_EPOCHS:-100}
EPOCH_SIZE=${EPOCH_SIZE:-12800}
BATCH_SIZE=${BATCH_SIZE:-256}
VAL_SIZE=${VAL_SIZE:-128}
ROLLOUT_SIZE=${ROLLOUT_SIZE:-1280}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
OUTPUT_DIR=${OUTPUT_DIR:-artifacts/models}
LOG_DIR=${LOG_DIR:-artifacts/logs}
DEVICE=${DEVICE:-auto}
ENABLE_WANDB=${ENABLE_WANDB:-0}
HOURS_PER_CUSTOMER=${HOURS_PER_CUSTOMER:-0.2}
ARRIVAL_LSTM=${ARRIVAL_LSTM:-auto}
VALIDATION_SEED=${VALIDATION_SEED:-12345}

HORIZON_ARGS=(--time-horizon-per-customer "$HOURS_PER_CUSTOMER")
# An explicit CLI horizon selects either fixed or scaled generation.
for ARGUMENT in "$@"; do
    case "$ARGUMENT" in
        --time-horizon*|--time_horizon*) HORIZON_ARGS=(); break ;;
    esac
done

DEVICE_ARGS=()
if [[ "$DEVICE" == "cpu" ]]; then
    DEVICE_ARGS+=(--no_cuda)
elif [[ "$DEVICE" != "auto" ]]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
fi

LOGGING_ARGS=(--no_tensorboard --no_videos)
[[ "$ENABLE_WANDB" == 1 ]] || LOGGING_ARGS+=(--no_wandb)

PROBLEM_ARGS=()
if [[ "$PROBLEM" == pdtrptw || "$PROBLEM" == pdctrptw ]]; then
    PROBLEM_ARGS+=(--use_time_feature --scale_times --use_arrival_times --gamma 1 --theta 1)
fi
case "$ARRIVAL_LSTM" in
    on) PROBLEM_ARGS+=(--use_arrival_lstm) ;;
    off) ;;
    auto)
        if [[ "$PROBLEM" == *tw ]]; then PROBLEM_ARGS+=(--use_arrival_lstm); fi
        ;;
    *) echo "ARRIVAL_LSTM must be auto, on or off" >&2; exit 2 ;;
esac
if [[ "$PROBLEM" == pdctrp || "$PROBLEM" == pdctrptw ]]; then
    PROBLEM_ARGS+=(--vehicle_capacity 1 --min_trips_required_lb 3 --min_trips_required_ub 9)
fi

python -u run.py \
    --problem "$PROBLEM" \
    --model dattention \
    --encoder gnn \
    --baseline rollout \
    --min-customers "$MIN_CUSTOMERS" \
    --max-customers "$MAX_CUSTOMERS" \
    --min_dod "$MIN_DOD" \
    --max_dod "$MAX_DOD" \
    "${HORIZON_ARGS[@]}" \
    --validation-seed "$VALIDATION_SEED" \
    --neighbors 0.2 \
    --knn_strat percentage \
    --n_subregions 1 \
    --n_epochs "$N_EPOCHS" \
    --epoch_size "$EPOCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --val_size "$VAL_SIZE" \
    --rollout_size "$ROLLOUT_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
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
    --checkpoint_epochs 10 \
    --run_name "${PROBLEM}-${MIN_CUSTOMERS}-${MAX_CUSTOMERS}-seed-${SEED}" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    "${DEVICE_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${PROBLEM_ARGS[@]}" \
    "$@"
