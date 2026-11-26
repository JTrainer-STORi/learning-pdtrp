#!/bin/bash
# From $HOME: sbatch "$global_storage/learning-pdtrp/experiments/revealed-ppo/train.sh"
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --job-name=revealed-ppo
#SBATCH --array=1-6
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=revealed-train.%A_%a.out

SUBMITTED_SCRIPT=$(realpath "${BASH_SOURCE[0]}")
source /etc/profile
set -eo pipefail
module add miniforge
cd "${global_storage:?}/learning-pdtrp"
source .venv/bin/activate
set -u

ARMS=(full full full revealed revealed revealed)
SEEDS=(0 1 2 0 1 2)
INDEX=$((${SLURM_ARRAY_TASK_ID:?} - 1))
((INDEX >= 0 && INDEX < 6))
ARM=${ARMS[$INDEX]}
SEED=${SEEDS[$INDEX]}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" NUM_WORKERS=2
export ENCODER=transformer NORMALIZATION=layer NEIGHBORS=-1 KNN_STRAT=fixed
export MIN_CUSTOMERS=50 HOURS_PER_CUSTOMER=0.2 MIN_DOD=0.2 MAX_DOD=0.8
export TIME_WINDOW_SLACK=on ARRIVAL_LSTM=off VALIDATION_SEED=12345
export N_EPOCHS=200 EPOCH_SIZE=12800 BATCH_SIZE=256 VAL_SIZE=128 VAL_BATCH_SIZE=1
export ROLLOUT_SIZE=1280 ROLLOUT_BATCH_SIZE=256 DEVICE=auto ENABLE_WANDB=0
if [[ ${1:-train} == pilot ]]; then
    export N_EPOCHS=1
elif [[ ${1:-train} != train ]]; then
    echo "Usage: sbatch train.sh [pilot]" >&2
    exit 2
fi
CASE="$PWD/artifacts/revealed-ppo/training/${SLURM_ARRAY_JOB_ID:?}/$ARM/seed-$SEED"
export OUTPUT_DIR="$CASE/models" LOG_DIR="$CASE/logs"
mkdir -p "$CASE"
git rev-parse HEAD > "$CASE/git-revision.txt"
sha256sum nets/dynamic_attention_model.py nets/encoders/transformer_encoder.py \
    train.py run.py options.py scripts/train-policy.sh "$SUBMITTED_SCRIPT" > "$CASE/source.sha256"
nvidia-smi --query-gpu=name,memory.total --format=csv > "$CASE/gpu.csv"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'

if [[ ${1:-train} == pilot ]]; then
    python -m pytest -q tests/test_revealed_ppo.py
fi
srun --export=ALL --cpu-bind=cores bash scripts/train-policy.sh pdctrptw 50 "$SEED" -- \
    --training-algorithm ppo --ppo-update-epochs 4 --ppo-encoding "$ARM" \
    --use-capacity-aware-lookahead --baseline rollout --bl_warmup_epochs 0 \
    --training-data-seed "$((91230 + SEED))" --lr_model 0.0001 \
    --validation-dods 0.2 0.5 0.8 --val_every 2 --save-best-validation --no_progress_bar
