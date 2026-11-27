#!/bin/bash
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --job-name=revealed-ppo
#SBATCH --array=1-6
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=revealed-train.%A_%a.out

source /etc/profile
set -eo pipefail
module add miniforge
cd "$global_storage/learning-pdtrp"
source .venv/bin/activate
set -u

ARMS=(full full full revealed revealed revealed)
SEEDS=(0 1 2 0 1 2)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
ARM=${ARMS[$INDEX]}
SEED=${SEEDS[$INDEX]}
N_EPOCHS=200
TRAINING="$PWD/artifacts/revealed-ppo/training"
CASE="$TRAINING/$SLURM_ARRAY_JOB_ID/$ARM/seed-$SEED"

mkdir -p "$CASE"
if [[ $SLURM_ARRAY_TASK_ID == 1 ]]; then
    ln -sfnT "$SLURM_ARRAY_JOB_ID" "$TRAINING/production"
fi
source experiments/revealed-ppo/run-training.sh 10
touch "$CASE/complete"
