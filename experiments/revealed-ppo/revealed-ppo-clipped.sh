#!/bin/bash
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --job-name=revealed-clipped
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=revealed-clipped.%A_%a.out

source /etc/profile
set -eo pipefail
module add miniforge
cd "$global_storage/learning-pdtrp"
source .venv/bin/activate
set -u
shopt -s nullglob

ARM=revealed
N_EPOCHS=100
SEED=$SLURM_ARRAY_TASK_ID
DATA_SEED=26413
ROOT="$PWD/artifacts/revealed-ppo"
DATA="$ROOT/data/$DATA_SEED"
CASE="$ROOT/clipped/$SLURM_ARRAY_JOB_ID/seed-$SEED"

(cd "$DATA" && sha256sum -c evaluation-inputs.sha256)
mkdir -p "$CASE"
source experiments/revealed-ppo/run-training.sh 10 current-revealed

CHECKPOINTS=("$CASE"/models/*/*/best-validation.pth)
(( ${#CHECKPOINTS[@]} == 1 )) || { echo "Expected one clipped model in $CASE" >&2; exit 1; }
source experiments/revealed-ppo/evaluate-greedy.sh
evaluate_greedy_case "${CHECKPOINTS[0]}" "$DATA" "$CASE/clipped-100" "$DATA_SEED"
