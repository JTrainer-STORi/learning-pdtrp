#!/bin/bash
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --job-name=revealed-pilot
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=revealed-pilot.%j.out

source /etc/profile
set -eo pipefail
module add miniforge
cd "$global_storage/learning-pdtrp"
source .venv/bin/activate
set -u

ARM=revealed
SEED=0
N_EPOCHS=1
CASE="$PWD/artifacts/revealed-ppo/pilot/$SLURM_JOB_ID"

mkdir -p "$CASE"
python -m pytest -q tests/test_revealed_ppo.py
source experiments/revealed-ppo/run-training.sh
cat "$CASE"/models/*/*/history.csv
