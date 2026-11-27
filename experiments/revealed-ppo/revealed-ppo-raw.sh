#!/bin/bash
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --job-name=revealed-raw
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=revealed-raw.%A_%a.out

source /etc/profile
set -eo pipefail
module add miniforge
cd "$global_storage/learning-pdtrp"
source .venv/bin/activate
set -u
shopt -s nullglob

ARM=revealed
SEED=$SLURM_ARRAY_TASK_ID
N_EPOCHS=200
ROOT="$PWD/artifacts/revealed-ppo"
CASE="$ROOT/raw/$SLURM_ARRAY_JOB_ID/seed-$SEED"
if (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
    CASE="$CASE/restart-$SLURM_RESTART_COUNT"
fi
DATA="$ROOT/data/26412"
CONTROL="$ROOT/evaluation/26469992/revealed/seed-$SEED"
LOCATIONS=(uniform-uniform ortec-locations)
DODS=(0.2 0.5 0.8)

mkdir -p "$CASE"
(cd "$DATA" && sha256sum -c evaluation-inputs.sha256)
cmp "$DATA/evaluation-inputs.sha256" "$CONTROL/evaluation-inputs.sha256"
cp "$DATA/evaluation-inputs.sha256" "$CASE/"

# Train from scratch with raw action scores throughout collection, replay and validation.
source experiments/revealed-ppo/run-training.sh 0
CHECKPOINTS=("$CASE"/models/*/*/best-validation.pth)
if (( ${#CHECKPOINTS[@]} != 1 )); then
    echo "Expected one best-validation checkpoint in $CASE" >&2
    exit 1
fi
sha256sum "${CHECKPOINTS[0]}" > "$CASE/checkpoint.sha256"

export OMP_NUM_THREADS=2  # Match the archived greedy evaluation.
for LOCATION in "${LOCATIONS[@]}"; do
    DATASETS=()
    for DOD in "${DODS[@]}"; do
        DATASETS+=("$DATA/$LOCATION/pdctrptw50-dod$DOD-th10-seed26412.txt")
    done
    srun --cpus-per-task=2 --export=ALL --cpu-bind=cores python -u evaluate_trained_policy.py --problem pdctrptw \
        --checkpoint "${CHECKPOINTS[0]}" --datasets "${DATASETS[@]}" --num-samples 300 \
        --speed 4 --time-horizon 10 --latest-end 2 --evaluation-seed 12346 --device cuda \
        --record-decision-times --rollout-candidates 0 --output-dir "$CASE/candidates-0/$LOCATION"
done
python experiments/revealed-ppo/compare_raw.py "$CASE" "$CONTROL" "$SEED"
