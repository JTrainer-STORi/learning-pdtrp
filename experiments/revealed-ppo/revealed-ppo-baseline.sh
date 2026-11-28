#!/bin/bash
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --job-name=revealed-baseline
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=revealed-baseline.%A_%a.out

source /etc/profile
set -eo pipefail
module add miniforge
cd "$global_storage/learning-pdtrp"
source .venv/bin/activate
set -u
shopt -s nullglob

ARM=revealed
N_EPOCHS=40
SEED=$SLURM_ARRAY_TASK_ID
ROOT="$PWD/artifacts/revealed-ppo"
DATA="$ROOT/data/26412"
ARCHIVE="$ROOT/raw/26479898/seed-$SEED"
TRIAL="$ROOT/baseline/$SLURM_ARRAY_JOB_ID/seed-$SEED"
if (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
    TRIAL="$TRIAL/restart-$SLURM_RESTART_COUNT"
fi
BASELINES=(current-full current-revealed)

source experiments/revealed-ppo/evaluate-greedy.sh
(cd "$DATA" && sha256sum -c evaluation-inputs.sha256)
cmp "$ARCHIVE/evaluation-inputs.sha256" "$DATA/evaluation-inputs.sha256"

# The archived best for each seed was selected within the first 40 epochs.
CASE="$TRIAL/rollout"
mkdir -p "$CASE"
CHECKPOINTS=("$ARCHIVE"/models/*/*/best-validation.pth)
if (( ${#CHECKPOINTS[@]} != 1 )); then
    echo "Expected one archived best-validation checkpoint in $ARCHIVE" >&2
    exit 1
fi
python -c 'import json, sys; epoch = json.load(open(sys.argv[1]))["epoch"]; sys.exit(0 if 0 <= epoch < 40 else "Archived checkpoint outside 40-epoch window")' \
    "$(dirname "${CHECKPOINTS[0]}")/best-validation.json"
sha256sum "${CHECKPOINTS[0]}" > "$CASE/checkpoint.sha256"
cmp <(cut -d' ' -f1 "$CASE/checkpoint.sha256") <(cut -d' ' -f1 "$ARCHIVE/checkpoint.sha256")
cp "$DATA/evaluation-inputs.sha256" "$CASE/"
evaluate_greedy "${CHECKPOINTS[0]}" "$DATA" "$CASE"

# Train each current-actor baseline from scratch and score its best validation checkpoint.
for BASELINE in "${BASELINES[@]}"; do
    CASE="$TRIAL/$BASELINE"
    mkdir -p "$CASE"
    cp "$DATA/evaluation-inputs.sha256" "$CASE/"
    source experiments/revealed-ppo/run-training.sh 0 "$BASELINE"
    CHECKPOINTS=("$CASE"/models/*/*/best-validation.pth)
    if (( ${#CHECKPOINTS[@]} != 1 )); then
        echo "Expected one best-validation checkpoint in $CASE" >&2
        exit 1
    fi
    sha256sum "${CHECKPOINTS[0]}" > "$CASE/checkpoint.sha256"
    evaluate_greedy "${CHECKPOINTS[0]}" "$DATA" "$CASE"
done

python experiments/revealed-ppo/compare_baselines.py "$TRIAL" "$SEED"
