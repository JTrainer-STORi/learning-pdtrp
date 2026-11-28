#!/bin/bash
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --job-name=revealed-continue
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=revealed-continue.%A_%a.out

source /etc/profile
set -eo pipefail
module add miniforge
cd "$global_storage/learning-pdtrp"
source .venv/bin/activate
set -u
shopt -s nullglob

ARM=revealed
N_EPOCHS=60
SEED=$SLURM_ARRAY_TASK_ID
DATA_SEED=26413
ROOT="$PWD/artifacts/revealed-ppo"
DATA="$ROOT/data/$DATA_SEED"
CASE="$ROOT/continuation/$SLURM_ARRAY_JOB_ID/seed-$SEED"
ARCHIVE="$ROOT/baseline/26513824/seed-$SEED/current-revealed"
REFERENCE_ARRAYS=(26338569 26342957 26342957)
EARLY=("$ARCHIVE"/models/*/*/best-validation.pth)
RESUME=("$ARCHIVE"/models/*/*/epoch-39.pt)
FULL=("$ROOT"/training/26442619/full/seed-"$SEED"/models/*/*/best-validation.pth)
REFERENCE=("$PWD"/artifacts/training-finetune/"${REFERENCE_ARRAYS[$SEED]}"/pdctrptw-seed"$SEED"-capacity-aware/models/*/*/best-validation.pth)

(( ${#EARLY[@]} == 1 && ${#RESUME[@]} == 1 && ${#FULL[@]} == 1 && ${#REFERENCE[@]} == 1 )) || {
    echo "Expected one 40-epoch checkpoint pair in $ARCHIVE and one checkpoint for each greedy benchmark" >&2
    exit 1
}
(cd "$DATA" && sha256sum -c evaluation-inputs.sha256)
mkdir -p "$CASE"
sha256sum "${RESUME[0]}" > "$CASE/resume.sha256"
source experiments/revealed-ppo/run-training.sh 0 current-revealed --resume "${RESUME[0]}"

LATE=("$CASE"/models/*/*/best-validation.pth)
(( ${#LATE[@]} == 1 )) || { echo "Expected one continued model in $CASE" >&2; exit 1; }
# Use the same selection rule across epochs 0–99. Ties keep the earlier checkpoint.
BEST=$(python - "${EARLY[0]}" "${LATE[0]}" <<'PY'
import json
import sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:]]
scores = [json.loads(p.with_suffix(".json").read_text())["mean_validation_objective"] for p in paths]
print(paths[1] if scores[1] < scores[0] else paths[0])
PY
)

source experiments/revealed-ppo/evaluate-greedy.sh
evaluate_greedy_case "${EARLY[0]}" "$DATA" "$CASE/raw-40" "$DATA_SEED"
evaluate_greedy_case "$BEST" "$DATA" "$CASE/raw-100" "$DATA_SEED"
evaluate_greedy_case "${FULL[0]}" "$DATA" "$CASE/full-ppo" "$DATA_SEED"
evaluate_greedy_case "${REFERENCE[0]}" "$DATA" "$CASE/reference" "$DATA_SEED"
