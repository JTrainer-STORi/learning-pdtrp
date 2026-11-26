#!/bin/bash
#SBATCH --partition=astro
#SBATCH --gres=gpu:1
#SBATCH --job-name=revealed-eval
#SBATCH --array=1-9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=revealed-eval.%A_%a.out

source /etc/profile
set -eo pipefail
module add miniforge
cd "$global_storage/learning-pdtrp"
source .venv/bin/activate
set -u
shopt -s nullglob

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
ARMS=(full full full revealed revealed revealed reference reference reference)
SEEDS=(0 1 2 0 1 2 0 1 2)
REFERENCE_ARRAYS=(26338569 26342957 26342957)
LOCATIONS=(uniform-uniform ortec-locations)
DODS=(0.2 0.5 0.8)
WIDTHS=(0 2)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
((INDEX >= 0 && INDEX < 9))
ARM=${ARMS[$INDEX]}
SEED=${SEEDS[$INDEX]}
ROOT="$PWD/artifacts/revealed-ppo"
DATA="$ROOT/data/26412"
EVALUATION="$ROOT/evaluation/$SLURM_ARRAY_JOB_ID"
OUT="$EVALUATION/$ARM/seed-$SEED"

(cd "$DATA" && sha256sum -c evaluation-inputs.sha256)
mkdir -p "$OUT"
# All nine tasks keep the same training run, even if another is submitted later.
if [[ ! -L "$EVALUATION/training" ]]; then
    if ! TRAIN_ARRAY=$(readlink "$ROOT/training/production"); then
        echo "Run production training before evaluation." >&2
        exit 1
    fi
    ln -sT "../../training/$TRAIN_ARRAY" "$EVALUATION/training" || [[ -L "$EVALUATION/training" ]]
fi
TRAINING=$(realpath -e "$EVALUATION/training")
if [[ $ARM == reference ]]; then
    CASE="$PWD/artifacts/training-finetune/${REFERENCE_ARRAYS[$SEED]}/pdctrptw-seed$SEED-capacity-aware"
else
    CASE="$TRAINING/$ARM/seed-$SEED"
    if [[ ! -f "$CASE/complete" ]]; then
        echo "Training has not finished successfully in $CASE" >&2
        exit 1
    fi
fi
CHECKPOINTS=("$CASE"/models/*/*/best-validation.pth)
if ((${#CHECKPOINTS[@]} != 1)); then
    echo "Expected one best-validation checkpoint in $CASE" >&2
    exit 1
fi
cp "$DATA/evaluation-inputs.sha256" "$OUT/"
sha256sum "${CHECKPOINTS[0]}" > "$OUT/checkpoint.sha256"
git rev-parse HEAD > "$OUT/git-revision.txt"

for LOCATION in "${LOCATIONS[@]}"; do
    DATASETS=()
    for DOD in "${DODS[@]}"; do
        DATASETS+=("$DATA/$LOCATION/pdctrptw50-dod$DOD-th10-seed26412.txt")
    done
    for WIDTH in "${WIDTHS[@]}"; do
        srun --export=ALL --cpu-bind=cores python -u evaluate_trained_policy.py --problem pdctrptw \
            --checkpoint "${CHECKPOINTS[0]}" --datasets "${DATASETS[@]}" --num-samples 300 \
            --speed 4 --time-horizon 10 --latest-end 2 --evaluation-seed 12346 --device cuda \
            --record-decision-times --rollout-candidates "$WIDTH" --rollout-wait-penalty 0 \
            --output-dir "$OUT/candidates-$WIDTH/$LOCATION"
    done
done
