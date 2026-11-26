#!/bin/bash
# From $HOME: sbatch evaluate.sh TRAIN_ARRAY_ID DATA_JOB_ID
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
cd "${global_storage:?}/learning-pdtrp"
source .venv/bin/activate
set -u
shopt -s nullglob
[[ $# == 2 && $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ ]] || { echo "Usage: sbatch evaluate.sh TRAIN_ARRAY_ID DATA_JOB_ID" >&2; exit 2; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
ARMS=(full full full revealed revealed revealed reference reference reference)
SEEDS=(0 1 2 0 1 2 0 1 2)
REFERENCE_ARRAYS=(26338569 26342957 26342957)
INDEX=$((${SLURM_ARRAY_TASK_ID:?} - 1))
((INDEX >= 0 && INDEX < 9))
ARM=${ARMS[$INDEX]}
SEED=${SEEDS[$INDEX]}
ROOT="$PWD/artifacts/revealed-ppo"
CASE="$ROOT/training/$1/$ARM/seed-$SEED"
if [[ $ARM == reference ]]; then
    CASE="$PWD/artifacts/training-finetune/${REFERENCE_ARRAYS[$SEED]}/pdctrptw-seed$SEED-capacity-aware"
fi
CHECKPOINTS=("$CASE"/models/*/*/best-validation.pth)
if ((${#CHECKPOINTS[@]} != 1)); then echo "Expected one best-validation checkpoint in $CASE" >&2; exit 1; fi
DATA="$ROOT/data/$2"
(cd "$DATA" && sha256sum -c evaluation-inputs.sha256)
OUT="$ROOT/evaluation/${SLURM_ARRAY_JOB_ID:?}/$ARM/seed-$SEED"
mkdir -p "$OUT"
cp "$DATA/evaluation-inputs.sha256" "$OUT/"
sha256sum "${CHECKPOINTS[0]}" > "$OUT/checkpoint.sha256"
git rev-parse HEAD > "$OUT/git-revision.txt"

LOCATIONS=(uniform-uniform ortec-locations)
DODS=(0.2 0.5 0.8)
for LOCATION in "${LOCATIONS[@]}"; do
    DATASETS=()
    for DOD in "${DODS[@]}"; do DATASETS+=("$DATA/$LOCATION/pdctrptw50-dod$DOD-th10-seed26412.txt"); done
    for WIDTH in 0 2; do
        srun --export=ALL --cpu-bind=cores python -u evaluate_trained_policy.py --problem pdctrptw \
            --checkpoint "${CHECKPOINTS[0]}" --datasets "${DATASETS[@]}" --num-samples 300 \
            --speed 4 --time-horizon 10 --latest-end 2 --evaluation-seed 12346 --device cuda \
            --record-decision-times --rollout-candidates "$WIDTH" --rollout-wait-penalty 0 \
            --output-dir "$OUT/candidates-$WIDTH/$LOCATION"
    done
done
