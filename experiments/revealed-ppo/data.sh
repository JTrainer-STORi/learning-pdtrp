#!/bin/bash
#SBATCH --partition=serial
#SBATCH --job-name=revealed-data
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=revealed-data.%j.out

source /etc/profile
set -eo pipefail
module add miniforge
cd "${global_storage:?}/learning-pdtrp"
source .venv/bin/activate
set -u
export OMP_NUM_THREADS=1
ORTEC_SOURCE=${1:-new_data/ortec_rl/ORTEC-VRPTW-ASYM-00c5356f-d1-n258-k12.txt}
[[ -r "$ORTEC_SOURCE" ]] || { echo "Pass the ORTEC coordinate source as the first argument" >&2; exit 1; }
ROOT="$PWD/artifacts/revealed-ppo/data"
mkdir -p "$ROOT"
STAGING=$(mktemp -d "$ROOT/.${SLURM_JOB_ID:?}.XXXXXX")
LOCATIONS=(uniform-uniform ortec-locations)
DODS=(0.2 0.5 0.8)
for LOCATION in "${LOCATIONS[@]}"; do
    SOURCE_ARGS=()
    if [[ $LOCATION == ortec-locations ]]; then SOURCE_ARGS=(--ortec-locations "$ORTEC_SOURCE"); fi
    for DOD in "${DODS[@]}"; do
        python generate_data.py --problem pdctrptw --num-customers 50 --time-horizon 10 --latest-end 2 \
            --num-samples 300 --seed 26412 --dod "$DOD" "${SOURCE_ARGS[@]}" \
            --output "$STAGING/$LOCATION/pdctrptw50-dod$DOD-th10-seed26412.txt"
    done
done
(cd "$STAGING" && sha256sum */*.txt > evaluation-inputs.sha256)
mv -T "$STAGING" "$ROOT/$SLURM_JOB_ID"
