#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 4 ]]; then
    echo "Usage: scripts/evaluate-policy.sh PROBLEM CHECKPOINT DATASET OUTPUT [-- EXTRA_ARGS...]"
    echo "Environment: DEVICE=auto|cpu|cuda, ORTEC_FILE=/path/file, EVALUATION_SEED=1234"
    [[ $# -ge 1 ]] && exit 0 || exit 2
fi

PROBLEM=$1
CHECKPOINT=$2
DATASET=$3
OUTPUT=$4
shift 4
[[ ${1:-} == "--" ]] && shift

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[[ -f "$DATASET" ]] || { echo "Dataset not found: $DATASET" >&2; exit 1; }

DEVICE=${DEVICE:-auto}
if [[ "$DEVICE" == auto ]]; then
    DEVICE=$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')
fi

ORTec_ARGS=()
[[ -z ${ORTEC_FILE:-} ]] || ORTec_ARGS+=(--use-ortec "$ORTEC_FILE")

python -u evaluate_trained_policy.py --problem "$PROBLEM" --checkpoint "$CHECKPOINT" \
    --datasets "$DATASET" --output-dir "$OUTPUT" --device "$DEVICE" \
    --evaluation-seed "${EVALUATION_SEED:-1234}" "${ORTec_ARGS[@]}" "$@"
