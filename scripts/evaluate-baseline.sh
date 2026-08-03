#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/evaluate-baseline.sh METHOD PROBLEM DATASET OUTPUT [-- EXTRA_ARGS...]

METHOD is nearest, fcfs, ortools, or msa. For MSA set DOD to the dataset's
degree of dynamism. Set ORTEC_FILE when evaluating an ORTEC-derived dataset.

Environment variables:
  NUM_SAMPLES=100
  DOD=0.5                 Required by msa
  ORTEC_FILE=/path/file   ORTEC source instance when needed
  ONLINE_TIME_MS=250
  STATIC_TIME_MS=3000
  SCENARIOS=10
  SOLVER_TIME_MS=250
  SEED=1234
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 4 ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0 || exit 2
fi

METHOD=$1
PROBLEM=$2
DATASET=$3
OUTPUT=$4
shift 4
[[ ${1:-} == "--" ]] && shift

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

[[ -f "$DATASET" ]] || { echo "Dataset not found: $DATASET" >&2; exit 1; }
NUM_SAMPLES=${NUM_SAMPLES:-100}
ORTec_ARGS=()
[[ -z ${ORTEC_FILE:-} ]] || ORTec_ARGS+=(--use-ortec "$ORTEC_FILE")

case "$METHOD" in
    nearest|fcfs)
        heuristic=nearest_neighbour
        [[ "$METHOD" == fcfs ]] && heuristic=first_come_first_served
        python -u evaluate_model_free.py --problem "$PROBLEM" --dataset "$DATASET" \
            --output "$OUTPUT" --num-samples "$NUM_SAMPLES" --method "$heuristic" \
            "${ORTec_ARGS[@]}" "$@"
        ;;
    ortools)
        python -u evaluate_ortools_baselines.py --problem "$PROBLEM" --dataset "$DATASET" \
            --output "$OUTPUT" --num-samples "$NUM_SAMPLES" \
            --online-time-ms "${ONLINE_TIME_MS:-250}" --static-time-ms "${STATIC_TIME_MS:-3000}" \
            "${ORTec_ARGS[@]}" "$@"
        ;;
    msa)
        : "${DOD:?Set DOD to the dataset degree of dynamism for MSA evaluation}"
        python -u evaluate_msa_ortools.py --problem "$PROBLEM" --dataset "$DATASET" \
            --output "$OUTPUT" --dod "$DOD" --num-samples "$NUM_SAMPLES" \
            --scenarios "${SCENARIOS:-10}" --solver-time-ms "${SOLVER_TIME_MS:-250}" \
            --seed "${SEED:-1234}" "${ORTec_ARGS[@]}" "$@"
        ;;
    *) echo "Unsupported method: $METHOD" >&2; usage >&2; exit 2 ;;
esac
