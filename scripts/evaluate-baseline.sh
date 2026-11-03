#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/evaluate-baseline.sh METHOD PROBLEM DATASET OUTPUT [-- EXTRA_ARGS...]

METHOD is nearest, fcfs, online, oracle (or clairvoyant), ortools, or msa.
ortools runs both online and oracle for a paired comparison. For MSA set DOD
to the dataset's degree of dynamism. Set ORTEC_FILE for ORTEC-derived datasets.

Environment variables:
  NUM_SAMPLES=100
  DOD=0.5                 Required by msa
  ORTEC_FILE=/path/file   ORTEC source instance when needed
  ONLINE_NO_IMPROVEMENT_MS=60000  Online wait since last strict best improvement
  ONLINE_TIME_MS                 Optional total limit per online solve; unset by default
  STATIC_NO_IMPROVEMENT_MS=60000  Oracle wait since last strict best improvement
  STATIC_TIME_MS                 Optional oracle total time limit; unset by default
  SCENARIOS=10
  SOLVER_NO_IMPROVEMENT_MS=60000  MSA wait since last strict best improvement
  SOLVER_TIME_MS                 Optional total limit per scenario; unset by default
  MSA_WORKERS=1                  CPU worker processes; 0 selects available CPUs
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
    online)
        ONLINE_ARGS=(--online-no-improvement-ms "${ONLINE_NO_IMPROVEMENT_MS:-60000}")
        [[ -z ${ONLINE_TIME_MS:-} ]] || ONLINE_ARGS+=(--online-time-ms "$ONLINE_TIME_MS")
        python -u evaluate_online_ortools.py --problem "$PROBLEM" --dataset "$DATASET" \
            --output "$OUTPUT" --num-samples "$NUM_SAMPLES" \
            "${ONLINE_ARGS[@]}" "${ORTec_ARGS[@]}" "$@"
        ;;
    oracle|clairvoyant)
        STATIC_ARGS=(--static-no-improvement-ms "${STATIC_NO_IMPROVEMENT_MS:-60000}")
        [[ -z ${STATIC_TIME_MS:-} ]] || STATIC_ARGS+=(--static-time-ms "$STATIC_TIME_MS")
        python -u evaluate_clairvoyant_ortools.py --problem "$PROBLEM" --dataset "$DATASET" \
            --output "$OUTPUT" --num-samples "$NUM_SAMPLES" \
            "${STATIC_ARGS[@]}" "${ORTec_ARGS[@]}" "$@"
        ;;
    ortools)
        ONLINE_ARGS=(--online-no-improvement-ms "${ONLINE_NO_IMPROVEMENT_MS:-60000}")
        [[ -z ${ONLINE_TIME_MS:-} ]] || ONLINE_ARGS+=(--online-time-ms "$ONLINE_TIME_MS")
        STATIC_ARGS=(--static-no-improvement-ms "${STATIC_NO_IMPROVEMENT_MS:-60000}")
        [[ -z ${STATIC_TIME_MS:-} ]] || STATIC_ARGS+=(--static-time-ms "$STATIC_TIME_MS")
        python -u evaluate_ortools_baselines.py --problem "$PROBLEM" --dataset "$DATASET" \
            --output "$OUTPUT" --num-samples "$NUM_SAMPLES" \
            "${ONLINE_ARGS[@]}" "${STATIC_ARGS[@]}" \
            "${ORTec_ARGS[@]}" "$@"
        ;;
    msa)
        : "${DOD:?Set DOD to the dataset degree of dynamism for MSA evaluation}"
        SOLVER_ARGS=(--solver-no-improvement-ms "${SOLVER_NO_IMPROVEMENT_MS:-60000}")
        [[ -z ${SOLVER_TIME_MS:-} ]] || SOLVER_ARGS+=(--solver-time-ms "$SOLVER_TIME_MS")
        python -u evaluate_msa_ortools.py --problem "$PROBLEM" --dataset "$DATASET" \
            --output "$OUTPUT" --dod "$DOD" --num-samples "$NUM_SAMPLES" \
            --scenarios "${SCENARIOS:-10}" --workers "${MSA_WORKERS:-1}" "${SOLVER_ARGS[@]}" \
            --seed "${SEED:-1234}" "${ORTec_ARGS[@]}" "$@"
        ;;
    *) echo "Unsupported method: $METHOD" >&2; usage >&2; exit 2 ;;
esac
