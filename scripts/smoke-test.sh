#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

python -m pytest -q

SMOKE_OUTPUT=${SMOKE_OUTPUT:-artifacts/smoke-test}
NUM_SAMPLES=2 "$SCRIPT_DIR/evaluate-baseline.sh" nearest pdtrp \
    experimental_study_datasets/pdtrp/pdtrp50dod0.2v4th8_testing_ortools.txt \
    "$SMOKE_OUTPUT/nearest"

echo "SMOKE TEST PASSED"
