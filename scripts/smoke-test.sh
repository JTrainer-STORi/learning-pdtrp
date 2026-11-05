#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

python -m pytest -q

python - <<'PY'
import torch

from evaluate_model_free import nearest_neighbour
from problems import PDTRP
from utils.data_utils import dataset_to_input

problem = PDTRP()
dataset = problem.make_dataset(
    min_customers=8,
    max_customers=8,
    min_dod=0.5,
    max_dod=0.5,
    num_samples=1,
    batch_size=1,
    n_subregions=1,
    neighbors=0.25,
    knn_strat="percentage",
)
model_input = dataset_to_input(dataset[0], problem.NAME, torch.device("cpu"))
tour, visit_times = nearest_neighbour(problem, model_input)
assert tour.shape[1] == 9
assert visit_times.shape == (tour.shape[0], tour.shape[1] + 1)
PY

echo "SMOKE TEST PASSED"
