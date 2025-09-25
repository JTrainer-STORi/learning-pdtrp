#!/usr/bin/env bash

PROBLEM="pdtrp"
MODEL_PATH="ablation_models/pdtrp/pdtrp_20-50-0.2-0.8/pdtrp-dynamic-seed-0_20250827T175700"
MODEL_NAME="example-test"
SEED=22
MODE="recursive"
FILE_ROOT="ablation_datasets/pdtrp/"

python testing.py --problem "$PROBLEM" --model_path "$MODEL_PATH" --seed "$SEED" --model_mode "$MODE" --model_name "$MODEL_NAME" --specific_file_root "$FILE_ROOT"