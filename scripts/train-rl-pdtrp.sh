#!/usr/bin/env bash
#SBATCH --cpus-per-task=4

# To get an idea of what each option does, please look at the options.py file

RUN_NAME="rl-ar-var-20pnn-gnn-max-pdtrp"

PROBLEM="pdtrp"

DEVICES="0"
NUM_WORKERS=4

MIN_TOTAL=50
MAX_TOTAL=50
MIN_DOD=0.5
MAX_DOD=0.5
NEIGHBORS=0.2
KNN_STRAT="percentage"

N_SUBREGIONS=1

VAL_DATASET1="new_data/pdtrp/pdtrp40dod0.2v4th8_testing_dimscale4_ortools.txt"
VAL_DATASET2="new_data/pdtrp/pdtrp40dod0.5v4th8_testing_dimscale4_ortools.txt"
VAL_DATASET3="new_data/pdtrp/pdtrp40dod0.8v4th8_testing_dimscale4_ortools.txt"

DIMENSION_SCALE=4

N_EPOCHS=100
EPOCH_SIZE=160
BATCH_SIZE=16
ACCUMULATION_STEPS=1

VAL_SIZE=64
ROLLOUT_SIZE=128
# Batched validation uses the fast masked monitoring path. Use 1 for recursive validation.
VAL_BATCH_SIZE=16
ROLLOUT_BATCH_SIZE=16

MODEL="dattention"
ENCODER="gnn"
AGGREGATION="max"
AGGREGATION_GRAPH="mean"
NORMALIZATION="batch"
EMBEDDING_DIM=128
N_ENCODE_LAYERS=3
EDGE_FEATURES="adjacency"

BASELINE="rollout"

LR_MODEL=0.0001
MAX_NORM=1
CHECKPOINT_EPOCHS=0

CUDA_VISIBLE_DEVICES="$DEVICES" python run.py --problem "$PROBLEM" \
    --model "$MODEL" --baseline "$BASELINE" \
    --min_total "$MIN_TOTAL" --max_total "$MAX_TOTAL" \
    --min_dod "$MIN_DOD" --max_dod "$MAX_DOD" \
    --neighbors "$NEIGHBORS" --knn_strat "$KNN_STRAT" \
    --val_datasets "$VAL_DATASET1" "$VAL_DATASET2" \
    --epoch_size "$EPOCH_SIZE" \
    --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUMULATION_STEPS" \
    --n_epochs "$N_EPOCHS" \
    --val_size "$VAL_SIZE" --rollout_size "$ROLLOUT_SIZE" \
    --encoder "$ENCODER" --aggregation "$AGGREGATION" --aggregation_graph "$AGGREGATION_GRAPH" \
    --n_encode_layers "$N_ENCODE_LAYERS" --gated \
    --normalization "$NORMALIZATION" --learn_norm \
    --embedding_dim "$EMBEDDING_DIM" --hidden_dim "$EMBEDDING_DIM" \
    --lr_model "$LR_MODEL" --max_grad_norm "$MAX_NORM" \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_epochs "$CHECKPOINT_EPOCHS" \
    --run_name "$RUN_NAME" \
    --edge_features "$EDGE_FEATURES" --val_batch_size "$VAL_BATCH_SIZE" \
    --no_tensorboard --no_videos --val_every 5 --baseline_update_every 5 \
    --n_subregions "$N_SUBREGIONS" --rollout_batch_size "$ROLLOUT_BATCH_SIZE" --dimension_scale "$DIMENSION_SCALE"
