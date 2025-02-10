#!/usr/bin/env bash

RUN_NAME="rl-ar-var-20pnn-gnn-max-pdcvrptw"

PROBLEM="pdcvrptw"

DEVICES="0"
NUM_WORKERS=0

MIN_TOTAL=40
MAX_TOTAL=100
MIN_DOD=0.2
MAX_DOD=0.8
NEIGHBORS=0.2
KNN_STRAT="percentage"

VAL_DATASET1="data/pdcvrptw/validation_data_PDCVRPTW.txt"

N_EPOCHS=100
EPOCH_SIZE=160
BATCH_SIZE=16
ACCUMULATION_STEPS=1

VAL_SIZE=16
ROLLOUT_SIZE=16
VAL_BATCH_SIZE=8

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
SET_START=True
TIME_HORIZON=1
L=0.125

CUDA_VISIBLE_DEVICES="$DEVICES" python run.py --problem "$PROBLEM" \
    --model "$MODEL" --baseline "$BASELINE" \
    --min_total "$MIN_TOTAL" --max_total "$MAX_TOTAL" \
    --min_dod "$MIN_DOD" --max_dod "$MAX_DOD" \
    --neighbors "$NEIGHBORS" --knn_strat "$KNN_STRAT" \
    --val_datasets "$VAL_DATASET1" \
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
    --run_name "$RUN_NAME" --set_start "$SET_START" \
    --edge_features "$EDGE_FEATURES" --time_horizon "$TIME_HORIZON" \
    --L "$L" --val_batch_size "$VAL_BATCH_SIZE"