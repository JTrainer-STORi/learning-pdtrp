#!/usr/bin/env bash

RUN_NAME="rl-ar-var-20pnn-gnn-max-dtsp"

PROBLEM="pdtsp"

DEVICES="0"
NUM_WORKERS=0

MIN_STATIC=20
MAX_STATIC=20
MIN_DYNA=20
MAX_DYNA=20
NEIGHBORS=0.2
KNN_STRAT="percentage"

SPATIAL_DISTRIBUTION="unimodal"
TD1="equal"
MEAN_1=(0.2 0.2)
COV_1=(0.1 0 0 0.1)

VAL_DATASET1="data/pdtsp/tsp2020_test_conc_insertion.txt"
VAL_DATASET2="data/pdtsp/tsp5050_test_conc_insertion.txt"
# VAL_DATASET3="data/tsp/tsp100_test_concorde.txt"

N_EPOCHS=100
EPOCH_SIZE=160
BATCH_SIZE=16
ACCUMULATION_STEPS=1

VAL_SIZE=160
ROLLOUT_SIZE=160

MODEL="dattention"
ENCODER="gnn"
AGGREGATION="max"
AGGREGATION_GRAPH="mean"
NORMALIZATION="batch"
EMBEDDING_DIM=128
N_ENCODE_LAYERS=3

BASELINE="rollout"

LR_MODEL=0.0001
MAX_NORM=1
CHECKPOINT_EPOCHS=0
SET_START=True

CUDA_VISIBLE_DEVICES="$DEVICES" python run.py --problem "$PROBLEM" \
    --model "$MODEL" --baseline "$BASELINE" \
    --min_static "$MIN_STATIC" --max_static "$MAX_STATIC" \
    --min_dyna "$MIN_DYNA" --max_dyna "$MAX_DYNA" \
    --neighbors "$NEIGHBORS" --knn_strat "$KNN_STRAT" \
    --spatial_distribution "$SPATIAL_DISTRIBUTION" \
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
    --run_name "$RUN_NAME" --set_start "$SET_START" \
    --mean1 "${MEAN_1[0]}" "${MEAN_1[1]}" \
    --cov1 "${COV_1[0]}" "${COV_1[1]}" "${COV_1[2]}" "${COV_1[3]}" \
    --td1  "$TD1"
