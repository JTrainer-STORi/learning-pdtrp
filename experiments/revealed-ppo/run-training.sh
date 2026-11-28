# Shared PPO settings. Arguments: decoder tanh clipping (0 = raw), then training baseline.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" NUM_WORKERS=2
export ENCODER=transformer NORMALIZATION=layer NEIGHBORS=-1 KNN_STRAT=fixed
export MIN_CUSTOMERS=50 HOURS_PER_CUSTOMER=0.2 MIN_DOD=0.2 MAX_DOD=0.8
export TIME_WINDOW_SLACK=on ARRIVAL_LSTM=off VALIDATION_SEED=12345
export N_EPOCHS EPOCH_SIZE=12800 BATCH_SIZE=256 VAL_SIZE=128 VAL_BATCH_SIZE=1
export ROLLOUT_SIZE=1280 ROLLOUT_BATCH_SIZE=256 DEVICE=auto ENABLE_WANDB=0
export OUTPUT_DIR="$CASE/models" LOG_DIR="$CASE/logs"
git rev-parse HEAD > "$CASE/git-revision.txt"
sha256sum nets/dynamic_attention_model.py nets/encoders/transformer_encoder.py \
    train.py run.py options.py reinforce_baselines.py scripts/train-policy.sh experiments/revealed-ppo/run-training.sh "$0" > "$CASE/source.sha256"
nvidia-smi --query-gpu=name,memory.total --format=csv > "$CASE/gpu.csv"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'

srun --ntasks=1 --export=ALL --cpu-bind=cores bash scripts/train-policy.sh pdctrptw 50 "$SEED" -- \
    --training-algorithm ppo --ppo-update-epochs 4 --ppo-encoding "$ARM" --tanh_clipping "${1:-10}" \
    --use-capacity-aware-lookahead --baseline "${2:-rollout}" --bl_warmup_epochs 0 \
    --training-data-seed "$((91230 + SEED))" --lr_model 0.0001 \
    --validation-dods 0.2 0.5 0.8 --val_every 2 --save-best-validation --no_progress_bar
