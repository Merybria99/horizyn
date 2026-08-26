#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/datastor2/deep-proteins/EnzymeDiscovery/horizyn"
ENV_ROOT="/datastor2/deep-proteins/EnzymeDiscovery/env"
RUN_NAME="${RUN_NAME:-sota_4gpu_b65536}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-65536}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

cd "$PROJECT_ROOT"
source "$ENV_ROOT/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1

LOG_DIR="logs/horizyn_retrieval/${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/horizyn_retrieval/${RUN_NAME}"
RESULT_DIR="results/horizyn_retrieval"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$RESULT_DIR"

echo "[$(date --iso-8601=seconds)] Starting Horizyn SOTA 4-GPU replication"
echo "Project: $PROJECT_ROOT"
echo "Environment: $ENV_ROOT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Per-GPU train batch size: $TRAIN_BATCH_SIZE"
echo "Global train batch size: $((TRAIN_BATCH_SIZE * 4))"
echo "Logs: $LOG_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  echo "Resume checkpoint: $RESUME_CHECKPOINT"
fi

TRAIN_ARGS=(
  train.py
  --config configs/sota.yaml \
  --training.devices 4 \
  --training.accelerator gpu \
  --training.strategy ddp \
  --training.enable_progress_bar false \
  --data.train_batch_size "$TRAIN_BATCH_SIZE" \
  --logging.log_dir "$LOG_DIR" \
  --logging.checkpoint_dir "$CHECKPOINT_DIR"
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
fi

python "${TRAIN_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] Training complete; starting evaluation"

python scripts/evaluate.py \
  --checkpoint "${CHECKPOINT_DIR}/last.ckpt" \
  --config configs/sota.yaml \
  --batch-size 4096 \
  --output "${RESULT_DIR}/${RUN_NAME}_last.json"

echo "[$(date --iso-8601=seconds)] SOTA 4-GPU replication complete"
