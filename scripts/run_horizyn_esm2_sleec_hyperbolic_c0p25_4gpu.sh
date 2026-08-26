#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
cd "$ROOT_DIR"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-configs/horizyn_esm2_sleec_hyperbolic_c0p25.yaml}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-horizyn-training}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-0}"

RUN_NAME="horizyn-esm2-sleec-hyperbolic-c0p25-frozen-stage1-4gpu-${RUN_STAMP}"
LOG_DIR="logs/SLEEC/${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/SLEEC/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/training.log"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" run_pids/SLEEC
printf '%s\n' "$$" > "run_pids/SLEEC/${RUN_NAME}.pid"

train_args=(
  --config "$CONFIG"
  --wandb
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$RUN_NAME"
  --wandb-tags horizyn esm2-650m sleec-guided-attention hyperbolic-c0p25 frozen-stage1 mlnce 4gpu
  --data.train_batch_size "$TRAIN_BATCH_SIZE"
  --training.max_epochs "$EPOCHS"
  --logging.log_dir "$LOG_DIR"
  --logging.checkpoint_dir "$CHECKPOINT_DIR"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  train_args+=(--wandb-entity "$WANDB_ENTITY")
fi
case "${WANDB_LOG_MODEL,,}" in
  1|true|yes) train_args+=(--wandb-log-model) ;;
esac

{
  echo "[$(date --iso-8601=seconds)] Starting Horizyn ESM2/SLEEC/hyperbolic training"
  echo "Root: $ROOT_DIR"
  echo "Python: $PYTHON_BIN"
  echo "Config: $CONFIG"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Train batch size: $TRAIN_BATCH_SIZE"
  echo "Epochs: $EPOCHS"
  echo "Checkpoint dir: $CHECKPOINT_DIR"
  echo "W&B project: $WANDB_PROJECT"
  echo "W&B mode: $WANDB_MODE"
  nvidia-smi || true
  "$PYTHON_BIN" scripts/train_protein_pooling.py "${train_args[@]}"
} 2>&1 | tee "$LOG_FILE"
