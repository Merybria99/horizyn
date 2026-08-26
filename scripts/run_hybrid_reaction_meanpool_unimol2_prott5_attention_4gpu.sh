#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENV_PATH="${ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DEFAULT_RUN_NAME="hybrid_reaction_meanpool_unimol2_prott5_attention_b16_4gpu_$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
CONFIG_PATH="${CONFIG_PATH:-configs/hybrid_reaction_meanpool_unimol2_prott5_attention_sota.yaml}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-512}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-128}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
WANDB_ENABLED="${WANDB_ENABLED:-${WANDB:-0}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-horizyn-training}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"
HORIZYN_WANDB_TAGS="${HORIZYN_WANDB_TAGS:-hybrid-reaction unimol2-mean-pooling prott5-attention 4gpu b16}"

cd "$PROJECT_ROOT"
source "$ENV_PATH/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

LOG_DIR="logs/${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
RESULT_DIR="results"
RESULT_PATH="${RESULT_DIR}/${RUN_NAME}_last.json"
RUN_LOG="${LOG_DIR}/run.log"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$RESULT_DIR"
printf '%s\n' "$$" > "${LOG_DIR}/run.pid"

{
  echo "[$(date --iso-8601=seconds)] Starting hybrid reaction mean-pooling Uni-Mol2 run"
  echo "Project: $PROJECT_ROOT"
  echo "Environment: $ENV_PATH"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Run name: $RUN_NAME"
  echo "Config: $CONFIG_PATH"
  echo "Per-GPU train batch size: $TRAIN_BATCH_SIZE"
  echo "Effective DDP batch size: $((TRAIN_BATCH_SIZE * 4))"
  echo "Logs: $LOG_DIR"
  echo "Checkpoints: $CHECKPOINT_DIR"
  echo "Results: $RESULT_PATH"
  echo "W&B enabled: $WANDB_ENABLED"
  echo "W&B mode: $WANDB_MODE"
  echo "W&B project: $WANDB_PROJECT"
  echo "W&B run name: $WANDB_RUN_NAME"

  TRAIN_ARGS=(
    scripts/train_protein_pooling.py
    --config "$CONFIG_PATH"
    --training.devices 4
    --training.accelerator gpu
    --training.strategy ddp
    --training.enable_progress_bar false
    --data.train_batch_size "$TRAIN_BATCH_SIZE"
    --logging.log_dir "$LOG_DIR"
    --logging.checkpoint_dir "$CHECKPOINT_DIR"
  )

  case "${WANDB_ENABLED,,}" in
    1|true|yes|on)
      TRAIN_ARGS+=(
        --wandb
        --wandb-mode "$WANDB_MODE"
        --wandb-project "$WANDB_PROJECT"
        --wandb-run-name "$WANDB_RUN_NAME"
      )

      if [[ -n "$HORIZYN_WANDB_TAGS" ]]; then
        read -r -a WANDB_TAG_ARRAY <<< "$HORIZYN_WANDB_TAGS"
        TRAIN_ARGS+=(--wandb-tags "${WANDB_TAG_ARRAY[@]}")
      fi
      ;;
  esac

  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
  fi

  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"

  echo "[$(date --iso-8601=seconds)] Training complete; starting evaluation"

  "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
    --checkpoint "${CHECKPOINT_DIR}/last.ckpt" \
    --config "$CONFIG_PATH" \
    --batch-size "$QUERY_BATCH_SIZE" \
    --target-batch-size "$TARGET_BATCH_SIZE" \
    --output "$RESULT_PATH"

  echo "[$(date --iso-8601=seconds)] Hybrid reaction mean-pooling Uni-Mol2 run complete"
} 2>&1 | tee "$RUN_LOG"
