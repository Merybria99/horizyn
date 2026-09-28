#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENV_PATH="${ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
RUN_NAME="reaction_conditioned_v1_sota_$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="configs/reaction_conditioned_sota.yaml"
RESIDUE_EMBEDS="data/sota/prots_t5_residue.h5"
WANDB_PROJECT="horizyn-training"
WANDB_MODE="online"
TRAIN_BATCH_SIZE="64"
MAX_PROTEIN_TOKENS="1024"
TARGET_CHUNK_SIZE="64"
RESUME_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --residue-embeds)
      RESIDUE_EMBEDS="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --train-batch-size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --max-protein-tokens)
      MAX_PROTEIN_TOKENS="$2"
      shift 2
      ;;
    --target-chunk-size)
      TARGET_CHUNK_SIZE="$2"
      shift 2
      ;;
    --resume)
      RESUME_CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

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
  echo "[$(date --iso-8601=seconds)] Starting reaction-conditioned Horizyn SOTA 4-GPU run"
  echo "Project: $PROJECT_ROOT"
  echo "Environment: $ENV_PATH"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Run name: $RUN_NAME"
  echo "Config: $CONFIG_PATH"
  echo "Residue embeddings: $RESIDUE_EMBEDS"
  echo "Per-GPU train batch size: $TRAIN_BATCH_SIZE"
  echo "Max protein tokens: $MAX_PROTEIN_TOKENS"
  echo "Logs: $LOG_DIR"
  echo "Checkpoints: $CHECKPOINT_DIR"
  echo "Results: $RESULT_PATH"

  TRAIN_ARGS=(
    scripts/train_reaction_conditioned.py
    --config "$CONFIG_PATH"
    --training.devices 4
    --training.accelerator gpu
    --training.strategy ddp
    --training.enable_progress_bar false
    --data.protein_residue_embeds_path "$RESIDUE_EMBEDS"
    --data.train_batch_size "$TRAIN_BATCH_SIZE"
    --data.max_protein_tokens "$MAX_PROTEIN_TOKENS"
    --logging.log_dir "$LOG_DIR"
    --logging.checkpoint_dir "$CHECKPOINT_DIR"
    --wandb
    --wandb-mode "$WANDB_MODE"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$RUN_NAME"
    --wandb-tags reaction-conditioned sota 4gpu v1
  )

  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
  fi

  python "${TRAIN_ARGS[@]}"

  echo "[$(date --iso-8601=seconds)] Training complete; starting evaluation"

  python scripts/evaluate_reaction_conditioned.py \
    --checkpoint "${CHECKPOINT_DIR}/last.ckpt" \
    --config "$CONFIG_PATH" \
    --batch-size 1 \
    --target-chunk-size "$TARGET_CHUNK_SIZE" \
    --output "$RESULT_PATH"

  echo "[$(date --iso-8601=seconds)] Reaction-conditioned SOTA 4-GPU run complete"
} 2>&1 | tee "$RUN_LOG"
