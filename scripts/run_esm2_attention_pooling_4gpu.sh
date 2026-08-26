#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENV_PATH="${ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"

RUN_NAME="protein_attention_pooling_esm2_650m_b128_len1022_4gpu_$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="configs/protein_attention_pooling_esm2_650m_sota.yaml"
FASTA_PATH="data/sota/prots.fasta"
RESIDUE_OUTPUT="data/sota/prots_esm2_650m_residue.h5"
TRAIN_BATCH_SIZE="128"
EXTRACT_BATCH_SIZE="8"
MAX_TOKENS_PER_BATCH="4096"
MAX_SEQUENCE_LENGTH="1022"
WANDB_PROJECT="horizyn-training"
WANDB_ENTITY=""
WANDB_TAGS=(esm2 attention-pooling 650m)
FORCE_EXTRACT="false"
SKIP_EXTRACT="false"

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
    --fasta)
      FASTA_PATH="$2"
      shift 2
      ;;
    --residue-output)
      RESIDUE_OUTPUT="$2"
      shift 2
      ;;
    --train-batch-size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --extract-batch-size)
      EXTRACT_BATCH_SIZE="$2"
      shift 2
      ;;
    --max-tokens-per-batch)
      MAX_TOKENS_PER_BATCH="$2"
      shift 2
      ;;
    --max-sequence-length)
      MAX_SEQUENCE_LENGTH="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb-tags)
      shift
      WANDB_TAGS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        WANDB_TAGS+=("$1")
        shift
      done
      ;;
    --force-extract)
      FORCE_EXTRACT="true"
      shift
      ;;
    --skip-extract)
      SKIP_EXTRACT="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$PROJECT_ROOT"
source "$ENV_PATH/bin/activate"

if "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 12) else 1)' >/dev/null 2>&1
then
  :
else
  if [[ -x /lusr/bin/python3.12 && -d "$ENV_PATH/lib/python3.12/site-packages" ]]; then
    PYTHON_BIN="/lusr/bin/python3.12"
    export PYTHONPATH="$ENV_PATH/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
  else
    echo "Could not find a Python 3.12 interpreter for the project environment" >&2
    exit 2
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/datastor2/deep-proteins/EnzymeDiscovery/hf_cache}"

LOG_DIR="logs/${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" results "$HF_HOME"
printf '%s\n' "$$" > "${LOG_DIR}/pipeline.pid"

{
  echo "[$(date --iso-8601=seconds)] Starting ESM2 attention-pooling pipeline"
  echo "Run: $RUN_NAME"
  echo "Project: $PROJECT_ROOT"
  echo "Python: $PYTHON_BIN"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "HF_HOME: $HF_HOME"
  echo "Config: $CONFIG_PATH"
  echo "FASTA: $FASTA_PATH"
  echo "Residue HDF5: $RESIDUE_OUTPUT"
  echo "Train batch size: $TRAIN_BATCH_SIZE"
  echo "W&B project: $WANDB_PROJECT"

  if [[ "$SKIP_EXTRACT" != "true" ]]; then
    EXTRACT_ARGS=(
      --run-name "${RUN_NAME}_extract"
      --fasta "$FASTA_PATH"
      --output "$RESIDUE_OUTPUT"
      --batch-size "$EXTRACT_BATCH_SIZE"
      --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH"
      --max-sequence-length "$MAX_SEQUENCE_LENGTH"
      --dtype float16
    )
    if [[ "$FORCE_EXTRACT" == "true" ]]; then
      EXTRACT_ARGS+=(--force)
    fi

    echo "[$(date --iso-8601=seconds)] Starting ESM2 residue extraction"
    scripts/run_extract_esm2_residue_embeddings_4gpu.sh "${EXTRACT_ARGS[@]}"
  else
    echo "[$(date --iso-8601=seconds)] Skipping extraction"
  fi

  echo "[$(date --iso-8601=seconds)] Starting ESM2 attention-pooling training"
  TRAIN_ARGS=(
    --config "$CONFIG_PATH"
    --wandb
    --wandb-mode online
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$RUN_NAME"
    --wandb-tags "${WANDB_TAGS[@]}"
    --data.protein_residue_embeds_path "$RESIDUE_OUTPUT"
    --data.train_batch_size "$TRAIN_BATCH_SIZE"
    --logging.log_dir "$LOG_DIR"
    --logging.checkpoint_dir "$CHECKPOINT_DIR"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi

  "$PYTHON_BIN" scripts/train_protein_pooling.py "${TRAIN_ARGS[@]}"
  echo "[$(date --iso-8601=seconds)] ESM2 attention-pooling pipeline complete"
} 2>&1 | tee "${LOG_DIR}/pipeline.log"
