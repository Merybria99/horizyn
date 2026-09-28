#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENV_PATH="${ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"
RUN_NAME="esm2_residue_extract_$(date +%Y%m%d_%H%M%S)"
FASTA_PATH="data/sota/prots.fasta"
OUTPUT_PATH="data/sota/prots_esm2_650m_residue.h5"
TMP_DIR=""
MODEL_NAME="facebook/esm2_t33_650M_UR50D"
BATCH_SIZE="8"
MAX_TOKENS_PER_BATCH="4096"
MAX_SEQUENCE_LENGTH="1022"
DTYPE="float16"
COMPRESSION="none"
SKIP_DOWNLOAD="false"
CLEANUP_SHARDS="false"
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --fasta)
      FASTA_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --tmp-dir)
      TMP_DIR="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-tokens-per-batch)
      MAX_TOKENS_PER_BATCH="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --max-sequence-length)
      MAX_SEQUENCE_LENGTH="$2"
      shift 2
      ;;
    --compression)
      COMPRESSION="$2"
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD="true"
      shift
      ;;
    --cleanup-shards)
      CLEANUP_SHARDS="true"
      shift
      ;;
    --force)
      FORCE="true"
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

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
WORLD_SIZE="${#GPUS[@]}"
if [[ "$WORLD_SIZE" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU" >&2
  exit 2
fi

LOG_DIR="logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run.log"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_PATH")" "$HF_HOME"
printf '%s\n' "$$" > "${LOG_DIR}/run.pid"

COMMON_ARGS=(
  --fasta "$FASTA_PATH"
  --output "$OUTPUT_PATH"
  --model-name "$MODEL_NAME"
  --world-size "$WORLD_SIZE"
  --batch-size "$BATCH_SIZE"
  --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH"
  --max-sequence-length "$MAX_SEQUENCE_LENGTH"
  --dtype "$DTYPE"
  --compression "$COMPRESSION"
  --resume
)

if [[ -n "$TMP_DIR" ]]; then
  COMMON_ARGS+=(--tmp-dir "$TMP_DIR")
fi
if [[ "$FORCE" == "true" ]]; then
  COMMON_ARGS+=(--force)
fi

{
  echo "[$(date --iso-8601=seconds)] Starting ESM2 residue embedding extraction"
  echo "Project: $PROJECT_ROOT"
  echo "Environment: $ENV_PATH"
  echo "Python: $PYTHON_BIN"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "World size: $WORLD_SIZE"
  echo "Model: $MODEL_NAME"
  echo "FASTA: $FASTA_PATH"
  echo "Output: $OUTPUT_PATH"
  echo "HF_HOME: $HF_HOME"
  echo "Batch size: $BATCH_SIZE"
  echo "Max tokens per batch: $MAX_TOKENS_PER_BATCH"
  echo "Max sequence length: $MAX_SEQUENCE_LENGTH"
  echo "Stored dtype: $DTYPE"
  echo "Compression: $COMPRESSION"

  if [[ -e "$OUTPUT_PATH" && "$FORCE" != "true" ]]; then
    echo "Output already exists: $OUTPUT_PATH (use --force to overwrite)" >&2
    exit 1
  fi

  if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
    echo "[$(date --iso-8601=seconds)] Prefetching tokenizer/model into HF cache"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" scripts/extract_esm2_residue_embeddings.py \
      --download-only \
      --device cuda \
      --model-name "$MODEL_NAME" \
      --dtype "$DTYPE"
  fi

  echo "[$(date --iso-8601=seconds)] Launching $WORLD_SIZE extraction workers"
  PIDS=()
  for rank in "${!GPUS[@]}"; do
    gpu="${GPUS[$rank]}"
    rank_log="${LOG_DIR}/rank${rank}.log"
    echo "Rank $rank -> GPU $gpu -> $rank_log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/extract_esm2_residue_embeddings.py \
      "${COMMON_ARGS[@]}" \
      --rank "$rank" \
      --device cuda \
      > "$rank_log" 2>&1 &
    PIDS+=("$!")
    printf '%s\n' "${PIDS[-1]}" > "${LOG_DIR}/rank${rank}.pid"
  done

  status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date --iso-8601=seconds)] One or more extraction workers failed" >&2
    exit "$status"
  fi

  echo "[$(date --iso-8601=seconds)] All workers finished; merging shards"
  MERGE_ARGS=("${COMMON_ARGS[@]}" --merge-only)
  if [[ "$CLEANUP_SHARDS" == "true" ]]; then
    MERGE_ARGS+=(--cleanup-shards)
  fi
  "$PYTHON_BIN" scripts/extract_esm2_residue_embeddings.py "${MERGE_ARGS[@]}"

  echo "[$(date --iso-8601=seconds)] Extraction complete"
} 2>&1 | tee "$RUN_LOG"
