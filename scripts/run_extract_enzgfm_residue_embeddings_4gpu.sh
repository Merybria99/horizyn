#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENZGFM_ENV_PATH="${ENZGFM_ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/enzgfm-env}"
PYTHON_BIN="${PYTHON_BIN:-$ENZGFM_ENV_PATH/bin/python}"
ENZGFM_REPO="${ENZGFM_REPO:-$PROJECT_ROOT/third_party/EnzGFM}"
MODEL_LOCATION="${MODEL_LOCATION:-$PROJECT_ROOT/checkpoints/EnzGFM-650M}"
TOKENIZER_LOCATION=""
RUN_NAME="enzgfm_650m_residue_extract_$(date +%Y%m%d_%H%M%S)"
FASTA_PATH="data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/proteins.fasta"
OUTPUT_PATH="data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/proteins_enzgfm_650m_residue.h5"
TMP_DIR=""
BATCH_SIZE="4"
MAX_TOKENS_PER_BATCH="2048"
MAX_SEQUENCE_LENGTH="1022"
DTYPE="float16"
COMPRESSION="none"
SKIP_LOAD="false"
CLEANUP_SHARDS="false"
DISABLE_FAST_KERNELS="false"
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --fasta) FASTA_PATH="$2"; shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --tmp-dir) TMP_DIR="$2"; shift 2 ;;
    --model-location) MODEL_LOCATION="$2"; shift 2 ;;
    --enzgfm-repo) ENZGFM_REPO="$2"; shift 2 ;;
    --tokenizer-location) TOKENIZER_LOCATION="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-tokens-per-batch) MAX_TOKENS_PER_BATCH="$2"; shift 2 ;;
    --max-sequence-length) MAX_SEQUENCE_LENGTH="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --compression) COMPRESSION="$2"; shift 2 ;;
    --skip-load) SKIP_LOAD="true"; shift ;;
    --cleanup-shards) CLEANUP_SHARDS="true"; shift ;;
    --disable-fast-kernels) DISABLE_FAST_KERNELS="true"; shift ;;
    --force) FORCE="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "EnzGFM Python is not executable: $PYTHON_BIN" >&2
  echo "Create the authors' Python 3.10 environment or set PYTHON_BIN." >&2
  exit 2
fi
if [[ ! -f "$ENZGFM_REPO/models/__init__.py" ]]; then
  echo "Official EnzGFM checkout not found: $ENZGFM_REPO" >&2
  exit 2
fi
if [[ ! -f "$MODEL_LOCATION/config.json" ]]; then
  echo "EnzGFM model directory is missing config.json: $MODEL_LOCATION" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
WORLD_SIZE="${#GPUS[@]}"
if [[ "$WORLD_SIZE" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU" >&2
  exit 2
fi

LOG_DIR="logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run.log"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_PATH")"
printf '%s\n' "$$" > "${LOG_DIR}/run.pid"

COMMON_ARGS=(
  --fasta "$FASTA_PATH"
  --output "$OUTPUT_PATH"
  --model-location "$MODEL_LOCATION"
  --enzgfm-repo "$ENZGFM_REPO"
  --world-size "$WORLD_SIZE"
  --batch-size "$BATCH_SIZE"
  --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH"
  --max-sequence-length "$MAX_SEQUENCE_LENGTH"
  --dtype "$DTYPE"
  --compression "$COMPRESSION"
  --resume
)
if [[ -n "$TOKENIZER_LOCATION" ]]; then
  COMMON_ARGS+=(--tokenizer-location "$TOKENIZER_LOCATION")
fi
if [[ -n "$TMP_DIR" ]]; then
  COMMON_ARGS+=(--tmp-dir "$TMP_DIR")
fi
if [[ "$DISABLE_FAST_KERNELS" == "true" ]]; then
  COMMON_ARGS+=(--disable-fast-kernels)
fi
if [[ "$FORCE" == "true" ]]; then
  COMMON_ARGS+=(--force)
fi

{
  echo "[$(date --iso-8601=seconds)] Starting EnzGFM residue extraction"
  echo "Model: $MODEL_LOCATION"
  echo "EnzGFM source: $ENZGFM_REPO"
  echo "FASTA: $FASTA_PATH"
  echo "Output: $OUTPUT_PATH"
  echo "GPUs: $CUDA_VISIBLE_DEVICES"

  if [[ -e "$OUTPUT_PATH" && "$FORCE" != "true" ]]; then
    echo "Output already exists: $OUTPUT_PATH (use --force to overwrite)" >&2
    exit 1
  fi

  if [[ "$SKIP_LOAD" != "true" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" \
      scripts/extract_enzgfm_residue_embeddings.py \
      "${COMMON_ARGS[@]}" --load-only --device cuda
  fi

  PIDS=()
  for rank in "${!GPUS[@]}"; do
    gpu="${GPUS[$rank]}"
    rank_log="${LOG_DIR}/rank${rank}.log"
    echo "Rank $rank -> GPU $gpu -> $rank_log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
      scripts/extract_enzgfm_residue_embeddings.py \
      "${COMMON_ARGS[@]}" --rank "$rank" --device cuda \
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
    echo "One or more EnzGFM extraction workers failed" >&2
    exit "$status"
  fi

  MERGE_ARGS=("${COMMON_ARGS[@]}" --merge-only)
  if [[ "$CLEANUP_SHARDS" == "true" ]]; then
    MERGE_ARGS+=(--cleanup-shards)
  fi
  "$PYTHON_BIN" scripts/extract_enzgfm_residue_embeddings.py "${MERGE_ARGS[@]}"
  echo "[$(date --iso-8601=seconds)] EnzGFM extraction complete"
} 2>&1 | tee "$RUN_LOG"
