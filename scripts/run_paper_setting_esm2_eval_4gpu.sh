#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENV_PATH="${ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"
RUN_NAME="paper_setting_esm2_eval_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="checkpoints/protein_attention/protein_attention_pooling_esm2_650m_b128_len1022_4gpu_screen_hfcache_20260518_151727/protein-pooling-epoch=09.ckpt"
CONFIG="configs/protein_attention_pooling_esm2_650m_sota.yaml"
DATA_ROOT="data/paper"
EXTRACT_BATCH_SIZE="8"
MAX_TOKENS_PER_BATCH="4096"
TARGET_BATCH_SIZE="512"
QUERY_BATCH_SIZE="128"
FORCE_EXTRACT="false"
SKIP_EXTRACT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
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
    --target-batch-size)
      TARGET_BATCH_SIZE="$2"
      shift 2
      ;;
    --query-batch-size)
      QUERY_BATCH_SIZE="$2"
      shift 2
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
RESULT_DIR="results/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run.log"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$HF_HOME"
printf '%s\n' "$$" > "${LOG_DIR}/run.pid"

CLIP_FASTA="${DATA_ROOT}/clipzyme/eval/enzymemap/proteins.fasta"
CLIP_H5="${DATA_ROOT}/clipzyme/eval/enzymemap/proteins_esm2_650m_residue.h5"
REACT_FASTA="${DATA_ROOT}/reactzyme/eval/all_proteins.fasta"
REACT_H5="${DATA_ROOT}/reactzyme/eval/all_proteins_esm2_650m_residue.h5"

extract_if_needed() {
  local fasta="$1"
  local output="$2"
  local name="$3"
  if [[ "$SKIP_EXTRACT" == "true" ]]; then
    echo "[$(date --iso-8601=seconds)] Skipping extraction for ${name}"
    return
  fi
  if [[ -e "$output" && "$FORCE_EXTRACT" != "true" ]]; then
    echo "[$(date --iso-8601=seconds)] Reusing existing ${name} residue HDF5: ${output}"
    return
  fi
  local args=(
    --run-name "${RUN_NAME}_${name}_extract"
    --fasta "$fasta"
    --output "$output"
    --batch-size "$EXTRACT_BATCH_SIZE"
    --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH"
    --cleanup-shards
  )
  if [[ "$FORCE_EXTRACT" == "true" ]]; then
    args+=(--force)
  fi
  echo "[$(date --iso-8601=seconds)] Extracting ${name} ESM2 residues -> ${output}"
  scripts/run_extract_esm2_residue_embeddings_4gpu.sh "${args[@]}"
}

run_eval() {
  local name="$1"
  shift
  local output="${RESULT_DIR}/${name}.json"
  local log="${LOG_DIR}/${name}.log"
  echo "[$(date --iso-8601=seconds)] Running evaluation: ${name}"
  "$PYTHON_BIN" scripts/evaluate_paper_setting.py "$@" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --device cuda \
    --batch-size "$QUERY_BATCH_SIZE" \
    --target-batch-size "$TARGET_BATCH_SIZE" \
    --output "$output" \
    > "$log" 2>&1
  echo "[$(date --iso-8601=seconds)] Finished evaluation: ${name} -> ${output}"
}

{
  echo "[$(date --iso-8601=seconds)] Starting paper-setting ESM2 evaluation pipeline"
  echo "Project: $PROJECT_ROOT"
  echo "Environment: $ENV_PATH"
  echo "Python: $PYTHON_BIN"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Checkpoint: $CHECKPOINT"
  echo "Config: $CONFIG"
  echo "Data root: $DATA_ROOT"
  echo "Log dir: $LOG_DIR"
  echo "Result dir: $RESULT_DIR"

  extract_if_needed "$CLIP_FASTA" "$CLIP_H5" "clipzyme_enzymemap"
  run_eval "clipzyme_enzymemap" \
    --setting enzymemap \
    --pairs "${DATA_ROOT}/clipzyme/eval/enzymemap/pairs.csv" \
    --reactions "${DATA_ROOT}/clipzyme/eval/enzymemap/reactions.csv" \
    --candidate-residue-h5 "$CLIP_H5" \
    --candidate-ids "${DATA_ROOT}/clipzyme/eval/enzymemap/candidate_ids.txt"

  extract_if_needed "$REACT_FASTA" "$REACT_H5" "reactzyme"
  for split in time enzyme_smi reaction_smi; do
    run_eval "reactzyme_${split}" \
      --setting reactzyme \
      --pairs "${DATA_ROOT}/reactzyme/eval/${split}/test_pairs.csv" \
      --reactions "${DATA_ROOT}/reactzyme/eval/${split}/reactions.csv" \
      --candidate-residue-h5 "$REACT_H5" \
      --candidate-ids "${DATA_ROOT}/reactzyme/eval/${split}/candidate_ids.txt" \
      --reactzyme-direction both
  done

  echo "[$(date --iso-8601=seconds)] Paper-setting ESM2 evaluation pipeline complete"
} 2>&1 | tee "$RUN_LOG"
