#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENV_PATH="${ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"
RUN_NAME="paper_setting_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT=""
CONFIG=""
BENCHMARK="all"
PROTEIN_EMBEDDING="prott5"
DATA_ROOT="data/paper"
TARGET_BATCH_SIZE="512"
QUERY_BATCH_SIZE="128"
TARGET_ENCODE_DEVICES="auto"
STORE_TARGETS_ON_CPU="false"
REACTZYME_DIRECTION="both"
REACTION_EMBEDS_H5=""
VALIDATE_ONLY="false"

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
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --protein-embedding)
      PROTEIN_EMBEDDING="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
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
    --target-encode-devices)
      TARGET_ENCODE_DEVICES="$2"
      shift 2
      ;;
    --reaction-embeds-h5)
      REACTION_EMBEDS_H5="$2"
      shift 2
      ;;
    --reactzyme-direction)
      REACTZYME_DIRECTION="$2"
      shift 2
      ;;
    --store-targets-on-cpu)
      STORE_TARGETS_ON_CPU="true"
      shift
      ;;
    --validate-only)
      VALIDATE_ONLY="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CHECKPOINT" || -z "$CONFIG" ]]; then
  echo "--checkpoint and --config are required" >&2
  exit 2
fi

case "$BENCHMARK" in
  all|reactzyme_time|reactzyme_enzyme_smi|reactzyme_reaction_smi|clipzyme_enzymemap) ;;
  *)
    echo "Unsupported --benchmark: $BENCHMARK" >&2
    exit 2
    ;;
esac

case "$PROTEIN_EMBEDDING" in
  prott5|esm2) ;;
  *)
    echo "Unsupported --protein-embedding: $PROTEIN_EMBEDDING" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
source "$ENV_PATH/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1

LOG_DIR="logs/${RUN_NAME}"
RESULT_DIR="results/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run.log"
mkdir -p "$LOG_DIR" "$RESULT_DIR"
printf '%s\n' "$$" > "${LOG_DIR}/run.pid"

if [[ "$BENCHMARK" == "all" ]]; then
  BENCHMARKS=(
    reactzyme_time
    reactzyme_enzyme_smi
    reactzyme_reaction_smi
    clipzyme_enzymemap
  )
else
  BENCHMARKS=("$BENCHMARK")
fi

run_eval() {
  local benchmark="$1"
  local output="${RESULT_DIR}/${benchmark}.json"
  local log="${LOG_DIR}/${benchmark}.log"
  local args=(
    scripts/evaluate_paper_setting.py
    --benchmark "$benchmark"
    --data-root "$DATA_ROOT"
    --protein-embedding "$PROTEIN_EMBEDDING"
    --checkpoint "$CHECKPOINT"
    --config "$CONFIG"
    --device cuda
    --batch-size "$QUERY_BATCH_SIZE"
    --target-batch-size "$TARGET_BATCH_SIZE"
    --target-encode-devices "$TARGET_ENCODE_DEVICES"
    --reactzyme-direction "$REACTZYME_DIRECTION"
    --output "$output"
  )
  if [[ "$STORE_TARGETS_ON_CPU" == "true" ]]; then
    args+=(--store-targets-on-cpu)
  fi
  if [[ "$VALIDATE_ONLY" == "true" ]]; then
    args+=(--validate-only)
  fi
  if [[ -n "$REACTION_EMBEDS_H5" ]]; then
    args+=(--reaction-embeds-h5 "$REACTION_EMBEDS_H5")
  fi

  echo "[$(date --iso-8601=seconds)] Running ${benchmark}"
  "$PYTHON_BIN" "${args[@]}" > "$log" 2>&1
  echo "[$(date --iso-8601=seconds)] Finished ${benchmark} -> ${output}"
}

{
  echo "[$(date --iso-8601=seconds)] Starting paper-setting evaluation pipeline"
  echo "Project: $PROJECT_ROOT"
  echo "Python: $PYTHON_BIN"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Checkpoint: $CHECKPOINT"
  echo "Config: $CONFIG"
  echo "Benchmark: $BENCHMARK"
  echo "Protein embedding: $PROTEIN_EMBEDDING"
  echo "Data root: $DATA_ROOT"
  echo "Result dir: $RESULT_DIR"

  for benchmark in "${BENCHMARKS[@]}"; do
    run_eval "$benchmark"
  done

  echo "[$(date --iso-8601=seconds)] Paper-setting evaluation pipeline complete"
} 2>&1 | tee "$RUN_LOG"
