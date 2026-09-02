#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
CAMPAIGN_NAME="${CAMPAIGN_NAME:-refseq_f3_other_splits_20260828}"
QUERY_CPU_THREADS="${QUERY_CPU_THREADS:-16}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/wet_lab/logs/$CAMPAIGN_NAME}"
SHARD_ROOT="$PROJECT_ROOT/wet_lab/databases/refseq/prokaryotes/query_shards_2"
CASE_ROOT="$PROJECT_ROOT/wet_lab/Case1/refseq_setting/F3_set_chemistry/D-fructose_to_D-tagatose"
RUNTIME_ROOT="$PROJECT_ROOT/wet_lab/runtime/$CAMPAIGN_NAME"
PID_FILE="$PROJECT_ROOT/run_pids/$CAMPAIGN_NAME.pid"

if [[ ! "$QUERY_CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'QUERY_CPU_THREADS must be a positive integer, got %s\n' "$QUERY_CPU_THREADS" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT" "$RUNTIME_ROOT/home" "$RUNTIME_ROOT/tmp" "$(dirname "$PID_FILE")"
if [[ -s "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    printf 'Campaign is already running as PID %s\n' "$existing_pid" >&2
    exit 1
  fi
fi
printf '%s\n' "$$" > "$PID_FILE"
export HOME="$RUNTIME_ROOT/home"
export TMPDIR="$RUNTIME_ROOT/tmp"

cd "$PROJECT_ROOT"

query_logs=(
  "$LOG_ROOT/enzyme_smi_shard_0.log"
  "$LOG_ROOT/enzyme_smi_shard_1.log"
  "$LOG_ROOT/time_shard_0.log"
  "$LOG_ROOT/time_shard_1.log"
)
query_configs=(
  "wet_lab/configs/refseq/f3_enzyme_smi.yaml"
  "wet_lab/configs/refseq/f3_enzyme_smi.yaml"
  "wet_lab/configs/refseq/f3_time.yaml"
  "wet_lab/configs/refseq/f3_time.yaml"
)
query_shards=(0 1 0 1)
query_gpus=(0 1 2 3)

run_query() {
  local index="$1"
  local log="${query_logs[$index]}"
  local config="${query_configs[$index]}"
  local shard="${query_shards[$index]}"
  local gpu="${query_gpus[$index]}"
  local status

  printf 'QUERY_START=%s GPU=%s SHARD=%s CONFIG=%s\n' \
    "$(date -Is)" "$gpu" "$shard" "$config" > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$QUERY_CPU_THREADS" \
    MKL_NUM_THREADS="$QUERY_CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$QUERY_CPU_THREADS" \
    NUMEXPR_NUM_THREADS="$QUERY_CPU_THREADS" \
    WET_LAB_CANDIDATE_POOL_ROOT_OVERRIDE="$SHARD_ROOT/shard_$shard" \
    "$PYTHON_BIN" -m wet_lab.query --config "$config" >> "$log" 2>&1
  status=$?
  printf 'QUERY_EXIT=%s AT=%s\n' "$status" "$(date -Is)" >> "$log"
  return "$status"
}

result_path() {
  sed -n 's/^Saved ranked enzymes to: //p' "$1" | tail -n 1
}

printf 'CAMPAIGN_START=%s\n' "$(date -Is)"
pids=()
for index in 0 1 2 3; do
  run_query "$index" &
  pids+=("$!")
done

failed=0
for index in 0 1 2 3; do
  if ! wait "${pids[$index]}"; then
    printf 'Query failed: %s\n' "${query_logs[$index]}" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  printf 'CAMPAIGN_EXIT=1 AT=%s\n' "$(date -Is)"
  exit 1
fi

enzyme_merged="$PROJECT_ROOT/wet_lab/runs/refseq/prokaryotes/tagatose_4_epimerase_f3_enzyme_smi_epoch28_merged"
time_merged="$PROJECT_ROOT/wet_lab/runs/refseq/prokaryotes/tagatose_4_epimerase_f3_time_epoch26_merged"

"$PYTHON_BIN" -m wet_lab.merge_query_shards \
  --result "$(result_path "${query_logs[0]}")" \
  --result "$(result_path "${query_logs[1]}")" \
  --output-dir "$enzyme_merged"

"$PYTHON_BIN" -m wet_lab.merge_query_shards \
  --result "$(result_path "${query_logs[2]}")" \
  --result "$(result_path "${query_logs[3]}")" \
  --output-dir "$time_merged"

"$PYTHON_BIN" wet_lab/Case1/refseq_setting/compare_top25_structures.py overlap \
  --results "$enzyme_merged/results.json" \
  --output-dir "$CASE_ROOT/enzyme_smi/comparison"

"$PYTHON_BIN" wet_lab/Case1/refseq_setting/compare_top25_structures.py overlap \
  --results "$time_merged/results.json" \
  --output-dir "$CASE_ROOT/time/comparison"

printf 'CAMPAIGN_EXIT=0 AT=%s\n' "$(date -Is)"
