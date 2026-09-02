#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python}"
CAMPAIGN_NAME="${CAMPAIGN_NAME:-refseq_f3mc_ec_unmapped_20260902}"
QUERY_CPU_THREADS="${QUERY_CPU_THREADS:-16}"
SHARD_ROOT="$PROJECT_ROOT/wet_lab/databases/refseq/prokaryotes/query_shards_2"
LOG_ROOT="$PROJECT_ROOT/wet_lab/logs/$CAMPAIGN_NAME"
RUNTIME_ROOT="$PROJECT_ROOT/wet_lab/runtime/$CAMPAIGN_NAME"
MERGED_ROOT="$PROJECT_ROOT/wet_lab/runs/refseq/prokaryotes"
COMPARISON_ROOT="$PROJECT_ROOT/wet_lab/Case1/refseq_setting/F3MC_EC_unmapped"

mkdir -p "$LOG_ROOT" "$RUNTIME_ROOT" "$COMPARISON_ROOT"
cd "$PROJECT_ROOT"

result_path() {
  sed -n 's/^Saved ranked enzymes to: //p' "$1" | tail -n 1
}

run_shard() {
  local model_id="$1"
  local config="$2"
  local shard="$3"
  local gpu="$4"
  local runtime="$RUNTIME_ROOT/${model_id}_shard_${shard}"
  local log="$LOG_ROOT/${model_id}_shard_${shard}.log"

  mkdir -p "$runtime/home" "$runtime/tmp"
  printf 'QUERY_START=%s MODEL=%s GPU=%s SHARD=%s CONFIG=%s\n' \
    "$(date -Is)" "$model_id" "$gpu" "$shard" "$config" > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    HOME="$runtime/home" \
    TMPDIR="$runtime/tmp" \
    OMP_NUM_THREADS="$QUERY_CPU_THREADS" \
    MKL_NUM_THREADS="$QUERY_CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$QUERY_CPU_THREADS" \
    NUMEXPR_NUM_THREADS="$QUERY_CPU_THREADS" \
    TOKENIZERS_PARALLELISM=false \
    WET_LAB_CANDIDATE_POOL_ROOT_OVERRIDE="$SHARD_ROOT/shard_$shard" \
    "$PYTHON_BIN" -m wet_lab.query --config "$config" >> "$log" 2>&1
  local status=$?
  printf 'QUERY_EXIT=%s AT=%s\n' "$status" "$(date -Is)" >> "$log"
  return "$status"
}

run_model() {
  local model_id="$1"
  local config="$2"
  local merged_name="$3"
  local gpu_0="$4"
  local gpu_1="$5"
  local log_0="$LOG_ROOT/${model_id}_shard_0.log"
  local log_1="$LOG_ROOT/${model_id}_shard_1.log"

  run_shard "$model_id" "$config" 0 "$gpu_0" &
  local pid_0=$!
  run_shard "$model_id" "$config" 1 "$gpu_1" &
  local pid_1=$!

  local failed=0
  wait "$pid_0" || failed=1
  wait "$pid_1" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    printf 'MODEL_FAILED=%s AT=%s\n' "$model_id" "$(date -Is)"
    return 1
  fi

  local merged="$MERGED_ROOT/$merged_name"
  "$PYTHON_BIN" -m wet_lab.merge_query_shards \
    --result "$(result_path "$log_0")" \
    --result "$(result_path "$log_1")" \
    --output-dir "$merged"

  "$PYTHON_BIN" wet_lab/Case1/refseq_setting/compare_top25_structures.py overlap \
    --results "$merged/results.json" \
    --output-dir "$COMPARISON_ROOT/$model_id"
  printf 'MODEL_COMPLETE=%s RESULTS=%s AT=%s\n' \
    "$model_id" "$merged/results.json" "$(date -Is)"
}

run_wave() {
  local first_id="$1"
  local first_config="$2"
  local first_merged="$3"
  local second_id="$4"
  local second_config="$5"
  local second_merged="$6"

  run_model "$first_id" "$first_config" "$first_merged" 0 1 &
  local first_pid=$!
  run_model "$second_id" "$second_config" "$second_merged" 2 3 &
  local second_pid=$!
  local failed=0
  wait "$first_pid" || failed=1
  wait "$second_pid" || failed=1
  return "$failed"
}

printf 'CAMPAIGN_START=%s CANDIDATES=3944613\n' "$(date -Is)"

failed=0
run_wave \
  source_collapse \
  wet_lab/configs/refseq/source_collapse_f3mc_ec_unmapped.yaml \
  tagatose_4_epimerase_source_collapse_f3mc_ec_unmapped_merged \
  time \
  wet_lab/configs/refseq/f3mc_ec_time_unmapped.yaml \
  tagatose_4_epimerase_f3mc_ec_time_epoch28_unmapped_merged || failed=1

run_wave \
  enzyme_smi \
  wet_lab/configs/refseq/f3mc_ec_enzyme_smi_unmapped.yaml \
  tagatose_4_epimerase_f3mc_ec_enzyme_smi_epoch29_unmapped_merged \
  reaction_smi \
  wet_lab/configs/refseq/f3mc_ec_reaction_smi_unmapped.yaml \
  tagatose_4_epimerase_f3mc_ec_reaction_smi_epoch29_unmapped_merged || failed=1

printf 'CAMPAIGN_EXIT=%s AT=%s\n' "$failed" "$(date -Is)"
exit "$failed"
