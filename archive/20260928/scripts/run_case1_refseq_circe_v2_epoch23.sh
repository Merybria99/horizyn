#!/usr/bin/env bash
set -eEuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/../.capability-run-py/bin/python}"
CAMPAIGN_NAME="${CAMPAIGN_NAME:-case1_refseq_circe_v2_epoch23}"
QUERY_CPU_THREADS="${QUERY_CPU_THREADS:-16}"
REFSEQ_CONFIG="wet_lab/configs/refseq/circe_v2_epoch23_reaction_smi.yaml"
GROUND_TRUTH_CONFIG="wet_lab/Case1/refseq_setting/F3_set_chemistry/D-fructose_to_D-tagatose/reaction_smi/augmented_ground_truth_23/query_circe_v2_epoch23.yaml"
SHARD_ROOT="$PROJECT_ROOT/wet_lab/databases/refseq/prokaryotes/query_shards_3"
LOG_ROOT="$PROJECT_ROOT/wet_lab/logs/$CAMPAIGN_NAME"
RUNTIME_ROOT="$PROJECT_ROOT/wet_lab/runtime/$CAMPAIGN_NAME"
PID_FILE="$PROJECT_ROOT/run_pids/$CAMPAIGN_NAME.pid"
MERGED_DIR="$PROJECT_ROOT/wet_lab/runs/refseq/prokaryotes/tagatose_4_epimerase_circe_v2_epoch23_merged"
COMPARISON_DIR="$PROJECT_ROOT/wet_lab/Case1/refseq_setting/CIRCE_V2/D-fructose_to_D-tagatose/reaction_smi/refseq_plus_ground_truth_23"

if [[ ! "$QUERY_CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'QUERY_CPU_THREADS must be a positive integer, got %s\n' "$QUERY_CPU_THREADS" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT" "$RUNTIME_ROOT/tmp" "$(dirname "$PID_FILE")" "$COMPARISON_DIR"
if [[ -s "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    printf 'Campaign is already running as PID %s\n' "$existing_pid" >&2
    exit 1
  fi
fi

CAMPAIGN_SHELL_PID="$BASHPID"
printf '%s\n' "$CAMPAIGN_SHELL_PID" > "$PID_FILE"

cleanup() {
  local status=$?
  if [[ "$BASHPID" != "$CAMPAIGN_SHELL_PID" ]]; then
    return
  fi
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE")" == "$CAMPAIGN_SHELL_PID" ]]; then
    rm -f "$PID_FILE"
  fi
  printf 'CAMPAIGN_EXIT=%s AT=%s\n' "$status" "$(date -Is)"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

result_path() {
  local log="$1"
  if [[ ! -f "$log" ]]; then
    return 0
  fi
  sed -n 's/^Saved ranked enzymes to: //p' "$log" | tail -n 1
}

require_result_path() {
  local log="$1"
  local result
  result="$(result_path "$log")"
  if [[ -z "$result" ]] || [[ ! -f "$result" ]]; then
    printf 'No valid query result found in %s\n' "$log" >&2
    return 1
  fi
  printf '%s\n' "$result"
}

run_query() {
  local name="$1"
  local config="$2"
  local gpu="$3"
  local candidate_override="${4:-}"
  local log="$LOG_ROOT/$name.log"
  local completed_result
  local previous_log
  local status
  local -a query_environment

  completed_result="$(result_path "$log")"
  if [[ -f "$log" ]] && grep -q '^QUERY_EXIT=0 ' "$log" && \
    [[ -n "$completed_result" ]] && [[ -f "$completed_result" ]]; then
    printf 'QUERY_REUSE=%s RESULT=%s\n' "$log" "$completed_result"
    return 0
  fi
  if [[ -f "$log" ]]; then
    previous_log="$log.interrupted.$(date +%Y%m%dT%H%M%S)"
    mv "$log" "$previous_log"
    printf 'Preserved interrupted query log: %s\n' "$previous_log"
  fi

  printf 'QUERY_START=%s GPU=%s CONFIG=%s CANDIDATE_OVERRIDE=%s\n' \
    "$(date -Is)" "$gpu" "$config" "${candidate_override:-none}" > "$log"

  query_environment=(
    "CUDA_VISIBLE_DEVICES=$gpu"
    "HORIZYN_CPU_THREADS=$QUERY_CPU_THREADS"
    "OMP_NUM_THREADS=$QUERY_CPU_THREADS"
    "MKL_NUM_THREADS=$QUERY_CPU_THREADS"
    "OPENBLAS_NUM_THREADS=$QUERY_CPU_THREADS"
    "NUMEXPR_NUM_THREADS=$QUERY_CPU_THREADS"
    "TOKENIZERS_PARALLELISM=false"
    "TMPDIR=$RUNTIME_ROOT/tmp"
  )
  if [[ -n "$candidate_override" ]]; then
    query_environment+=("WET_LAB_CANDIDATE_POOL_ROOT_OVERRIDE=$candidate_override")
  fi

  if env "${query_environment[@]}" \
      "$PYTHON_BIN" -m wet_lab.query --config "$config" >> "$log" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf 'QUERY_EXIT=%s AT=%s\n' "$status" "$(date -Is)" >> "$log"
  return "$status"
}

cd "$PROJECT_ROOT"
printf 'CAMPAIGN_START=%s\n' "$(date -Is)"

# Score the 23 unique recoverable members of the workbook's 25-entry set first.
# This also materializes the reaction features before the long RefSeq jobs begin.
run_query "ground_truth_23" "$GROUND_TRUTH_CONFIG" 3

declare -a pids=()
for shard in 0 1 2; do
  run_query \
    "refseq_shard_$shard" \
    "$REFSEQ_CONFIG" \
    "$((shard + 1))" \
    "$SHARD_ROOT/shard_$shard" &
  pids+=("$!")
done

failed=0
for shard in 0 1 2; do
  if ! wait "${pids[$shard]}"; then
    printf 'Query failed: %s\n' "$LOG_ROOT/refseq_shard_$shard.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

declare -a shard_results=()
for shard in 0 1 2; do
  shard_results+=(--result "$(require_result_path "$LOG_ROOT/refseq_shard_$shard.log")")
done
"$PYTHON_BIN" -m wet_lab.merge_query_shards \
  "${shard_results[@]}" \
  --output-dir "$MERGED_DIR"

"$PYTHON_BIN" wet_lab/Case1/refseq_setting/ground_truth_augmented.py analyze \
  --ground-truth-results "$(require_result_path "$LOG_ROOT/ground_truth_23.log")" \
  --refseq-results "$MERGED_DIR/results.json" \
  --output-dir "$COMPARISON_DIR"

printf 'CAMPAIGN_COMPLETE=%s MERGED_RESULTS=%s COMPARISON=%s\n' \
  "$(date -Is)" "$MERGED_DIR/results.json" "$COMPARISON_DIR/summary.json"
