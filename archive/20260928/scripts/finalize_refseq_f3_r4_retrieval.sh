#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/wet_lab/logs/refseq_f3_r4_retrieval_20260817}"
FINALIZER_LOG="$LOG_ROOT/finalizer.log"

cd "$PROJECT_ROOT"

finish() {
  status=$?
  printf 'FINALIZER_EXIT=%s AT=%s\n' "$status" "$(date -Is)" >> "$FINALIZER_LOG"
}
trap finish EXIT

logs=(
  "$LOG_ROOT/f3_shard_0.log"
  "$LOG_ROOT/f3_shard_1.log"
  "$LOG_ROOT/r4_shard_0.log"
  "$LOG_ROOT/r4_shard_1.log"
)

printf 'Waiting for four retrieval shards at %s\n' "$(date -Is)" > "$FINALIZER_LOG"
while true; do
  completed=0
  for log in "${logs[@]}"; do
    if grep -q '^QUERY_EXIT=' "$log" 2>/dev/null; then
      completed=$((completed + 1))
    fi
  done
  printf '%s completed_shards=%s/4\n' "$(date -Is)" "$completed" >> "$FINALIZER_LOG"
  if [[ "$completed" -eq 4 ]]; then
    break
  fi
  sleep 30
done

for log in "${logs[@]}"; do
  if ! grep -q '^QUERY_EXIT=0 ' "$log"; then
    printf 'Retrieval shard failed: %s\n' "$log" >&2
    exit 1
  fi
done

result_path() {
  sed -n 's/^Saved ranked enzymes to: //p' "$1" | tail -n 1
}

f3_shard_0_result="$(result_path "${logs[0]}")"
f3_shard_1_result="$(result_path "${logs[1]}")"
r4_shard_0_result="$(result_path "${logs[2]}")"
r4_shard_1_result="$(result_path "${logs[3]}")"

f3_merged="$PROJECT_ROOT/wet_lab/runs/refseq/prokaryotes/tagatose_4_epimerase_f3_epoch28_merged"
r4_merged="$PROJECT_ROOT/wet_lab/runs/refseq/prokaryotes/tagatose_4_epimerase_r4_epoch11_merged"

"$PYTHON_BIN" -m wet_lab.merge_query_shards \
  --result "$f3_shard_0_result" \
  --result "$f3_shard_1_result" \
  --output-dir "$f3_merged" >> "$FINALIZER_LOG" 2>&1

"$PYTHON_BIN" -m wet_lab.merge_query_shards \
  --result "$r4_shard_0_result" \
  --result "$r4_shard_1_result" \
  --output-dir "$r4_merged" >> "$FINALIZER_LOG" 2>&1

campaign_config="$PROJECT_ROOT/wet_lab/configs/refseq/f3_r4_consensus.yaml"
"$PYTHON_BIN" -c '
import sys
from pathlib import Path
import yaml
from wet_lab.query_campaign import consolidate_results

campaign = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = consolidate_results(
    reaction=dict(campaign["reaction"]),
    result_paths=[Path(sys.argv[2]), Path(sys.argv[3])],
    output_dir=Path(campaign["output"]["directory"]),
    rrf_constant=float(campaign["consensus"]["rrf_constant"]),
    max_rank=int(campaign["consensus"]["max_rank"]),
)
print(f"Saved consensus results to: {result}")
' "$campaign_config" "$f3_merged/results.json" "$r4_merged/results.json" \
  >> "$FINALIZER_LOG" 2>&1

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" -m wet_lab.fold \
    --config wet_lab/configs/fold_refseq_tagatose_f3_r4_consensus_top10_esmfold.yaml \
    >> "$FINALIZER_LOG" 2>&1
